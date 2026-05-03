"""
QumulatorSimulator â€” Cirq adapter for the Qumulator service.

Converts Cirq circuits to the Qumulator gate format and executes them
via the Qumulator API.  All simulation runs server-side.

Install cirq before using this module::

    pip install cirq

Usage
-----
::

    import os
    import cirq
    from qumulator import QumulatorClient
    from qumulator.backends.cirq_simulator import QumulatorSimulator

    client = QumulatorClient(
        api_url=os.environ["QUMULATOR_API_URL"],
        api_key=os.environ["QUMULATOR_API_KEY"],
    )
    sim = QumulatorSimulator(client)

    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.H(q0),
        cirq.CNOT(q0, q1),
        cirq.measure(q0, q1, key='result'),
    )

    # Sample-based execution
    result = sim.run(circuit, repetitions=1024)
    print(result.histogram(key='result'))   # Counter({0: ~512, 3: ~512})

    # Exact statevector
    sv_result = sim.simulate(circuit)
    print(sv_result.final_state_vector)     # [1/sqrt(2), 0, 0, 1/sqrt(2)]

Notes
-----
*  All Cirq gates are submitted as unitary matrices via :func:`cirq.unitary`,
   which means any gate that has a unitary representation is supported.
*  Qubit ordering: Cirq sorts qubits and treats the first qubit as the
   most-significant bit, which matches the Qumulator convention.
*  Measurement gates determine which qubits appear in sample results.
   Circuits without measurement gates are handled with a default key ``'m'``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    import cirq
    _CIRQ_OK = True
except ImportError:
    _CIRQ_OK = False


def _require() -> None:
    if not _CIRQ_OK:
        raise ImportError(
            "cirq is required for QumulatorSimulator.  "
            "Install with:  pip install cirq"
        )


# ---------------------------------------------------------------------------
#  Lightweight result wrappers
# ---------------------------------------------------------------------------


class _QumulatorStateVectorResult:
    """Minimal statevector result compatible with cirq.SimulationResult."""

    def __init__(
        self,
        circuit: "cirq.Circuit",
        final_state_vector: Optional[np.ndarray],
        qubit_order: List["cirq.Qid"],
    ) -> None:
        self.circuit = circuit
        self.final_state_vector = final_state_vector
        self.qubit_map = {q: i for i, q in enumerate(qubit_order)}

    @property
    def n_qubits(self) -> int:
        return int(np.log2(len(self.final_state_vector)))

    def probabilities(self) -> np.ndarray:
        return np.abs(self.final_state_vector) ** 2

    def dirac_notation(self, decimals: int = 3) -> str:
        n = self.n_qubits
        terms = []
        for idx, amp in enumerate(self.final_state_vector):
            if abs(amp) > 0.5 ** (decimals + 1):
                bs = format(idx, f"0{n}b")
                r, i = round(amp.real, decimals), round(amp.imag, decimals)
                if i == 0:
                    terms.append(f"{r}|{bs}>")
                elif r == 0:
                    terms.append(f"{i}j|{bs}>")
                else:
                    terms.append(f"({r}+{i}j)|{bs}>")
        return " + ".join(terms) or "0"

    def __repr__(self) -> str:
        return f"_QumulatorStateVectorResult({self.dirac_notation()})"


class _QumulatorSampleResult:
    """Sample-based result with histogram() and measurements[] access."""

    def __init__(
        self,
        measurements: Dict[str, np.ndarray],
        repetitions: int,
    ) -> None:
        self.measurements = measurements
        self.repetitions = repetitions

    def histogram(self, key: str, fold_func=None):
        """Return a Counter mapping integer measurement outcomes to counts."""
        from collections import Counter
        data = self.measurements[key]
        counts: Counter = Counter()
        for row in data:
            outcome = fold_func(row) if fold_func is not None else int(
                "".join(str(b) for b in row), 2
            )
            counts[outcome] += 1
        return counts

    def __repr__(self) -> str:
        return (
            f"_QumulatorSampleResult(keys={list(self.measurements)}, "
            f"repetitions={self.repetitions})"
        )


# ---------------------------------------------------------------------------
#  QumulatorSimulator
# ---------------------------------------------------------------------------


class QumulatorSimulator:
    """
    Cirq-compatible simulator backed by the Qumulator service.

    Any Cirq circuit built from standard gates is supported.  Gates without
    a named Qumulator equivalent are submitted as arbitrary unitary matrices
    via :func:`cirq.unitary`.

    Parameters
    ----------
    client : QumulatorClient or CircuitClient
        Authenticated Qumulator client.  Pass ``client`` or
        ``client.circuit`` â€” both are accepted.
    seed : int or numpy.random.Generator, optional
        Default RNG seed for :meth:`run`.

    Examples
    --------
    ::

        sim = QumulatorSimulator(client)

        q = cirq.LineQubit.range(3)
        ghz = cirq.Circuit(
            cirq.H(q[0]), cirq.CNOT(q[0], q[1]), cirq.CNOT(q[0], q[2])
        )
        sv = sim.simulate(ghz)
        print(sv.dirac_notation())   # 0.707|000> + 0.707|111>

        ghz.append(cirq.measure(*q, key='m'))
        res = sim.run(ghz, repetitions=2000)
        print(res.histogram(key='m'))
    """

    def __init__(
        self,
        client: Any = None,
        seed: Optional[Union[int, "np.random.Generator"]] = None,
    ) -> None:
        _require()
        # Accept either a QumulatorClient or a CircuitClient directly.
        from qumulator.circuit import CircuitClient
        if client is None:
            import os
            from qumulator import QumulatorClient
            api_key = os.environ.get("QUMULATOR_API_KEY", "")
            api_url = os.environ.get("QUMULATOR_API_URL", "https://api.qumulator.com")
            if not api_key:
                raise ValueError(
                    "No API key found. Set QUMULATOR_API_KEY or pass a QumulatorClient."
                )
            client = QumulatorClient(api_url=api_url, api_key=api_key)
        if hasattr(client, "circuit"):
            self._circuit_client: CircuitClient = client.circuit
        else:
            self._circuit_client = client
        self._seed = seed

    # -- internal conversion --------------------------------------------------

    def _circuit_to_gates(
        self,
        circuit: "cirq.Circuit",
        qubit_order: List["cirq.Qid"],
    ) -> List[Dict]:
        """Convert a Cirq circuit into the Qumulator gate-list format."""
        qubit_idx = {q: i for i, q in enumerate(qubit_order)}
        gates = []
        for moment in circuit:
            for op in moment.operations:
                if isinstance(op.gate, cirq.MeasurementGate):
                    continue  # measurement handled separately
                try:
                    U = cirq.unitary(op)
                except (TypeError, ValueError) as exc:
                    raise NotImplementedError(
                        f"Operation {op!r} does not expose a unitary and "
                        f"cannot be submitted: {exc}"
                    ) from exc
                q_indices = [qubit_idx[q] for q in op.qubits]
                # Serialise the complex matrix as [[real, imag], ...]
                flat = U.ravel().astype(complex)
                matrix = [[float(v.real), float(v.imag)] for v in flat]
                gates.append({
                    "gate": "unitary",
                    "qubits": q_indices,
                    "params": [matrix],
                })
        return gates

    def _measurement_keys(
        self, circuit: "cirq.Circuit", qubit_order: List["cirq.Qid"]
    ) -> Dict[str, List[int]]:
        """Return {measurement_key: [qubit_indices]} from the circuit."""
        qubit_idx = {q: i for i, q in enumerate(qubit_order)}
        keys: Dict[str, List[int]] = {}
        for moment in circuit:
            for op in moment.operations:
                if isinstance(op.gate, cirq.MeasurementGate):
                    key = op.gate.key
                    keys[key] = [qubit_idx[q] for q in op.qubits]
        return keys

    # -- public API -----------------------------------------------------------

    def simulate(
        self,
        circuit: "cirq.Circuit",
        qubit_order: Optional[List["cirq.Qid"]] = None,
    ) -> _QumulatorStateVectorResult:
        """
        Compute the final statevector of a circuit.

        Parameters
        ----------
        circuit : cirq.Circuit
        qubit_order : list of cirq.Qid, optional
            Qubit ordering.  Defaults to ``sorted(circuit.all_qubits())``.

        Returns
        -------
        _QumulatorStateVectorResult
        """
        _require()
        if qubit_order is None:
            qubit_order = sorted(circuit.all_qubits())
        gates = self._circuit_to_gates(circuit, qubit_order)
        result = self._circuit_client.run(
            gates=gates,
            n_qubits=len(qubit_order),
            return_statevector=True,
        )
        return _QumulatorStateVectorResult(
            circuit, result.statevector, qubit_order
        )

    def run(
        self,
        circuit: "cirq.Circuit",
        repetitions: int = 1024,
        qubit_order: Optional[List["cirq.Qid"]] = None,
    ) -> _QumulatorSampleResult:
        """
        Sample the circuit the specified number of times.

        Measurement gates determine which qubits and keys appear in the
        result.  If no measurement gates are present, all qubits are
        measured under a default key ``'m'``.

        Parameters
        ----------
        circuit : cirq.Circuit
        repetitions : int
        qubit_order : list of cirq.Qid, optional

        Returns
        -------
        _QumulatorSampleResult
        """
        _require()
        if qubit_order is None:
            qubit_order = sorted(circuit.all_qubits())
        n = len(qubit_order)

        gates = self._circuit_to_gates(circuit, qubit_order)
        mkeys = self._measurement_keys(circuit, qubit_order)
        if not mkeys:
            mkeys = {"m": list(range(n))}

        seed = self._seed if isinstance(self._seed, int) else None
        result = self._circuit_client.run(
            gates=gates,
            n_qubits=n,
            shots=repetitions,
            seed=seed,
        )

        counts = result.counts  # {bitstring: count}

        # Build per-key measurements array
        measurements: Dict[str, np.ndarray] = {}
        shots_list: List[List[int]] = []
        for bs, cnt in counts.items():
            bits = [int(b) for b in bs]
            shots_list.extend([bits] * cnt)

        all_bits = np.array(shots_list, dtype=np.uint8)  # (total_shots, n)

        for key, q_indices in mkeys.items():
            measurements[key] = all_bits[:, q_indices]

        return _QumulatorSampleResult(measurements, repetitions)
