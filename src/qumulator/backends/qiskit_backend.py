"""
QumulatorBackend â€” Qiskit adapter for the Qumulator service.

Converts Qiskit circuits to the Qumulator gate format and executes them
via the Qumulator API.  All simulation runs server-side.

Install qiskit before using this module::

    pip install qiskit>=1.0

Usage
-----
::

    import os
    from qumulator import QumulatorClient
    from qumulator.backends.qiskit_backend import QumulatorBackend
    from qiskit import QuantumCircuit

    client = QumulatorClient(
        api_url=os.environ["QUMULATOR_API_URL"],
        api_key=os.environ["QUMULATOR_API_KEY"],
    )
    backend = QumulatorBackend(client)

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    job    = backend.run(qc, shots=2048)
    result = job.result()
    counts = result.get_counts()   # {'00': ~1024, '11': ~1024}

Notes
-----
*  Any Qiskit circuit is automatically transpiled to the supported basis-gate
   set before submission.  Gates outside the basis set that expose
   ``gate.to_matrix()`` are submitted as arbitrary unitary matrices.

*  Qubit-bit ordering: Qiskit's convention is that qubit 0 is the
   least-significant bit in the output bitstring.  This adapter reverses
   bitstrings at output so ``get_counts()`` returns Qiskit-standard keys.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    from qiskit import transpile
    from qiskit.circuit import QuantumCircuit
    _QISKIT_OK = True
except ImportError:
    _QISKIT_OK = False


def _require() -> None:
    if not _QISKIT_OK:
        raise ImportError(
            "qiskit is required for QumulatorBackend.  "
            "Install with:  pip install qiskit"
        )


# ---------------------------------------------------------------------------
#  Minimal Job / Result wrappers  (Qiskit-compatible)
# ---------------------------------------------------------------------------


class _QumulatorResult:
    """Lightweight result object compatible with the Qiskit Result API."""

    def __init__(
        self,
        results: List[Dict],
        backend_name: str = "QumulatorBackend",
        job_id: str = "",
    ) -> None:
        self._results = results
        self.backend_name = backend_name
        self.job_id = job_id
        self.success = all(r["success"] for r in results)
        self.status = "COMPLETED" if self.success else "FAILED"

    def _idx(self, experiment: Union[int, "QuantumCircuit", str]) -> int:
        if isinstance(experiment, int):
            return experiment
        if isinstance(experiment, str):
            for i, r in enumerate(self._results):
                if r.get("name") == experiment:
                    return i
        return 0

    def get_counts(
        self, experiment: Union[int, "QuantumCircuit", str] = 0
    ) -> Dict[str, int]:
        """Return measurement outcome counts as ``{bitstring: count}``."""
        return dict(self._results[self._idx(experiment)]["counts"])

    def get_memory(
        self, experiment: Union[int, "QuantumCircuit", str] = 0
    ) -> List[str]:
        """Return individual shot outcomes as a list of bitstrings."""
        counts = self.get_counts(experiment)
        memory: List[str] = []
        for bs, n in counts.items():
            memory.extend([bs] * n)
        return memory

    def get_statevector(
        self, experiment: Union[int, "QuantumCircuit", str] = 0
    ) -> Optional[np.ndarray]:
        """Return the final statevector (if ``save_statevector=True``)."""
        return self._results[self._idx(experiment)].get("statevector")

    def __repr__(self) -> str:
        return (
            f"_QumulatorResult(backend='{self.backend_name}', "
            f"circuits={len(self._results)}, success={self.success})"
        )


class _QumulatorJob:
    """Synchronous job: the result is available immediately."""

    def __init__(self, result: _QumulatorResult) -> None:
        self._result = result
        self._job_id = result.job_id

    def job_id(self) -> str:
        return self._job_id

    def status(self) -> str:
        return "DONE"

    def done(self) -> bool:
        return True

    def result(self, timeout: Optional[float] = None) -> _QumulatorResult:
        return self._result

    def __repr__(self) -> str:
        return f"_QumulatorJob(id='{self._job_id}', status='DONE')"


# ---------------------------------------------------------------------------
#  QumulatorBackend
# ---------------------------------------------------------------------------

# Named gates natively supported by the Qumulator service.
_SUPPORTED_GATES = [
    "h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx", "sxdg", "id",
    "rx", "ry", "rz", "p", "u", "u1", "u2", "u3",
    "cx", "cy", "cz", "ch", "csx", "swap", "iswap", "ecr",
    "cp", "crx", "cry", "crz", "rxx", "ryy", "rzz",
    "ccx", "cswap",
    "reset", "barrier", "measure",
]


class QumulatorBackend:
    """
    Qiskit-compatible backend backed by the Qumulator service.

    Drop-in replacement for ``qiskit_aer.AerSimulator`` for running
    Qiskit circuits via the Qumulator API.  All simulation is server-side.

    Parameters
    ----------
    client : QumulatorClient or CircuitClient
        Authenticated Qumulator client.
    max_qubits : int
        Maximum number of qubits enforced client-side.

    Examples
    --------
    ::

        backend = QumulatorBackend(client)
        qc = QuantumCircuit(3)
        qc.h(range(3))
        qc.measure_all()
        counts = backend.run(qc, shots=8192).result().get_counts()
    """

    name: str = "QumulatorBackend"
    backend_version: str = "1.0.0"
    description: str = "Quantum circuit simulator via the Qumulator API"

    def __init__(self, client: Any = None, max_qubits: int = 50) -> None:
        _require()
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
        self._max_qubits = max_qubits

    @property
    def max_circuits(self) -> int:
        return 1000

    @property
    def num_qubits(self) -> int:
        return self._max_qubits

    # -- execution -----------------------------------------------------------

    def run(
        self,
        run_input: Union["QuantumCircuit", List["QuantumCircuit"]],
        **options: Any,
    ) -> _QumulatorJob:
        """
        Execute one or more circuits and return a completed job.

        Parameters
        ----------
        run_input : QuantumCircuit or list of QuantumCircuit
        shots : int (keyword, default 1024)
        seed : int (keyword, optional)
            Alias: ``seed_simulator``.
        save_statevector : bool (keyword, default False)
            Store the final statevector in the result.

        Returns
        -------
        _QumulatorJob
            Call ``.result()`` immediately â€” the job is synchronous.
        """
        _require()
        shots = int(options.get("shots", 1024))
        seed = options.get("seed", options.get("seed_simulator"))
        save_sv = bool(options.get("save_statevector", False))

        circuits = run_input if isinstance(run_input, list) else [run_input]

        transpiled = transpile(
            circuits,
            basis_gates=[g for g in _SUPPORTED_GATES
                         if g not in ("reset", "barrier", "measure")],
            optimization_level=0,
        )
        if not isinstance(transpiled, list):
            transpiled = [transpiled]

        results: List[Dict] = []
        for circ in transpiled:
            n = circ.num_qubits
            if n > self._max_qubits:
                raise ValueError(
                    f"Circuit has {n} qubits but max_qubits={self._max_qubits}"
                )
            gates, measure_map = _extract_qiskit_gates(circ)

            api_result = self._circuit_client.run(
                gates=gates,
                n_qubits=n,
                shots=shots,
                seed=seed,
                return_statevector=save_sv,
            )

            # Reverse bitstrings to Qiskit LSB-first convention.
            qiskit_counts = {
                _to_qiskit_bitstring(bs, measure_map, n): cnt
                for bs, cnt in api_result.counts.items()
            }

            entry: Dict[str, Any] = {
                "success": True,
                "name": getattr(circ, "name", ""),
                "counts": qiskit_counts,
            }
            if save_sv and api_result.statevector is not None:
                entry["statevector"] = api_result.statevector
            results.append(entry)

        job_id = str(uuid.uuid4())
        return _QumulatorJob(_QumulatorResult(results, job_id=job_id))


# ---------------------------------------------------------------------------
#  Qiskit circuit -> gate list conversion
# ---------------------------------------------------------------------------


def _extract_qiskit_gates(
    circ: "QuantumCircuit",
) -> tuple[List[Dict], Dict[int, int]]:
    """
    Walk a transpiled Qiskit circuit and produce (gate_list, measure_map).

    Returns
    -------
    gates : list of gate dicts for the Qumulator API
    measure_map : {qubit_index: clbit_index} for all measure instructions
    """
    gates: List[Dict] = []
    measure_map: Dict[int, int] = {}

    for instr in circ.data:
        op = instr.operation
        name = op.name

        if name == "barrier":
            continue

        if name == "measure":
            q_idx = circ.find_bit(instr.qubits[0]).index
            c_idx = circ.find_bit(instr.clbits[0]).index
            measure_map[q_idx] = c_idx
            continue

        q_indices = [circ.find_bit(q).index for q in instr.qubits]
        params = list(op.params)

        if name == "reset":
            # Represent reset as a projective "reset" hint to the server.
            gates.append({"gate": "reset", "qubits": q_indices})
            continue

        # Try named gate first.
        entry: Dict[str, Any] = {"gate": name, "qubits": q_indices}
        if params:
            entry["params"] = [float(p) for p in params]

        # Fall back to explicit unitary for unknown gates.
        if name not in _SUPPORTED_GATES:
            try:
                U = op.to_matrix()
            except Exception:
                raise NotImplementedError(
                    f"Gate '{name}' is not in the supported gate set and does "
                    "not expose to_matrix().  Transpile the circuit first or "
                    "use a supported gate."
                )
            flat = U.ravel().astype(complex)
            matrix = [[float(v.real), float(v.imag)] for v in flat]
            entry = {
                "gate": "unitary",
                "qubits": q_indices,
                "params": [matrix],
            }

        gates.append(entry)

    return gates, measure_map


def _to_qiskit_bitstring(
    api_bs: str, measure_map: Dict[int, int], n_qubits: int
) -> str:
    """
    Convert an MSB-first bitstring from the Qumulator API to a Qiskit
    LSB-first classical register string.

    If no explicit measure_map is present (full measurement), just reverse
    the bitstring.
    """
    if not measure_map:
        return api_bs[::-1]

    n_clbits = max(measure_map.values()) + 1 if measure_map else n_qubits
    clbits = ["0"] * n_clbits
    for q_idx, c_idx in measure_map.items():
        # api_bs is MSB-first: qubit 0 is index 0.
        if q_idx < len(api_bs):
            clbits[c_idx] = api_bs[q_idx]
    # Qiskit convention: clbit 0 is rightmost character.
    return "".join(reversed(clbits))
