"""
Qumulator Circuit Client — quantum circuit execution via the Qumulator API.

Submit quantum gate circuits to the Qumulator service and retrieve
measurement counts, statevectors, probabilities, or entropy diagnostics.

Quickstart
----------
::

    from qumulator import QumulatorClient
    import os

    client = QumulatorClient(
        api_url=os.environ["QUMULATOR_API_URL"],
        api_key=os.environ["QUMULATOR_API_KEY"],
    )

    # Fluent builder API
    eng = client.circuit.engine(n_qubits=2)
    eng.apply('h', 0).apply('cx', [0, 1])
    print(eng.sample(shots=1024))         # {'00': ~512, '11': ~512}

    # Gate-list API
    result = client.circuit.run(
        gates=[('h', 0), ('cx', [0, 1])],
        n_qubits=2,
        shots=2048,
        return_entropy_map=True,
    )
    print(result.entropy_map)             # [~1.0, ~1.0]

Execution modes
---------------
``'auto'``         Server selects the optimal backend for your circuit.
``'exact'``        Full statevector. Correct for any circuit. N <= ~25.
``'compressed'``   Memory-efficient. Suited for large N, low-to-moderate
                   entanglement (VQE, QAOA, chemistry).
``'tensor'``       Tensor-network backend. Efficient for structured and
                   1D circuits. Supports N > 50.
``'hamiltonian'``  Direct Hamiltonian evolution without gate decomposition.
                   Use with :meth:`CircuitEngine.evolve_hamiltonian`.
``'gaussian'``     Gaussian covariance matrix simulation. Exact for Clifford
                   circuits; principled approximation for non-Clifford content.
                   Returns a :class:`~qumulator.models.GaussianCertificate` in
                   ``result.gaussian_certificate`` classifying the circuit as
                   simulable, likely Gaussian, or requiring a correction.
                   Memory scales as O(n²) instead of O(2ⁿ).
"""
from __future__ import annotations

import dataclasses
import time as _time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from qumulator._http import _BaseClient, QumulatorHTTPError
from qumulator.models import GaussianCertificate

# ---------------------------------------------------------------------------
#  Mode translation table
# ---------------------------------------------------------------------------

_MODE_MAP: Dict[str, str] = {
    "auto":        "klt_nexus_graph",
    "exact":       "statevector",
    "compressed":  "klt_cluster_mps",
    "tensor":      "klt_mps",
    "hamiltonian": "klt_stone",
    "gaussian":    "klt_gaussian",
}


# ---------------------------------------------------------------------------
#  Result dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class CircuitResult:
    """Result returned by a circuit execution request."""

    counts: Dict[str, int]
    """Measurement outcome counts, e.g. ``{'00': 512, '11': 512}``."""

    n_qubits: int
    shots: int

    statevector: Optional[np.ndarray] = None
    """Complex amplitude vector of length 2^N; populated when
    ``return_statevector=True`` is requested."""

    probabilities: Optional[np.ndarray] = None
    """Probability vector of length 2^N; populated when
    ``return_probabilities=True`` is requested."""

    entropy_map: Optional[List[float]] = None
    """Per-qubit entanglement entropy in bits; populated when
    ``return_entropy_map=True`` is requested.  Values near 1 indicate
    high entanglement."""

    gaussian_certificate: Optional[GaussianCertificate] = None
    """Populated when ``mode='gaussian'`` is used.  Contains the circuit
    classification and entanglement regime diagnostics."""

    @property
    def most_probable(self) -> str:
        """Most probable measurement outcome bitstring."""
        return max(self.counts, key=self.counts.__getitem__)


# ---------------------------------------------------------------------------
#  Fluent circuit builder
# ---------------------------------------------------------------------------


class CircuitEngine:
    """
    Fluent quantum circuit builder.

    Accumulates gate operations locally and submits them to the
    Qumulator service when :meth:`run` or :meth:`sample` is called.
    No simulation is performed client-side.

    Parameters
    ----------
    client : CircuitClient
        Parent client used to submit the circuit.
    n_qubits : int
        Number of qubits in the circuit.
    mode : str, optional
        Execution mode.  See :class:`CircuitClient` for available modes.
    bond_dim : int, optional
        Bond-dimension cap for ``'tensor'`` mode.

    Examples
    --------
    Bell state::

        eng = client.circuit.engine(n_qubits=2)
        eng.apply('h', 0).apply('cx', [0, 1])
        print(eng.sample(shots=1024))   # {'00': ~512, '11': ~512}

    Run and retrieve statevector::

        eng = client.circuit.engine(n_qubits=3)
        eng.apply('h', 0).apply('cx', [0, 1]).apply('cx', [0, 2])
        result = eng.run(shots=4096, return_statevector=True)
        print(result.statevector)
    """

    def __init__(
        self,
        client: "CircuitClient",
        n_qubits: int,
        mode: str = "auto",
        bond_dim: Optional[int] = None,
    ) -> None:
        self._client = client
        self.n_qubits = int(n_qubits)
        self.mode = mode
        self.bond_dim = bond_dim
        self._gates: List[Dict] = []

    # -- gate API -------------------------------------------------------------

    def apply(
        self,
        gate_name: str,
        qubits: Union[int, Sequence[int]],
        params: Optional[Sequence[Any]] = None,
    ) -> "CircuitEngine":
        """
        Append a gate to the circuit.

        Parameters
        ----------
        gate_name : str
            Gate identifier.  Common values: ``'h'``, ``'x'``, ``'y'``,
            ``'z'``, ``'s'``, ``'t'``, ``'rx'``, ``'ry'``, ``'rz'``,
            ``'cx'``, ``'cz'``, ``'swap'``, ``'ccx'``, ``'unitary'``.
        qubits : int or list of int
            Target qubit index or list of qubit indices.
        params : list, optional
            Gate parameters: rotation angles for parametric gates; a 2-D
            complex matrix for ``'unitary'``.

        Returns
        -------
        self
            Enables method chaining: ``eng.apply(...).apply(...)``.
        """
        if isinstance(qubits, (int, np.integer)):
            qubits = [int(qubits)]
        entry: Dict[str, Any] = {
            "gate": gate_name,
            "qubits": [int(q) for q in qubits],
        }
        if params is not None:
            entry["params"] = _serialise_params(params)
        self._gates.append(entry)
        return self

    def reset(self) -> "CircuitEngine":
        """Clear all accumulated gates (returns self for chaining)."""
        self._gates = []
        return self

    # -- execution API --------------------------------------------------------

    def run(
        self,
        shots: int = 1024,
        seed: Optional[int] = None,
        return_statevector: bool = False,
        return_probabilities: bool = False,
        return_entropy_map: bool = False,
    ) -> CircuitResult:
        """
        Submit the circuit to the Qumulator service and return results.

        Parameters
        ----------
        shots : int
            Number of measurement samples.
        seed : int, optional
            RNG seed for reproducible sampling.
        return_statevector : bool
            Include the final statevector in the response (N <= ~25).
        return_probabilities : bool
            Include the probability vector in the response (N <= ~25).
        return_entropy_map : bool
            Include per-qubit entanglement entropies in the response.

        Returns
        -------
        CircuitResult
        """
        return self._client._execute(
            n_qubits=self.n_qubits,
            gates=self._gates,
            mode=self.mode,
            bond_dim=self.bond_dim,
            shots=shots,
            seed=seed,
            return_statevector=return_statevector,
            return_probabilities=return_probabilities,
            return_entropy_map=return_entropy_map,
        )

    def sample(
        self,
        shots: int = 1024,
        seed: Optional[int] = None,
    ) -> Dict[str, int]:
        """
        Sample measurement outcomes.

        Returns ``{bitstring: count}`` where bitstrings are MSB-first
        (qubit 0 is the leftmost character).
        """
        return self.run(shots=shots, seed=seed).counts

    def evolve_hamiltonian(
        self,
        pauli_terms: List[Tuple],
        t: float = 1.0,
    ) -> "CircuitEngine":
        """
        Append a direct Hamiltonian evolution step (``'hamiltonian'`` mode).

        Evolves the state by U = e^{-iHt} where H is a weighted sum of
        Pauli-string terms, without requiring gate decomposition.

        Parameters
        ----------
        pauli_terms : list of (coefficient, pauli_string) tuples
            e.g. ``[(0.5, 'ZZ'), (-0.3, 'XI'), (1.0, 'IZ')]``
            Pauli strings use ``I``, ``X``, ``Y``, ``Z`` per qubit
            (leftmost character = qubit 0).
        t : float
            Evolution time (default ``1.0``).

        Returns
        -------
        self
        """
        self._gates.append({
            "gate": "hamiltonian_evolution",
            "pauli_terms": [
                {"coefficient": float(c), "pauli": str(p)}
                for c, p in pauli_terms
            ],
            "t": float(t),
        })
        return self


# ---------------------------------------------------------------------------
#  Circuit HTTP client
# ---------------------------------------------------------------------------


class CircuitClient(_BaseClient):
    """
    Client for quantum circuit execution.

    Obtain via ``QumulatorClient.circuit`` or instantiate directly.

    Parameters
    ----------
    api_url : str
        Base URL of the Qumulator service.
    api_key : str
        API key for authentication.

    Execution modes
    ---------------
    ``'auto'``         Server selects the optimal backend automatically.
    ``'exact'``        Full statevector simulation. Correct for any circuit.
                       Practical for N <= ~25 qubits.
    ``'compressed'``   Compressed representation. Efficient for large N
                       with low-to-moderate entanglement (VQE, QAOA, chemistry).
    ``'tensor'``       Tensor-network backend. Efficient for 1D and shallow
                       circuits. Supports N > 50 at tunable fidelity.
    ``'hamiltonian'``  Direct Hamiltonian evolution. Use
                       :meth:`CircuitEngine.evolve_hamiltonian`; does not
                       require gate decomposition.
    ``'gaussian'``     Gaussian covariance matrix simulation (O(n²) memory).
                       Exact for Clifford circuits. Returns a
                       :class:`~qumulator.models.GaussianCertificate`.
    """

    def engine(
        self,
        n_qubits: int,
        mode: str = "auto",
        bond_dim: Optional[int] = None,
    ) -> CircuitEngine:
        """
        Create a fluent circuit builder.

        Parameters
        ----------
        n_qubits : int
        mode : str, optional
            Execution mode hint.  See class docstring for available modes.
        bond_dim : int, optional
            Bond-dimension cap for ``'tensor'`` mode.
        """
        return CircuitEngine(self, n_qubits, mode=mode, bond_dim=bond_dim)

    def run(
        self,
        gates: Union[List[Dict], List[Tuple]],
        n_qubits: int,
        mode: str = "auto",
        bond_dim: Optional[int] = None,
        shots: int = 1024,
        seed: Optional[int] = None,
        return_statevector: bool = False,
        return_probabilities: bool = False,
        return_entropy_map: bool = False,
    ) -> CircuitResult:
        """
        Submit a gate list and return the result.

        Parameters
        ----------
        gates : list
            Gate list.  Each element is either a dict
            ``{'gate': str, 'qubits': [...], 'params': [...]}`` or a
            ``(gate_name, qubits)`` / ``(gate_name, qubits, params)`` tuple.
        n_qubits : int
        mode, bond_dim, shots, seed : see :meth:`engine`.
        return_statevector, return_probabilities, return_entropy_map : bool
            Request optional data in the response.
        """
        normalised = _normalise_gate_list(gates)
        return self._execute(
            n_qubits=n_qubits,
            gates=normalised,
            mode=mode,
            bond_dim=bond_dim,
            shots=shots,
            seed=seed,
            return_statevector=return_statevector,
            return_probabilities=return_probabilities,
            return_entropy_map=return_entropy_map,
        )

    def _execute(self, **kwargs: Any) -> CircuitResult:
        # Resolve mode: map user alias → backend internal name.
        raw_mode = kwargs.get("mode", "auto")
        backend_mode = _MODE_MAP.get(raw_mode, raw_mode)

        gates        = kwargs.get("gates", [])
        n_qubits     = kwargs.get("n_qubits")
        bond_dim     = kwargs.get("bond_dim")
        shots        = kwargs.get("shots", 1024)
        seed         = kwargs.get("seed")
        return_sv    = kwargs.get("return_statevector", False)
        return_em    = kwargs.get("return_entropy_map", False)

        instructions = [_gate_to_instruction(g) for g in gates]

        body: Dict[str, Any] = {
            "instructions": instructions,
            "shots":        shots,
            "mode":         backend_mode,
        }
        if n_qubits  is not None: body["n_qubits"]          = n_qubits
        if bond_dim  is not None: body["bond_dim"]           = bond_dim
        if seed      is not None: body["seed"]               = seed
        if return_sv:             body["return_statevector"] = True
        if return_em:             body["return_entropy_map"] = True

        # Submit (async job) and poll until complete.
        submit    = self._post("/circuits", body)
        job_id    = submit["job_id"]
        deadline  = _time.monotonic() + 300.0
        while True:
            job = self._get(f"/circuits/{job_id}")
            if job["status"] in ("completed", "failed"):
                break
            if _time.monotonic() > deadline:
                raise TimeoutError(
                    f"Circuit job {job_id} did not complete within 300 s"
                )
            _time.sleep(2.0)

        if job["status"] == "failed":
            raise QumulatorHTTPError(500, job.get("error") or "Circuit simulation failed")

        result = job.get("result") or {}

        sv = None
        if "statevector" in result:
            sv = np.array(
                [complex(r, i) for r, i in result["statevector"]], dtype=complex
            )
        probs = (
            np.array(result["probabilities"]) if "probabilities" in result else None
        )
        gc_raw = result.get("gaussian_certificate")
        gc = GaussianCertificate(**gc_raw) if gc_raw else None

        return CircuitResult(
            counts=result.get("counts", {}),
            n_qubits=result.get("n_qubits", n_qubits or 0),
            shots=result.get("shots", shots),
            statevector=sv,
            probabilities=probs,
            entropy_map=result.get("entropy_map"),
            gaussian_certificate=gc,
        )


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------


def _serialise_params(params: Sequence[Any]) -> List[Any]:
    """Convert gate params (floats, complex matrices) to JSON-safe form."""
    result = []
    for p in params:
        if isinstance(p, np.ndarray):
            flat = p.ravel().astype(complex)
            result.append([[float(v.real), float(v.imag)] for v in flat])
        elif isinstance(p, complex):
            result.append([float(p.real), float(p.imag)])
        elif isinstance(p, (float, int, np.floating, np.integer)):
            result.append(float(p))
        else:
            result.append(p)
    return result


def _normalise_gate_list(gates: Union[List[Dict], List[Tuple]]) -> List[Dict]:
    """Accept both dict-form and tuple-form gate lists."""
    out: List[Dict] = []
    for g in gates:
        if isinstance(g, dict):
            out.append(g)
        else:
            name, qubits, *rest = g
            entry: Dict[str, Any] = {
                "gate": name,
                "qubits": [qubits] if isinstance(qubits, int) else list(qubits),
            }
            if rest:
                entry["params"] = _serialise_params(rest[0])
            out.append(entry)
    return out


def _gate_to_instruction(gate: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an SDK gate dict to the backend instruction format.

    For ``'unitary'`` gates the SDK stores the matrix as a list of
    ``[real, imag]`` pairs inside ``params``.  The backend expects
    ``matrix_real`` and ``matrix_imag`` as separate 2-D float arrays.
    All other gate types pass through unchanged.
    """
    if gate.get("gate") != "unitary" or not gate.get("params"):
        return gate
    matrix_flat = gate["params"][0]   # list of [real, imag] pairs, row-major
    n = int(len(matrix_flat) ** 0.5)
    matrix_real = [[matrix_flat[r * n + c][0] for c in range(n)] for r in range(n)]
    matrix_imag = [[matrix_flat[r * n + c][1] for c in range(n)] for r in range(n)]
    out = {k: v for k, v in gate.items() if k != "params"}
    out["matrix_real"] = matrix_real
    out["matrix_imag"] = matrix_imag
    return out
