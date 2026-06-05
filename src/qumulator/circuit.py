"""
Qumulator Circuit Client � quantum circuit execution via the Qumulator API.

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
``'statevector'``  Full statevector. Correct for any circuit. N <= ~25.
``'cluster_mps'``   Memory-efficient. Suited for large N, low-to-moderate
                   entanglement (VQE, QAOA, chemistry).
``'mps'``          Tensor-network backend. Efficient for structured and
                   1D circuits. Supports N > 50.
``'hamiltonian'``  Direct Hamiltonian evolution without gate decomposition.
                   Use with :meth:`CircuitEngine.evolve_hamiltonian`.
``'gaussian'``     Gaussian covariance matrix simulation. Exact for Clifford
                   circuits; principled approximation for non-Clifford content.
                   Returns a :class:`~qumulator.models.GaussianCertificate` in
                   ``result.gaussian_certificate`` classifying the circuit as
                   simulable, likely Gaussian, or requiring a correction.
                   Memory scales as O(n�) instead of O(2n).
``'cluster_statevector'``  Exact cluster-factorization engine. No 2n state vector is
                   ever allocated. Memory O(S 2^k_c) where k_c is the size of
                   each entangled cluster. Exact for ALL circuits (TVD = 0).
                   Returns per-qubit marginal probabilities.
``'greens'``       Green's function / Bloch encoding. Exact within the
                   free-fermion (Gaussian) subspace. O(N�) memory. Returns
                   1-RDM and entropy map. Note: CNOT in the exchange subspace
                   is not faithfully represented; use ``'cluster'`` instead.
"""
from __future__ import annotations

import dataclasses
import time as _time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from qumulator._http import _BaseClient, QumulatorHTTPError
from qumulator.models import GaussianCertificate

# ---------------------------------------------------------------------------
# Tier depth limits: (max_n_qubits, max_entangling_depth)
# These mirror the published tier table. Modes that are unconditionally exact
# (statevector, cluster, greens, gaussian) are handled separately.
_TIER_DEPTH_LIMITS: List[Tuple[int, int]] = [
    (20,   20),   # Tier 1
    (54,    9),   # Tier 2
    (105,   8),   # Tier 3
    (1000,  7),   # Tier 4
]

# Modes that have no depth restriction (exact or non-MPS, see _EXACT_MODES set)
_EXACT_MODES = {"statevector", "cluster_statevector", "greens", "gaussian"}
# Statevector mode qubit cap
_STATEVECTOR_MAX_QUBITS = 20


def _validate_circuit(
    n_qubits: int,
    gates: List[Dict],
    mode: str,
) -> Optional[str]:
    """
    Check circuit parameters against published tier limits. Pure client-side �
    no API call. Returns an error message string if invalid, else None.
    """
    if mode in _EXACT_MODES:
        if mode == "statevector" and n_qubits > _STATEVECTOR_MAX_QUBITS:
            return (
                f"'statevector' mode supports at most {_STATEVECTOR_MAX_QUBITS} "
                f"qubits; circuit has {n_qubits}. "
                f"Consider 'mps', 'cluster_mps', or 'cluster_statevector' mode for larger circuits."
            )
        return None  # cluster/greens/gaussian have no depth constraint

    if mode == "auto":
        return None  # mode will be resolved server-side; no client-side depth check

    # Count entangling layers (gates acting on 2+ qubits)
    entangling_depth = sum(
        1 for g in gates if len(g.get("qubits", [])) >= 2
    )

    max_depth: Optional[int] = None
    for max_n, max_d in _TIER_DEPTH_LIMITS:
        if n_qubits <= max_n:
            max_depth = max_d
            break

    if max_depth is None:
        return (
            f"Circuit has {n_qubits} qubits; maximum supported is 1,000 qubits "
            f"in MPS modes."
        )

    if entangling_depth > max_depth:
        return (
            f"Circuit has {n_qubits} qubits and {entangling_depth} entangling "
            f"layer(s); the tier limit for this qubit range is {max_depth}. "
            f"Reduce entangling depth or use 'cluster_statevector' mode (exact for any depth, "
            f"memory O(S 2^k_c))."
        )

    return None


# ---------------------------------------------------------------------------
#  CU cost model � client-side estimator (no API call)
# ---------------------------------------------------------------------------
#
# 1 CU � 1 second of engine wall-clock CPU time.
#
# Calibration benchmarks (observed, from published docs):
#   Simple statevector circuit         :   1�3 CU
#   20-qubit depth-20 statevector      :  ~45 CU
#   54-qubit chi=16 MPS depth-6        :  ~58 CU
#   105-qubit chi=16 MPS depth-5       :  ~46 CU
#   1000-qubit chi=16 MPS depth-3      : ~112 CU
#
# Statevector formula: O(2^N � G)  with constant _K_SV
# MPS formula:         O(N � ?� � G) with constant _K_MPS
#
# G = number of 2-qubit gates (dominant cost driver).
# Both formulae are approximations; actual cost depends on entanglement structure.
# ---------------------------------------------------------------------------

_BASE_CU = 0.3
"""Fixed overhead per circuit submission (CU)."""

_PER_SHOT_CU = 0.15e-3
"""Additional cost per sample shot (CU per shot). ~0.15 CU per 1000 shots."""

_DEFAULT_BOND_DIM = 16
"""Assumed ? when bond_dim is not specified for MPS modes."""

# O(2^N � G) calibration constant.
# Calibrated: 20q, ~200 entangling gates ? 45 CU
# k = 45 / (2^20 � 200) � 2.14e-7
_K_SV = 2.14e-7

# O(N � ?� � G) calibration constant.
# Calibrated: 54q, ?=16, ~162 entangling gates ? 58 CU
# k = 58 / (54 � 16^3 � 162) � 1.62e-6
_K_MPS = 1.62e-6

_MODE_MULTIPLIER: Dict[str, float] = {
    "statevector":          1.0,
    "cluster_mps":          1.5,
    "mps":                  2.0,
    "cluster_statevector":  3.0,
    "greens":               1.2,
    "gaussian":             1.0,
    "hamiltonian":          1.5,
    "cluster_exact":        3.0,
    "cluster_exact_graph":  3.0,
    "dyson":                2.0,
    "matrix":               1.0,
    "phase":                1.0,
    "local":                0.0,  # local simulation — no server-side billing
    "auto":                 0.0,  # cost unknown until resolved; actual cost in job receipt
}

# Modes that use the statevector (2^N) kernel rather than MPS
_SV_MODES = {"statevector", "cluster_statevector", "greens", "gaussian"}


@dataclasses.dataclass
class CostEstimate:
    """
    Client-side estimate of the compute-unit (CU) cost for a circuit.

    Produced by :meth:`CircuitEngine.estimated_cost` and
    :meth:`CircuitClient.estimate_cost`.  This estimate is purely
    formula-based � no API call is made.

    Attributes
    ----------
    total_cu : float
        Total estimated CU cost.  1 CU � 1 second of engine CPU time.
    breakdown : dict
        Per-component cost breakdown with keys:
        ``'base'``, ``'depth_surcharge'``, ``'shots'``, ``'mode_multiplier'``.

    Notes
    -----
    The model is calibrated to observed benchmarks (see module constants)
    and should be accurate to within a factor of 2 for typical circuits.
    Exotic entanglement structure or very deep circuits may deviate more.
    """

    total_cu: float
    breakdown: Dict[str, float]


def _compute_cu_estimate(
    n_qubits: int,
    gates: List[Dict[str, Any]],
    mode: str,
    bond_dim: Optional[int],
    shots: int,
) -> CostEstimate:
    """Compute a CostEstimate for the given circuit parameters (pure function)."""
    chi = bond_dim if bond_dim is not None else _DEFAULT_BOND_DIM
    multiplier = _MODE_MULTIPLIER.get(mode, _MODE_MULTIPLIER.get(
        mode, 1.5
    ))

    n_2q = sum(1 for g in gates if len(g.get("qubits", [])) >= 2)

    if mode in _SV_MODES or n_qubits <= 20:
        # Statevector kernel: O(2^N � G)
        surcharge = _K_SV * (2 ** min(n_qubits, 30)) * max(n_2q, 1)
    else:
        # MPS kernel: O(N � ?� � G)
        surcharge = _K_MPS * n_qubits * (chi ** 3) * max(n_2q, 1)

    base = _BASE_CU
    shots_cost = shots * _PER_SHOT_CU
    # Apply mode multiplier to the compute-intensive part only
    compute_subtotal = (base + surcharge) * multiplier
    total = compute_subtotal + shots_cost

    breakdown: Dict[str, float] = {
        "base": round(base * multiplier, 6),
        "depth_surcharge": round(surcharge * multiplier, 6),
        "shots": round(shots_cost, 6),
        "mode_multiplier": multiplier,
    }
    return CostEstimate(total_cu=round(total, 4), breakdown=breakdown)


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

    f_Q_density: Optional[float] = None
    """Quantum Fisher Information density (T�th�G�hne 2012).

    ``f_Q > k`` certifies genuine ``(k+1)``-partite entanglement.
    Values above 1 indicate multi-partite entanglement; values above
    ``n_qubits - 1`` indicate full ``n``-partite entanglement.
    Always returned (never ``None``) unless the mode does not support it."""

    entanglement_depth: Optional[int] = None
    """Certified entanglement depth: ``floor(f_Q_density)``.

    The output state is guaranteed to contain genuine entanglement spanning
    at least ``entanglement_depth + 1`` qubits.  A value of 0 means the
    state is consistent with a product state; 1 means at least pairwise
    entanglement is certified, and so on."""

    predicted_tvd: Optional[float] = None
    """Predicted total variation distance (TVD) to the exact output distribution.

    Model-based accuracy bound calibrated per entanglement phase (Z1�Z5).
    ``0.0`` for unconditionally exact modes (``'statevector'``, ``'cluster'``).
    Use this as a conservative upper bound on the simulation error."""

    phase_label: Optional[str] = None
    """Entanglement-phase label (``'Z1'``�``'Z5'``).

    Indicates the entanglement regime of the circuit's structure.
    Values: `'product_state'` (S<0.1 bits), `'area_law'` (0.1-0.6),
    `'topological_class'` (0.6-1.2), `'near_volume_law'` (1.2-2.5), `'volume_law'` (>=2.5).
    ``None`` when the backend does not return this field."""
    resolved_mode: Optional[str] = None
    """The simulation mode actually used.  Populated when ``mode='auto'`` was
    requested; equals the requested mode for all explicit mode selections."""

    preflight_report: Optional[Dict] = None
    """Routing diagnostics returned when ``mode='auto'`` was used.
    Contains: ``d_ky``, ``entanglement_regime``, ``reasoning``, ``is_tree``,
    ``edge_density``, ``d_s``, ``n_2q_gates``, ``n_t_gates``,
    ``ky_gp_consistent``."""
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
        Bond-dimension cap for ``'mps'`` mode.

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
        mode: str = "mps",
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

    def validate(self) -> None:
        """
        Check the circuit against published tier depth limits client-side.

        Raises ``ValueError`` with a descriptive message if the circuit
        exceeds the limit for its qubit count and mode � no API call is made.
        Does nothing if the circuit is within limits.

        Examples
        --------
        ::

            eng = client.circuit.engine(n_qubits=1000, mode="mps")
            for i in range(0, 1000, 2):
                eng.apply("h", i).apply("cx", [i, i + 1])
            eng.validate()   # passes (depth 1, limit 7 for N=1000)

            # Exceeds tier limit:
            deep_eng = client.circuit.engine(n_qubits=200, mode="mps")
            for _ in range(10):
                for i in range(0, 200, 2):
                    deep_eng.apply("cx", [i, i + 1])
            deep_eng.validate()
            # ValueError: Circuit has 200 qubits and 10 entangling layer(s);
            # the tier limit for this qubit range is 7. ...
        """
        error = _validate_circuit(self.n_qubits, self._gates, self.mode)
        if error:
            raise ValueError(error)

    def estimated_cost(self, shots: int = 1024) -> CostEstimate:
        """
        Estimate the compute-unit (CU) cost of submitting this circuit.

        Purely client-side � no API call is made.  The estimate is based on
        the gates accumulated so far, the current :attr:`mode`, and
        ``shots``.

        Parameters
        ----------
        shots : int
            Number of measurement samples to assume for the cost estimate.
            Defaults to ``1024``.

        Returns
        -------
        CostEstimate
            ``total_cu`` is the estimated cost in compute units (1 CU � 1 s
            of engine CPU time).  ``breakdown`` shows the per-component split.

        Examples
        --------
        ::

            eng = client.circuit.engine(n_qubits=20, mode='statevector')
            for i in range(0, 20, 2):
                eng.apply('cx', [i, i + 1])
            est = eng.estimated_cost(shots=4096)
            print(f"Estimated cost: {est.total_cu:.2f} CU")
            print(est.breakdown)
        """
        return _compute_cu_estimate(
            self.n_qubits, self._gates, self.mode, self.bond_dim, shots
        )

    # -- execution API --------------------------------------------------------

    def run(
        self,
        shots: int = 1024,
        seed: Optional[int] = None,
        return_statevector: bool = False,
        return_probabilities: bool = False,
        return_entropy_map: bool = False,
        dry_run: bool = False,
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
        dry_run : bool
            If ``True``, validate the circuit against tier depth limits
            client-side and return without submitting to the API.  Raises
            ``ValueError`` if the circuit is invalid; returns ``None``
            if it passes.  Useful for pre-flight checks in loops or notebooks.

        Returns
        -------
        CircuitResult
        """
        self.validate()
        if dry_run:
            return None  # type: ignore[return-value]
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
    ``'statevector'``  Full statevector simulation. Correct for any circuit.
                       Practical for N <= ~25 qubits.
    ``'cluster_mps'``  Compressed representation. Efficient for large N
                       with low-to-moderate entanglement (VQE, QAOA, chemistry).
    ``'mps'``          Tensor-network backend. Efficient for 1D and shallow
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
        mode: str = "mps",
        bond_dim: Optional[int] = None,
    ) -> "Union[CircuitEngine, Any]":
        """
        Create a fluent circuit builder.

        Parameters
        ----------
        n_qubits : int
        mode : str, optional
            Execution mode hint.  See class docstring for available modes.
            Use ``'local'`` to get a :class:`~qumulator.local.LocalStatevectorEngine`
            for in-process simulation without an API call.
        bond_dim : int, optional
            Bond-dimension cap for ``'mps'`` mode.
        """
        if mode == "local":
            from qumulator.local import LocalStatevectorEngine
            return LocalStatevectorEngine(n_qubits)
        return CircuitEngine(self, n_qubits, mode=mode, bond_dim=bond_dim)

    def run(
        self,
        gates: Union[List[Dict], List[Tuple]],
        n_qubits: int,
        mode: str = "mps",
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

    def run_qasm(
        self,
        qasm_source: str,
        shots: int = 1024,
        seed: Optional[int] = None,
        mode: str = "mps",
        bond_dim: Optional[int] = None,
    ) -> "CircuitResult":
        """
        Submit an OpenQASM 2 or 3 string and return the result.

        Parameters
        ----------
        qasm_source : str
            OpenQASM 2 or 3 source code.
        shots : int
            Number of measurement samples.
        seed : int, optional
            RNG seed for reproducible sampling.
        mode : str, optional
            Execution mode hint.
        bond_dim : int, optional
            Bond-dimension cap for ``'mps'`` mode.
        """
        backend_mode = mode
        body: Dict[str, Any] = {
            "qasm":  qasm_source,
            "shots": shots,
            "mode":  backend_mode,
        }
        if bond_dim is not None: body["bond_dim"] = bond_dim
        if seed     is not None: body["seed"]     = seed

        submit   = self._post("/circuits", body)
        job_id   = submit["job_id"]
        deadline = _time.monotonic() + 3600.0
        while True:
            job = self._get(f"/circuits/{job_id}")
            if job["status"] in ("completed", "failed"):
                break
            if _time.monotonic() > deadline:
                raise TimeoutError(
                    f"Circuit job {job_id} did not complete within 3600 s"
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
        elif result.get("statevector_real") is not None:
            sv_re = result["statevector_real"]
            sv_im = result.get("statevector_imag") or [0.0] * len(sv_re)
            sv = np.array([complex(r, i) for r, i in zip(sv_re, sv_im)], dtype=complex)
        probs = (
            np.array(result["probabilities"]) if "probabilities" in result else None
        )
        gc_raw = result.get("gaussian_certificate")
        gc = GaussianCertificate(**gc_raw) if gc_raw else None

        return CircuitResult(
            counts=result.get("counts", {}),
            n_qubits=result.get("n_qubits", 0),
            shots=result.get("shots", shots),
            statevector=sv,
            probabilities=probs,
            entropy_map=result.get("entropy_map"),
            gaussian_certificate=gc,
            f_Q_density=result.get("f_Q_density"),
            entanglement_depth=result.get("entanglement_depth"),
            predicted_tvd=result.get("predicted_tvd"),
            phase_label=result.get("phase_label"),
            resolved_mode=result.get("resolved_mode"),
            preflight_report=result.get("preflight_report"),
        )

    def estimate_cost(
        self,
        gates: Union[List[Dict], List[Tuple]],
        n_qubits: int,
        shots: int = 1024,
        mode: str = "mps",
        bond_dim: Optional[int] = None,
    ) -> CostEstimate:
        """
        Estimate the compute-unit (CU) cost of a circuit without submitting it.

        Purely client-side � no API call is made.

        Parameters
        ----------
        gates : list
            Gate list in dict or tuple format (same as :meth:`run`).
        n_qubits : int
            Number of qubits.
        shots : int
            Number of measurement samples to assume.  Default ``1024``.
        mode : str
            Execution mode.  Default ``'mps'``.
        bond_dim : int, optional
            Bond-dimension cap assumed for MPS modes.

        Returns
        -------
        CostEstimate
            ``total_cu`` is the estimated cost (1 CU � 1 s of engine CPU).
            ``breakdown`` contains per-component costs.

        Examples
        --------
        ::

            est = client.circuit.estimate_cost(
                gates=[('h', 0), ('cx', [0, 1])],
                n_qubits=2,
                shots=4096,
            )
            print(f"{est.total_cu:.3f} CU")
        """
        normalised = _normalise_gate_list(gates)
        return _compute_cu_estimate(n_qubits, normalised, mode, bond_dim, shots)

    def preflight(self, qasm_source: str) -> Dict[str, Any]:
        """
        Zero-cost circuit pre-flight analysis (no simulation, no CU cost).

        Analyses the QASM circuit, builds the entanglement graph, estimates
        the Kaplan-Yorke dimension D_KY, and returns the recommended
        simulation mode — without running any simulation.

        Parameters
        ----------
        qasm_source : str
            OpenQASM 2 or 3 source code.

        Returns
        -------
        dict
            Keys: ``recommended_mode``, ``reasoning``, ``d_ky``,
            ``entanglement_regime``, ``d_s``, ``is_tree``, ``edge_density``,
            ``n_2q_gates``, ``n_t_gates``, ``ky_gp_consistent``.

        Examples
        --------
        ::

            report = client.circuit.preflight(qasm_source)
            print(report["recommended_mode"])   # e.g. 'mps'
            print(report["reasoning"])
        """
        return self._post("/circuits/preflight", {"qasm": qasm_source})

    def preflight_instructions(
        self,
        n_qubits: int,
        gates: Union[List[Dict], List[Tuple]],
    ) -> Dict[str, Any]:
        """
        Zero-cost pre-flight analysis for a gate-instruction circuit.

        Same as :meth:`preflight` but accepts a gate list instead of QASM.

        Parameters
        ----------
        n_qubits : int
        gates : list
            Gate list in dict or tuple format (same as :meth:`run`).

        Returns
        -------
        dict
            Same keys as :meth:`preflight`.
        """
        normalised = _normalise_gate_list(gates)
        instructions = [_gate_to_instruction(g) for g in normalised]
        return self._post("/circuits/preflight", {
            "n_qubits": n_qubits,
            "instructions": instructions,
        })

    def _execute(self, **kwargs: Any) -> CircuitResult:
        # Resolve mode: map user alias ? backend internal name.
        raw_mode = kwargs.get("mode", "mps")
        backend_mode = raw_mode

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
        deadline  = _time.monotonic() + 3600.0
        while True:
            job = self._get(f"/circuits/{job_id}")
            if job["status"] in ("completed", "failed"):
                break
            if _time.monotonic() > deadline:
                raise TimeoutError(
                    f"Circuit job {job_id} did not complete within 3600 s"
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
        elif result.get("statevector_real") is not None:
            sv_re = result["statevector_real"]
            sv_im = result.get("statevector_imag") or [0.0] * len(sv_re)
            sv = np.array([complex(r, i) for r, i in zip(sv_re, sv_im)], dtype=complex)
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
            f_Q_density=result.get("f_Q_density"),
            entanglement_depth=result.get("entanglement_depth"),
            predicted_tvd=result.get("predicted_tvd"),
            phase_label=result.get("phase_label"),
            resolved_mode=result.get("resolved_mode"),
            preflight_report=result.get("preflight_report"),
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
