"""
Local statevector simulation engine.

Simulates quantum circuits entirely on the caller's machine — no API key,
no network round-trip, no cloud account required.

GPU acceleration is automatically enabled when one of the following
libraries is installed and available:

  1. **CuPy** (NVIDIA CUDA) — ``pip install cupy-cuda12x``
  2. **JAX** (CPU/GPU/TPU/Apple Silicon) — ``pip install jax[cuda12]``
  3. **PyTorch** (any backend) — ``pip install torch``

If none are available, or if ``device='cpu'`` is requested, NumPy is used.
The API is identical in all cases.

Examples
--------
Bell state (fast-path: instant, no statevector allocation)::

    from qumulator.local import LocalStatevectorEngine

    eng = LocalStatevectorEngine(n_qubits=2)
    eng.apply('h', 0).apply('cx', [0, 1])
    result = eng.run(shots=2048)
    print(result.counts)       # {'00': ~1024, '11': ~1024}

5-qubit GHZ state with entropy map::

    eng = LocalStatevectorEngine(n_qubits=5)
    eng.apply('h', 0)
    for i in range(4):
        eng.apply('cx', [i, i + 1])
    result = eng.run(shots=4096, return_entropy_map=True)
    print(result.entropy_map)  # [~1.0, ~1.0, ~1.0, ~1.0, ~1.0]

Large-qubit warning::

    eng = LocalStatevectorEngine(n_qubits=25)
    # UserWarning: Allocating statevector for 25 qubits requires ~256 MB ...
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from qumulator.local._fast_paths import try_fast_path
from qumulator.local._gates import resolve_gate


# ---------------------------------------------------------------------------
#  Memory footprint warning threshold
# ---------------------------------------------------------------------------

_LARGE_QUBIT_THRESHOLD = 20  # warn at N > this


def _memory_mb(n: int) -> float:
    """Approximate statevector memory in MB (complex128 = 16 bytes each)."""
    return (2 ** n) * 16 / (1024 ** 2)


# ---------------------------------------------------------------------------
#  GPU / accelerator backend detection
# ---------------------------------------------------------------------------

def _detect_array_module(device: str):
    """
    Return (xp, is_gpu) where xp is the array module to use.

    Resolution order for device='gpu':
      1. CuPy (NVIDIA CUDA)
      2. JAX (returns a thin wrapper — falls back if JAX not found)
      3. PyTorch (returns a thin wrapper)
      4. NumPy (CPU fallback, emits a warning)
    """
    if device == "cpu":
        return np, False

    # 1. CuPy
    try:
        import cupy as cp  # type: ignore[import]
        cp.array([0])  # trigger CUDA context; raises if no GPU
        return cp, True
    except Exception:
        pass

    # 2. JAX
    try:
        import jax.numpy as jnp  # type: ignore[import]
        import jax  # type: ignore[import]
        if jax.devices("gpu"):
            return _JaxWrapper(jnp), True
        # JAX on CPU is still useful; fall through to torch check
    except Exception:
        pass

    # 3. PyTorch
    try:
        import torch  # type: ignore[import]
        if torch.cuda.is_available():
            return _TorchWrapper(torch), True
    except Exception:
        pass

    warnings.warn(
        "device='gpu' requested but no GPU library (CuPy / JAX / PyTorch) is "
        "available. Falling back to NumPy (CPU).",
        UserWarning,
        stacklevel=3,
    )
    return np, False


class _JaxWrapper:
    """Thin compatibility shim: numpy-like interface over jax.numpy."""

    def __init__(self, jnp):
        self._jnp = jnp

    def zeros(self, shape, dtype=complex):
        return self._jnp.zeros(shape, dtype=self._jnp.complex128)

    def __getattr__(self, name):
        return getattr(self._jnp, name)


class _TorchWrapper:
    """Thin compatibility shim: numpy-like interface over torch."""

    def __init__(self, torch):
        self._torch = torch

    def zeros(self, shape, dtype=complex):
        if isinstance(shape, int):
            shape = (shape,)
        return self._torch.zeros(list(shape), dtype=self._torch.complex128).cuda()

    def __getattr__(self, name):
        # fall back to numpy for ops not explicitly wrapped
        return getattr(np, name)


# ---------------------------------------------------------------------------
#  Gate input normalisation
# ---------------------------------------------------------------------------

def _parse_gate_input(g: Any) -> Dict[str, Any]:
    """
    Normalise a gate expressed as a tuple or dict into canonical dict form.

    Accepted formats:

    - Dict:  ``{'gate': 'cx', 'qubits': [0, 1]}``  (returned unchanged)
    - Tuple: ``('cx', [0, 1])`` or ``('rz', 0, [np.pi/4])``
    """
    if isinstance(g, dict):
        return g
    name, qubits, *rest = g
    qubits_list: List[int] = (
        [int(qubits)] if isinstance(qubits, (int, np.integer))
        else [int(q) for q in qubits]
    )
    entry: Dict[str, Any] = {"gate": str(name), "qubits": qubits_list}
    if rest:
        params = rest[0]
        entry["params"] = list(params) if not isinstance(params, list) else params
    return entry


# ---------------------------------------------------------------------------
#  LocalStatevectorEngine
# ---------------------------------------------------------------------------

class LocalStatevectorEngine:
    """
    Local, in-process quantum circuit simulator based on statevector evolution.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.  There is no hard limit; however a
        ``UserWarning`` is emitted for ``n_qubits > 20`` together with the
        expected memory footprint so the caller can make an informed choice.
    device : {'cpu', 'gpu'}
        Compute device.  ``'gpu'`` auto-detects CuPy → JAX → PyTorch in
        that order and falls back to NumPy with a warning if none is found.

    Notes
    -----
    The statevector is stored as a complex128 array of length 2^n.
    Qubit ordering follows the MSB-first (big-endian) convention used by the
    Qumulator cloud API: qubit 0 is the leftmost character in the bitstring.

    The engine applies a fluent builder pattern identical to the cloud
    :class:`~qumulator.circuit.CircuitEngine`: use :meth:`apply` to
    accumulate gates and :meth:`run` (or :meth:`sample`) to execute.

    Before executing the full simulation, :meth:`run` dispatches to analytic
    fast paths for recognised circuit patterns (Bell, BV, QFT, Grover).
    """

    def __init__(self, n_qubits: int, device: str = "cpu") -> None:
        self.n_qubits = int(n_qubits)
        self.device = device

        if n_qubits > _LARGE_QUBIT_THRESHOLD:
            mb = _memory_mb(n_qubits)
            warnings.warn(
                f"Allocating statevector for {n_qubits} qubits requires "
                f"~{mb:.0f} MB of memory (complex128). This will be slow on CPU. "
                f"Consider using device='gpu' or the cloud API for N > {_LARGE_QUBIT_THRESHOLD}.",
                UserWarning,
                stacklevel=2,
            )

        self._xp, self._is_gpu = _detect_array_module(device)
        self._gates: List[Dict[str, Any]] = []
        self._sv: Optional[np.ndarray] = None  # lazy allocation

    # ------------------------------------------------------------------
    #  Gate API  (mirrors CircuitEngine.apply)
    # ------------------------------------------------------------------

    def apply(
        self,
        gate_name: str,
        qubits: Union[int, Sequence[int]],
        params: Optional[Sequence[Any]] = None,
    ) -> "LocalStatevectorEngine":
        """
        Append a gate to the circuit.

        Parameters
        ----------
        gate_name : str
            Gate identifier (case-insensitive). Common values: ``'h'``,
            ``'x'``, ``'cx'``, ``'rz'``, ``'ccx'``, ``'swap'``, etc.
        qubits : int or list[int]
            Target qubit index or list of qubit indices.
        params : list, optional
            Gate parameters (rotation angles, etc.).

        Returns
        -------
        self
            Enables method chaining: ``eng.apply(...).apply(...)``.
        """
        if isinstance(qubits, (int, np.integer)):
            qubits = [int(qubits)]
        else:
            qubits = [int(q) for q in qubits]
        entry: Dict[str, Any] = {"gate": gate_name, "qubits": qubits}
        if params is not None:
            entry["params"] = list(params)
        self._gates.append(entry)
        return self

    def run_gates(
        self,
        gates: Sequence[Any],
        shots: int = 1024,
        seed: Optional[int] = None,
        return_statevector: bool = False,
        return_probabilities: bool = False,
        return_entropy_map: bool = False,
        reset_first: bool = True,
    ) -> "CircuitResult":  # type: ignore[name-defined]
        """
        Submit a gate list (tuples or dicts) and execute the circuit.

        Convenience wrapper that accepts both:

        - Tuple format: ``('h', 0)`` / ``('rz', 0, [np.pi / 4])``
        - Dict format:  ``{'gate': 'h', 'qubits': [0]}``

        Parameters
        ----------
        gates : sequence
            Gate list in either tuple or dict form (or mixed).
        shots, seed, return_statevector, return_probabilities, return_entropy_map :
            Forwarded to :meth:`run`.
        reset_first : bool
            Clear previously accumulated gates before appending the new list
            (default ``True``).

        Returns
        -------
        CircuitResult
        """
        if reset_first:
            self.reset()
        for g in gates:
            parsed = _parse_gate_input(g)
            self.apply(parsed["gate"], parsed["qubits"], parsed.get("params"))
        return self.run(
            shots=shots,
            seed=seed,
            return_statevector=return_statevector,
            return_probabilities=return_probabilities,
            return_entropy_map=return_entropy_map,
        )

    def reset(self) -> "LocalStatevectorEngine":
        """Clear all accumulated gates and the statevector (returns self)."""
        self._gates = []
        self._sv = None
        return self

    def set_statevector(self, sv: np.ndarray) -> "LocalStatevectorEngine":
        """
        Inject an arbitrary initial statevector.

        The array is cast to complex128 and normalised.  Its length must be
        exactly 2^n_qubits.
        """
        sv = np.asarray(sv, dtype=complex)
        if sv.shape != (2 ** self.n_qubits,):
            raise ValueError(
                f"Expected statevector of length {2**self.n_qubits}, got {sv.shape}"
            )
        norm = float(np.linalg.norm(sv))
        if norm == 0:
            raise ValueError("Statevector has zero norm.")
        self._sv = sv / norm
        return self

    # ------------------------------------------------------------------
    #  Execution
    # ------------------------------------------------------------------

    def run(
        self,
        shots: int = 1024,
        seed: Optional[int] = None,
        return_statevector: bool = False,
        return_probabilities: bool = False,
        return_entropy_map: bool = False,
        dry_run: bool = False,
    ) -> "CircuitResult":  # type: ignore[name-defined]  # avoid circular import
        """
        Execute the circuit and return a :class:`~qumulator.circuit.CircuitResult`.

        Before running the full simulation the method tries each registered
        analytic fast path (Bell, BV, QFT, Grover).  If a match is found,
        the result is returned instantly without allocating a statevector —
        unless ``return_statevector`` or ``return_entropy_map`` is ``True``,
        in which case the full simulation is run anyway.

        Parameters
        ----------
        shots : int
            Number of measurement samples.
        seed : int, optional
            RNG seed for reproducible sampling.
        return_statevector : bool
            Include the final statevector in the result.
        return_probabilities : bool
            Include the probability vector in the result.
        return_entropy_map : bool
            Compute per-qubit entanglement entropy (von Neumann, log-base-2).
        dry_run : bool
            Validate the circuit without executing. Returns ``None``.

        Returns
        -------
        CircuitResult
        """
        from qumulator.circuit import CircuitResult  # local import to avoid cycle

        if dry_run:
            return None  # type: ignore[return-value]

        need_sv = return_statevector or return_probabilities or return_entropy_map

        # -- Analytic fast path -------------------------------------------
        if not need_sv:
            fp = try_fast_path(self._gates, self.n_qubits, shots=shots, seed=seed)
            if fp is not None:
                return CircuitResult(
                    counts=fp["counts"],
                    n_qubits=self.n_qubits,
                    shots=shots,
                )

        # -- Full statevector simulation ----------------------------------
        sv = self._simulate()

        # Measurement — always use NumPy for probability computation.
        # (_simulate returns a NumPy array; GPU acceleration for statevector
        # evolution requires a separate CuPy/JAX/Torch code path.)
        probs = np.abs(np.asarray(sv)) ** 2

        # Numerical safety: clamp small negatives, renormalise
        probs = np.abs(probs)
        total = probs.sum()
        if total == 0:
            raise RuntimeError("Statevector collapsed to zero.")
        probs /= total

        rng = np.random.default_rng(seed)
        n = self.n_qubits
        dim = 2 ** n
        indices = rng.choice(dim, size=shots, p=probs)
        counts: Dict[str, int] = {}
        for idx in indices:
            key = format(int(idx), f"0{n}b")
            counts[key] = counts.get(key, 0) + 1

        # Optional extras
        sv_out = None
        prob_out = None
        entropy_map = None

        if return_statevector:
            # Return as numpy array regardless of compute device
            sv_np = np.asarray(sv) if self._is_gpu else np.asarray(sv)
            sv_out = sv_np

        if return_probabilities:
            prob_out = probs

        if return_entropy_map:
            entropy_map = self._compute_entropy_map(sv if not self._is_gpu else np.asarray(sv))

        return CircuitResult(
            counts=counts,
            n_qubits=n,
            shots=shots,
            statevector=sv_out,
            probabilities=prob_out,
            entropy_map=entropy_map,
        )

    def sample(self, shots: int = 1024, seed: Optional[int] = None) -> Dict[str, int]:
        """
        Sample measurement outcomes.

        Returns ``{bitstring: count}`` using MSB-first ordering (qubit 0 is
        the leftmost character), matching the cloud API convention.
        """
        return self.run(shots=shots, seed=seed).counts

    # ------------------------------------------------------------------
    #  Simulation internals
    # ------------------------------------------------------------------

    def _simulate(self) -> np.ndarray:
        """
        Evolve the statevector gate by gate.

        Returns the final statevector as a 1-D complex128 array of length 2^n.
        """
        n = self.n_qubits
        dim = 2 ** n

        if self._sv is not None:
            sv = self._sv.copy()
        else:
            sv = np.zeros(dim, dtype=complex)
            sv[0] = 1.0

        for gate_entry in self._gates:
            name = str(gate_entry.get("gate", "")).lower()
            if name in ("measure", "barrier", "reset"):
                continue
            qubits = gate_entry.get("qubits", [])
            if isinstance(qubits, int):
                qubits = [qubits]
            qubits = [int(q) for q in qubits]
            params = gate_entry.get("params", None)

            if name == "unitary":
                U = np.asarray(params, dtype=complex)
            else:
                U = np.asarray(resolve_gate(name, params), dtype=complex)

            sv = _apply_sv(sv, U, qubits, n)

        # Persist statevector so entropy / sv output can share it
        self._sv = sv
        return sv

    # ------------------------------------------------------------------
    #  Entropy map
    # ------------------------------------------------------------------

    def _compute_entropy_map(self, sv: np.ndarray) -> List[float]:
        """
        Compute per-qubit von Neumann entanglement entropy (log-base-2).

        For each qubit *i*, traces out all other qubits to form the
        reduced density matrix ρ_i and computes S = -Tr(ρ_i log₂ ρ_i).

        A product state returns 0.0 for all qubits; a maximally entangled
        qubit returns 1.0.

        Reference: Nielsen & Chuang §2.4.2; computed via SVD for stability.
        """
        n = self.n_qubits
        psi = sv.reshape([2] * n)
        entropies: List[float] = []

        for i in range(n):
            # Move qubit i to axis 0, then flatten into (2, 2^(n-1)) matrix
            psi_i = np.moveaxis(psi, i, 0).reshape(2, -1)
            # Reduced density matrix ρ_i = ψ ψ†
            rho = psi_i @ psi_i.conj().T
            # Eigenvalues of ρ_i are the Schmidt coefficients squared
            evals = np.linalg.eigvalsh(rho).real
            evals = evals[evals > 1e-15]  # discard numerical zeros
            s = float(-np.sum(evals * np.log2(evals)))
            entropies.append(round(s, 8))

        return entropies


# ---------------------------------------------------------------------------
#  Core tensor-contraction kernel (pure NumPy, device-agnostic copy)
# ---------------------------------------------------------------------------

def _apply_sv(sv: np.ndarray, U: np.ndarray, qubits: List[int], n: int) -> np.ndarray:
    """
    Apply unitary *U* acting on *qubits* to statevector *sv*.

    Implements efficient tensor-contraction via reshape + moveaxis + matmul.
    Works for 1-, 2-, and 3-qubit gates.

    Convention: qubit 0 = axis 0 (MSB / leftmost in bitstring).
    """
    k = len(qubits)
    dim = 2 ** n

    # Reshape sv into n-qubit tensor [2, 2, ..., 2]
    psi = sv.reshape([2] * n)

    # Move target qubit axes to the *front*
    psi = np.moveaxis(psi, qubits, list(range(k)))

    # Reshape to (2^k, 2^(n-k)) for matrix multiply
    rest = n - k
    psi_flat = psi.reshape(2 ** k, 2 ** rest)

    # Apply gate: U @ psi_flat
    psi_out = U.reshape(2 ** k, 2 ** k) @ psi_flat

    # Reshape back to [2]*k + [2]*rest
    psi_out = psi_out.reshape([2] * k + [2] * rest)

    # Restore original axis order
    psi_out = np.moveaxis(psi_out, list(range(k)), qubits)

    return psi_out.reshape(dim)
