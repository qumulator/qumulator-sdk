"""
Response models matching the backend API.
These are plain Pydantic models — no engine source involved.
"""
from typing import Optional

from pydantic import BaseModel


# ── Job wrapper ──────────────────────────────────────────────────────────────

class JobStatus(BaseModel):
    job_id: str
    status: str          # queued | running | completed | failed
    endpoint: str
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    result: dict | None = None
    error: str | None = None

    @property
    def is_done(self) -> bool:
        return self.status in ("completed", "failed")

    @property
    def ok(self) -> bool:
        return self.status == "completed"


# ── HOMO / DFT ───────────────────────────────────────────────────────────────

class HomoResult(BaseModel):
    homo_E_eV: float
    lumo_E_eV: float
    gap_eV: float
    homo_density: list[float]
    lumo_density: list[float]
    n_occ: int
    n_heavy: int
    heavy_symbols: list[str]
    basis: str
    xc: str


# ── KLT ──────────────────────────────────────────────────────────────────────

class KLTResult(BaseModel):
    energy: float
    states: list[float]
    entropy_list: list[float]
    max_S: float
    mean_S: float


# ── Hafnian ──────────────────────────────────────────────────────────────────

class HafnianResult(BaseModel):
    haf_real: float
    haf_imag: float
    phase_used: str
    elapsed: float
    max_S: float
    mean_S: float
    n_edges: int
    est_matchings: float
    bond_dimension: int
    v2_energy: float

    @property
    def value(self) -> complex:
        return complex(self.haf_real, self.haf_imag)


# ── Gaussian circuit certificate ─────────────────────────────────────────────

class GaussianCertificate(BaseModel):
    """
    Diagnostic certificate returned when a circuit is run in Gaussian mode.

    The certificate classifies how well the Gaussian approximation holds for
    the submitted circuit and provides entanglement regime diagnostics.
    """

    rcs_certificate: str
    """One of ``"GAUSSIAN_SIMULABLE"``, ``"LIKELY_GAUSSIAN"``, or
    ``"NON_GAUSSIAN_CORRECTION_NEEDED"``."""

    entanglement_regime: Optional[str] = None
    """``"area_law"``, ``"transitional"``, or ``"volume_law"``."""

    wigner_negativity_estimate: Optional[float] = None
    """Estimated non-Gaussian correction from non-Clifford gate content."""

    gaussian_fidelity: Optional[float] = None
    """Estimated fidelity of the Gaussian approximation (0–1)."""

    xeb_lower_bound: Optional[float] = None
    """Cross-entropy benchmark lower bound."""

    kaplan_yorke_dim: Optional[float] = None
    """Kaplan-Yorke dimension of the entanglement structure (0–3)."""

    koopman_mode_count: Optional[int] = None
    """Number of independent modes sufficient to represent the entanglement structure."""

    compression_ratio: Optional[float] = None
    """``koopman_mode_count / n_qubits`` — fraction of modes active."""


# ── Time Evolution / TEBD ─────────────────────────────────────────────────────

class HamiltonianTerm(BaseModel):
    """A single nearest-neighbor or single-site Pauli term."""
    sites:    list[int]
    operator: str
    strength: float


class HamiltonianSpec(BaseModel):
    """
    Pauli-sum Hamiltonian.

    Use a named preset or provide ``terms`` explicitly.

    Presets
    -------
    ``"ising_1d"``
        Transverse-field Ising model: ``−J·ZZ − h·X``.
        Parameters: ``J`` (default 1.0), ``h`` (default 1.0).
    ``"xx_model"``
        Free-fermion XX chain: ``−(t/2)(XX+YY)``.
        Parameter: ``t`` (default 1.0).
    ``"heisenberg"``
        XXX Heisenberg chain: ``J(XX+YY+ZZ)``.
        Parameter: ``J`` (default 1.0).
    ``"kuramoto_ising"``
        KLT Kuramoto-Ising: ``J·ZZ + (K/2)(XY−YX)``.
        Parameters: ``J`` (default 1.0), ``K`` (default 1.5).
    """
    terms:  Optional[list[HamiltonianTerm]] = None
    preset: Optional[str] = None
    J:      float = 1.0
    h:      float = 1.0
    t:      float = 1.0
    K:      float = 1.5
    frame:  str   = "SCHRODINGER"   # "SCHRODINGER" | "INTERACTION"


class EvolveTrajectoryPoint(BaseModel):
    """A single time snapshot in a TEBD trajectory."""
    t:           float
    entropy:     Optional[list[float]] = None      # per-qubit entanglement entropy
    magnetization: Optional[list[float]] = None    # ⟨σᶻᵢ⟩ for each qubit
    energy:      Optional[float] = None            # ⟨H⟩
    qfi:         Optional[float] = None            # f_Q density
    correlators: Optional[list[float]] = None      # C(R) = ⟨Z₀Zᴿ⟩


class EvolveResult(BaseModel):
    """
    Result of a real-time or sudden-quench TEBD evolution.

    Attributes
    ----------
    n_qubits : int
    n_steps : int
    dt : float
    trajectory : list[EvolveTrajectoryPoint]
        Observable snapshots recorded every ``record_every`` steps.
    final_bond_entropy : list[float]
        Schmidt entropies at each MPS bond at the end of evolution.
    final_max_bond_dim : int
        Largest active bond dimension at end of evolution.
    truncation_error : float
        Accumulated SVD truncation error bound.
    """
    n_qubits:           int
    n_steps:            int
    dt:                 float
    trajectory:         list[dict]
    final_bond_entropy: list[float]
    final_max_bond_dim: int
    truncation_error:   float


class GroundStateResult(BaseModel):
    """
    Result of imaginary-time TEBD ground-state preparation.

    Attributes
    ----------
    n_qubits : int
    converged : bool
        Whether the Schmidt spectrum converged within ``n_steps``.
    steps : int
        Number of imaginary-time steps taken.
    energy : float or None
        Ground-state energy ``⟨H⟩``.
    bond_entropy : list[float]
        Entanglement entropy at each bond of the ground-state MPS.
    max_bond_dim : int
    """
    n_qubits:     int
    converged:    bool
    steps:        int
    energy:       Optional[float]
    bond_entropy: list[float]
    max_bond_dim: int


class QKZMResult(BaseModel):
    """
    Result of the full 3-phase QKZM protocol.

    Phases
    ------
    1. Imaginary-time ground state at ``h₀``.
    2. Linear ramp h(t) = h₀ → h_f crossing the critical point h_c = J.
    3. Post-quench real-time evolution at ``h_f``.

    Attributes
    ----------
    kzm_defect_density : float
        ⟨(1 − σᶻ)/2⟩ averaged over all qubits at the end of the post-quench phase.
        KZM prediction: n_d ∝ τ_Q^{−1/2} (TFIM universality, ν = z = 1).
    ground_state_energy : float or None
    ramp_trajectory : list[dict]
    post_trajectory : list[dict]
    final_bond_entropy : list[float]
    """
    n_qubits:               int
    ground_state_energy:    Optional[float]
    ground_state_converged: bool
    ground_state_steps:     int
    ramp_trajectory:        list[dict]
    post_trajectory:        list[dict]
    final_bond_entropy:     list[float]
    final_max_bond_dim:     int
    truncation_error:       float
    kzm_defect_density:     float


class LatticeResult(BaseModel):
    """
    Result of a 2D lattice snake-MPS regime classifier.

    Attributes
    ----------
    bond_entropy_2d : list[list[float]]
        n_rows × n_cols matrix of per-qubit entanglement entropy values,
        mapped back from the snake ordering to the original 2D grid topology.
        High-entropy sites = strongly correlated assets (market stress).
    bond_entropy : list[float]
        Raw 1D bond entropy along the snake (length n_qubits − 1).
    ground_energy : float or None
    converged : bool
    """
    n_qubits:        int
    grid_shape:      list[int]
    converged:       bool
    steps:           int
    ground_energy:   Optional[float]
    bond_entropy:    list[float]
    bond_entropy_2d: list[list[float]]
    max_bond_dim:    int
