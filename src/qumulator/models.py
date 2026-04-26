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
