"""
Gate matrix constants for the local statevector simulator.

All gates are standard, published definitions. References:
  - Nielsen & Chuang, "Quantum Computation and Quantum Information" (2000)
  - IBM Qiskit gate documentation: https://docs.quantum.ibm.com/api/qiskit/circuit_library
  - Google Cirq gate documentation: https://quantumai.google/cirq/gates
  - Arute et al., "Quantum supremacy using a programmable superconducting processor",
    Nature 574, 505-510 (2019) — fSIM / SYC gate definitions
"""
from __future__ import annotations

import math

import numpy as np

_R2 = 1.0 / math.sqrt(2)

# ---------------------------------------------------------------------------
#  Single-qubit fixed gates
# ---------------------------------------------------------------------------

GATES_1Q: dict[str, np.ndarray] = {
    "h":    np.array([[_R2,  _R2], [_R2, -_R2]], dtype=complex),
    "x":    np.array([[0, 1], [1, 0]], dtype=complex),
    "y":    np.array([[0, -1j], [1j, 0]], dtype=complex),
    "z":    np.array([[1, 0], [0, -1]], dtype=complex),
    "s":    np.array([[1, 0], [0, 1j]], dtype=complex),
    "sdg":  np.array([[1, 0], [0, -1j]], dtype=complex),
    "t":    np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex),
    "tdg":  np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex),
    # SX = sqrt(X)
    "sx":   np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=complex) * 0.5,
    "sxdg": np.array([[1 - 1j, 1 + 1j], [1 + 1j, 1 - 1j]], dtype=complex) * 0.5,
    "id":   np.eye(2, dtype=complex),
    "i":    np.eye(2, dtype=complex),
    # Google Willow / Sycamore single-qubit gate set
    # SY = e^{iπ/4} Ry(π/2), SY^2 = Y  [Arute et al. 2019, Supplementary]
    "sy":   np.array([[1 + 1j, -1 - 1j], [1 + 1j, 1 + 1j]], dtype=complex) * 0.5,
    "sydg": np.array([[1 - 1j, 1 - 1j], [-(1 - 1j), 1 - 1j]], dtype=complex) * 0.5,
    # SW = e^{iπ/4} Rw(π/2) where W = (X+Y)/√2, SW^2 = W
    "sw":   np.array([[1 + 1j, -1j * _R2 * 2], [_R2 * 2, 1 + 1j]], dtype=complex) * 0.5,
    "swdg": np.array([[1 - 1j, _R2 * 2], [1j * _R2 * 2, 1 - 1j]], dtype=complex) * 0.5,
}

# Gate name aliases (all resolve to canonical names above)
_ALIASES_1Q: dict[str, str] = {
    "identity": "id",
    "cnot": "x",   # single-qubit alias only — full CNOT is in 2Q gates
}

# ---------------------------------------------------------------------------
#  Single-qubit parametric gates
# ---------------------------------------------------------------------------

def rx(theta: float) -> np.ndarray:
    """Rx(θ) = exp(-iθX/2)."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def ry(theta: float) -> np.ndarray:
    """Ry(θ) = exp(-iθY/2)."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(theta: float) -> np.ndarray:
    """Rz(θ) = exp(-iθZ/2)."""
    h = theta / 2
    return np.array([[np.exp(-1j * h), 0], [0, np.exp(1j * h)]], dtype=complex)


def phase(theta: float) -> np.ndarray:
    """Phase(θ) = diag(1, e^{iθ}). Equivalent to Rz up to global phase."""
    return np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=complex)


def u(theta: float, phi: float, lam: float) -> np.ndarray:
    """IBM U(θ,φ,λ) gate — the most general single-qubit unitary up to global phase."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array(
        [[c, -np.exp(1j * lam) * s],
         [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]],
        dtype=complex,
    )


# ---------------------------------------------------------------------------
#  Two-qubit fixed gates
# ---------------------------------------------------------------------------

CNOT  = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
CZ    = np.diag([1, 1, 1, -1]).astype(complex)
CY    = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,-1j],[0,0,1j,0]], dtype=complex)
CH    = np.eye(4, dtype=complex); CH[2:4, 2:4] = GATES_1Q["h"]
CSX   = np.eye(4, dtype=complex); CSX[2:4, 2:4] = GATES_1Q["sx"]
SWAP  = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
ISWAP = np.array([[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]], dtype=complex)
ECR   = np.array(
    [[0,0,_R2,1j*_R2],[0,0,1j*_R2,_R2],[_R2,-1j*_R2,0,0],[-1j*_R2,_R2,0,0]],
    dtype=complex,
)

GATES_2Q: dict[str, np.ndarray] = {
    "cnot": CNOT, "cx": CNOT,
    "cz": CZ,
    "cy": CY,
    "ch": CH,
    "csx": CSX,
    "swap": SWAP,
    "iswap": ISWAP,
    "ecr": ECR,
}

# ---------------------------------------------------------------------------
#  Two-qubit parametric gates
# ---------------------------------------------------------------------------

def cp(theta: float) -> np.ndarray:
    """Controlled-phase: diag(1,1,1,e^{iθ})."""
    return np.diag([1, 1, 1, np.exp(1j * theta)]).astype(complex)


def crx(theta: float) -> np.ndarray:
    """Controlled-Rx(θ)."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[1,0,0,0],[0,1,0,0],[0,0,c,-1j*s],[0,0,-1j*s,c]], dtype=complex)


def cry(theta: float) -> np.ndarray:
    """Controlled-Ry(θ)."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[1,0,0,0],[0,1,0,0],[0,0,c,-s],[0,0,s,c]], dtype=complex)


def crz(theta: float) -> np.ndarray:
    """Controlled-Rz(θ)."""
    h = theta / 2
    return np.array(
        [[1,0,0,0],[0,1,0,0],[0,0,np.exp(-1j*h),0],[0,0,0,np.exp(1j*h)]],
        dtype=complex,
    )


def rxx(theta: float) -> np.ndarray:
    """Rxx(θ) = exp(-iθ X⊗X / 2)."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c,0,0,-1j*s],[0,c,-1j*s,0],[0,-1j*s,c,0],[-1j*s,0,0,c]], dtype=complex)


def ryy(theta: float) -> np.ndarray:
    """Ryy(θ) = exp(-iθ Y⊗Y / 2)."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c,0,0,1j*s],[0,c,-1j*s,0],[0,-1j*s,c,0],[1j*s,0,0,c]], dtype=complex)


def rzz(theta: float) -> np.ndarray:
    """Rzz(θ) = exp(-iθ Z⊗Z / 2)."""
    h = theta / 2
    return np.diag(
        [np.exp(-1j * h), np.exp(1j * h), np.exp(1j * h), np.exp(-1j * h)]
    ).astype(complex)


def fsim(theta: float, phi: float) -> np.ndarray:
    """fSIM(θ,φ) — Google Sycamore-family parametric two-qubit gate.

    Reference: Arute et al., Nature 574, 505-510 (2019), Supplementary §S1.
    """
    c, s = math.cos(theta), math.sin(theta)
    return np.array(
        [[1, 0,       0,               0            ],
         [0, c,      -1j * s,          0            ],
         [0, -1j * s, c,               0            ],
         [0, 0,       0,  np.exp(-1j * phi)         ]],
        dtype=complex,
    )


# Google Sycamore gate: fSIM(π/2, π/6)  [Arute et al. 2019]
SYC = fsim(math.pi / 2, math.pi / 6)

# ---------------------------------------------------------------------------
#  Three-qubit gates
#  Convention: qubits = [q0, q1, q2], state index = q0*4 + q1*2 + q2 (MSB=q0)
# ---------------------------------------------------------------------------

# Toffoli (CCX): control q0 AND q1, flip q2
CCX = np.eye(8, dtype=complex)
CCX[[6, 7], :] = CCX[[7, 6], :]

# Fredkin (CSWAP): control q0, swap q1 and q2
CSWAP = np.eye(8, dtype=complex)
CSWAP[[5, 6], :] = CSWAP[[6, 5], :]

GATES_3Q: dict[str, np.ndarray] = {
    "ccx": CCX, "toffoli": CCX,
    "cswap": CSWAP, "fredkin": CSWAP,
}

# ---------------------------------------------------------------------------
#  Gate resolver: name + optional params → unitary matrix
# ---------------------------------------------------------------------------

_PARAM_1Q: dict[str, object] = {
    "rx": rx, "ry": ry, "rz": rz,
    "phase": phase, "p": phase,
    "u": u, "u3": u,
}

_PARAM_2Q: dict[str, object] = {
    "cp": cp, "cu1": cp,
    "crx": crx, "cry": cry, "crz": crz,
    "rxx": rxx, "ryy": ryy, "rzz": rzz,
    "fsim": fsim,
    "syc": lambda: SYC,
}


def resolve_gate(name: str, params: list | None) -> np.ndarray:
    """Return the unitary matrix for *name* with optional *params*."""
    n = name.lower().strip().replace("-", "_")

    # 1-qubit fixed
    if n in GATES_1Q:
        return GATES_1Q[n]

    # 2-qubit fixed
    if n in GATES_2Q:
        return GATES_2Q[n]

    # 3-qubit fixed
    if n in GATES_3Q:
        return GATES_3Q[n]

    # SYC (no params)
    if n == "syc":
        return SYC

    # 1-qubit parametric
    if n in _PARAM_1Q:
        p = _coerce_params(params)
        fn = _PARAM_1Q[n]
        return fn(*p)  # type: ignore[operator]

    # 2-qubit parametric
    if n in _PARAM_2Q:
        if n == "syc":
            return SYC
        p = _coerce_params(params)
        fn = _PARAM_2Q[n]
        return fn(*p)  # type: ignore[operator]

    raise ValueError(
        f"Unknown gate '{name}'. "
        f"Supported: {sorted(GATES_1Q) + sorted(GATES_2Q) + sorted(GATES_3Q) + list(_PARAM_1Q) + list(_PARAM_2Q)}"
    )


def _coerce_params(params) -> list:
    if params is None:
        return []
    if isinstance(params, (int, float)):
        return [float(params)]
    return [float(p) for p in params]
