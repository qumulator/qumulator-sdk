"""
Qumulator SDK â€” quantum computing API client.

Usage
-----
::

    import os
    from qumulator import QumulatorClient

    client = QumulatorClient(
        api_url=os.environ["QUMULATOR_API_URL"],
        api_key=os.environ["QUMULATOR_API_KEY"],
    )

    # --- Circuit simulation ---
    eng = client.circuit.engine(n_qubits=2)
    eng.apply('h', 0).apply('cx', [0, 1])
    print(eng.sample(shots=1024))   # {'00': ~512, '11': ~512}

    result = eng.run(shots=2048, return_entropy_map=True)
    print(result.entropy_map)       # [~1.0, ~1.0]

    # --- DFT HOMO/LUMO ---
    homo = client.homo.run("Oc1ccc(/C=C/c2cc(O)cc(O)c2)cc1")
    print(homo.homo_E_eV)           # e.g. -3.488

    # --- Energy optimization ---
    import numpy as np
    J = np.random.randn(8, 8); J = (J + J.T) / 2
    energy = client.klt.run(J.tolist())
    print(energy.energy)

    # --- Hafnian ---
    A = np.random.randn(8, 8); A = (A + A.T) / 2
    h = client.hafnian.run(A.tolist())
    print(h.value)

    # --- Hamiltonian time evolution (TEBD) ---
    result = client.evolve.run(
        n_qubits=10,
        hamiltonian={"preset": "ising_1d", "J": 1.0, "h": 1.0},
        t_max=2.0, dt=0.05,
        observables=["entropy", "qfi", "magnetization"],
    )
    for pt in result.trajectory:
        print(pt["t"], pt.get("qfi"))

    # --- Ground state ---
    gs = client.evolve.ground(
        n_qubits=10,
        hamiltonian={"preset": "ising_1d", "J": 1.0, "h": 1.0},
    )
    print(gs.energy, gs.bond_entropy)

    # --- QKZM quench ---
    q = client.evolve.qkzm(n_qubits=20, J=1.0, h0=5.0, h_f=0.2, t_ramp=5.0)
    print(q.kzm_defect_density)   # n_d ∝ τ_Q^{-1/2}

    # --- Multi-asset 2D lattice ---
    lat = client.evolve.lattice(n_rows=4, n_cols=4)
    print(lat.bond_entropy_2d)    # 4×4 entanglement heatmap
"""

from qumulator._http import QumulatorHTTPError
from qumulator.circuit import CircuitClient, CircuitEngine, CircuitResult
from qumulator.models import (
    EvolveResult,
    GaussianCertificate,
    GroundStateResult,
    HafnianResult,
    HamiltonianSpec,
    HamiltonianTerm,
    HomoResult,
    JobStatus,
    KLTResult,
    LatticeResult,
    QKZMResult,
)
from qumulator.resources import (
    EvolveClient,
    HafnianClient,
    HomoClient,
    KLTClient,
    NotebookClient,
)


class QumulatorClient:
    """
    Top-level Qumulator API client.

    Bundles all service endpoints under a single authenticated connection.

    Parameters
    ----------
    api_url : str
        Base URL of the Qumulator service,
        e.g. ``"https://api.qumulator.com"``.
    api_key : str
        API key for authentication.

    Attributes
    ----------
    circuit : CircuitClient
        Quantum circuit simulation.
    homo : HomoClient
        DFT HOMO/LUMO energy calculations.
    klt : KLTClient
        Ground-state energy optimization.
    hafnian : HafnianClient
        Hafnian estimation for Gaussian boson sampling.
    evolve : EvolveClient
        Hamiltonian time evolution (TEBD): real-time, imaginary-time,
        sudden quench, QKZM protocol, and 2D lattice regime classifier.
    """

    def __init__(self, api_url: str | None = None, api_key: str | None = None) -> None:
        import os
        if api_url is None:
            api_url = os.environ.get("QUMULATOR_API_URL", "https://api.qumulator.com")
        if api_key is None:
            api_key = os.environ.get("QUMULATOR_API_KEY", "")
        self.circuit  = CircuitClient(api_url, api_key)
        self.homo     = HomoClient(api_url, api_key)
        self.klt      = KLTClient(api_url, api_key)
        self.hafnian  = HafnianClient(api_url, api_key)
        self.notebook = NotebookClient(api_url, api_key)
        self.evolve   = EvolveClient(api_url, api_key)


__all__ = [
    # Top-level client
    "QumulatorClient",
    # Circuit
    "CircuitClient",
    "CircuitEngine",
    "CircuitResult",
    # Sub-clients
    "HomoClient",
    "KLTClient",
    "HafnianClient",
    "NotebookClient",
    "EvolveClient",
    # Result models
    "HomoResult",
    "KLTResult",
    "HafnianResult",
    "GaussianCertificate",
    "JobStatus",
    # Evolve models
    "HamiltonianTerm",
    "HamiltonianSpec",
    "EvolveResult",
    "GroundStateResult",
    "QKZMResult",
    "LatticeResult",
    # Errors
    "QumulatorHTTPError",
]

