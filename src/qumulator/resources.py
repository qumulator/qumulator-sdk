"""
Resource clients — one class per engine endpoint group.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from qumulator._http import _BaseClient, QumulatorHTTPError
from qumulator.models import (
    EvolveResult,
    GroundStateResult,
    HafnianResult,
    HamiltonianSpec,
    HomoResult,
    JobStatus,
    KLTResult,
    LatticeResult,
    QKZMResult,
)


class HomoClient(_BaseClient):
    """DFT HOMO/LUMO calculations via B3LYP/STO-3G."""

    def submit(self, smiles: str, basis: str = "sto-3g", xc: str = "b3lyp") -> str:
        """Submit a HOMO energy job. Returns job_id immediately."""
        data = self._post("/jobs/homo/energy", {"smiles": smiles, "basis": basis, "xc": xc})
        return data["job_id"]

    def run(
        self,
        smiles: str,
        basis: str = "sto-3g",
        xc: str = "b3lyp",
        timeout: float = 600.0,
    ) -> HomoResult:
        """Submit and block until the result is ready. Returns HomoResult."""
        status = self._submit_and_wait(
            "/homo/energy",
            {"smiles": smiles, "basis": basis, "xc": xc},
            timeout=timeout,
        )
        if not status.ok:
            raise QumulatorHTTPError(500, status.error or "Job failed")
        return HomoResult(**status.result)


class KLTClient(_BaseClient):
    """Ground-state energy solver for quantum spin systems and Hamiltonians."""

    def submit(
        self,
        interaction_matrix: Optional[list[list[float]]] = None,
        confinement_strength: float = 0.1,
        is_fermionic: bool = False,
        two_electron_tensor: Optional[list] = None,
        onsite_energies: Optional[list[float]] = None,
        pauli_pairs: Optional[list[list[int]]] = None,
        pauli_strength: float = 1.0,
        cluster_size: int = 2,
        pauli_hamiltonian: Optional[dict] = None,
    ) -> str:
        body = dict(
            interaction_matrix=interaction_matrix,
            confinement_strength=confinement_strength,
            is_fermionic=is_fermionic,
            two_electron_tensor=two_electron_tensor,
            onsite_energies=onsite_energies,
            pauli_pairs=pauli_pairs,
            pauli_strength=pauli_strength,
            cluster_size=cluster_size,
            pauli_hamiltonian=pauli_hamiltonian,
        )
        return self._post("/jobs/klt/relax", body)["job_id"]

    def run(
        self,
        interaction_matrix: Optional[list[list[float]]] = None,
        confinement_strength: float = 0.1,
        is_fermionic: bool = False,
        two_electron_tensor: Optional[list] = None,
        onsite_energies: Optional[list[float]] = None,
        pauli_pairs: Optional[list[list[int]]] = None,
        pauli_strength: float = 1.0,
        cluster_size: int = 2,
        timeout: float = 300.0,
        pauli_hamiltonian: Optional[dict] = None,
    ) -> KLTResult:
        """
        Run KLT relaxation and return the ground-state energy and diagnostics.

        Parameters
        ----------
        pauli_hamiltonian : dict[str, float], optional (recommended)
            Pauli string Hamiltonian. Keys are N-character strings over
            {I, X, Y, Z} (one character per qubit). Coefficients are floats.

            Example — H2 STO-3G (Jordan-Wigner, 2 qubits)::

                result = client.klt.run(
                    pauli_hamiltonian={
                        "II": g0,   # constant (nuclear repulsion + core energy)
                        "ZI": g1,   # Z on qubit 0
                        "IZ": g2,   # Z on qubit 1
                        "ZZ": g3,   # ZZ coupling
                        "XX": g4,   # XX exchange
                        "YY": g4,   # YY exchange  (same coeff as XX for H2)
                    },
                    cluster_size=2,
                )
                # result.energy is the full ground-state energy (g0 already included)

            The identity (II...I) constant term is automatically added to the
            returned energy — no manual offset needed.

        interaction_matrix : list[list[float]], optional
            Legacy J-matrix input. Must be square. Required if
            pauli_hamiltonian is not provided.
        """
        body = dict(
            interaction_matrix=interaction_matrix,
            confinement_strength=confinement_strength,
            is_fermionic=is_fermionic,
            two_electron_tensor=two_electron_tensor,
            onsite_energies=onsite_energies,
            pauli_pairs=pauli_pairs,
            pauli_strength=pauli_strength,
            cluster_size=cluster_size,
            pauli_hamiltonian=pauli_hamiltonian,
        )
        status = self._submit_and_wait("/klt/relax", body, timeout=timeout)
        if not status.ok:
            raise QumulatorHTTPError(500, status.error or "Job failed")
        return KLTResult(**status.result)


class HafnianClient(_BaseClient):
    """Hafnian and Gaussian boson sampling amplitude calculator."""

    def submit(
        self,
        matrix_real: list[list[float]],
        matrix_imag: Optional[list[list[float]]] = None,
        budget_bits: int = 4,
        threshold_sigma: float = 1.5,
        mp_dps: int = 34,
    ) -> str:
        body = dict(
            matrix_real=matrix_real,
            matrix_imag=matrix_imag,
            budget_bits=budget_bits,
            threshold_sigma=threshold_sigma,
            mp_dps=mp_dps,
        )
        return self._post("/jobs/hafnian", body)["job_id"]

    def run(
        self,
        matrix_real: list[list[float]],
        matrix_imag: Optional[list[list[float]]] = None,
        budget_bits: int = 4,
        threshold_sigma: float = 1.5,
        mp_dps: int = 34,
        timeout: float = 300.0,
    ) -> HafnianResult:
        body = dict(
            matrix_real=matrix_real,
            matrix_imag=matrix_imag,
            budget_bits=budget_bits,
            threshold_sigma=threshold_sigma,
            mp_dps=mp_dps,
        )
        status = self._submit_and_wait("/hafnian", body, timeout=timeout)
        if not status.ok:
            raise QumulatorHTTPError(500, status.error or "Job failed")
        return HafnianResult(**status.result)


class NotebookClient(_BaseClient):
    """Submit and execute Jupyter notebooks in the Qumulator sandbox."""

    def submit(self, notebook_bytes: bytes) -> str:
        """
        Submit a ``.ipynb`` notebook for execution.  Returns ``job_id``.

        The notebook is executed in an isolated sandbox environment with
        ``qumulator-sdk`` and standard scientific libraries pre-installed.

        Parameters
        ----------
        notebook_bytes : bytes
            Raw ``.ipynb`` file content (``open('my_notebook.ipynb', 'rb').read()``).

        Returns
        -------
        str
            Job ID.  Poll :meth:`status` or use :meth:`run` to wait for the result.
        """
        import httpx
        with httpx.Client(
            base_url=self._api_url,
            headers={k: v for k, v in self._headers.items() if k != "Content-Type"},
            timeout=30.0,
        ) as client:
            resp = client.post(
                "/notebooks",
                content=notebook_bytes,
                headers={"Content-Type": "application/octet-stream"},
            )
            self._raise_for_status(resp)
            return resp.json()["job_id"]

    def status(self, job_id: str) -> dict:
        """
        Poll for the status of a submitted notebook job.

        Returns a dict with keys: ``job_id``, ``status``, ``success``,
        ``error``, ``outputs``, ``output_notebook``.
        """
        return self._get(f"/notebooks/{job_id}")

    def run(
        self,
        notebook_bytes: bytes,
        timeout: float = 600.0,
    ) -> dict:
        """
        Submit a notebook and block until execution completes.

        Parameters
        ----------
        notebook_bytes : bytes
            Raw ``.ipynb`` file content.
        timeout : float
            Maximum seconds to wait for completion.

        Returns
        -------
        dict
            Execution result with keys: ``job_id``, ``status``, ``success``,
            ``outputs`` (list of per-cell output dicts), ``error``.
        """
        import time as _time

        job_id = self.submit(notebook_bytes)
        deadline = _time.monotonic() + timeout
        while True:
            result = self.status(job_id)
            if result.get("status") in ("completed", "failed"):
                return result
            if _time.monotonic() > deadline:
                raise TimeoutError(
                    f"Notebook job {job_id} did not complete within {timeout:.0f}s"
                )
            _time.sleep(3.0)


class EvolveClient(_BaseClient):
    """
    Hamiltonian time evolution via TEBD.

    All methods call the ``/evolve`` backend endpoints directly (inline, not
    async job-queue).  The server enforces a hard timeout; use the
    ``timeout`` parameter to match your client-side expectation.

    Quick start
    -----------
    ::

        client = QumulatorClient(api_url=..., api_key=...)

        # Real-time evolution of 10-site TFIM
        result = client.evolve.run(
            n_qubits=10,
            hamiltonian={"preset": "ising_1d", "J": 1.0, "h": 1.0},
            t_max=2.0,
            dt=0.05,
            observables=["entropy", "qfi", "magnetization"],
        )
        for pt in result.trajectory:
            print(pt["t"], pt.get("qfi"))

        # Ground state
        gs = client.evolve.ground(
            n_qubits=10,
            hamiltonian={"preset": "ising_1d", "J": 1.0, "h": 1.0},
        )
        print(gs.energy)       # ground-state energy
        print(gs.bond_entropy) # entanglement profile

        # QKZM quench
        qkzm = client.evolve.qkzm(n_qubits=20, J=1.0, h0=5.0, h_f=0.2, t_ramp=5.0)
        print(qkzm.kzm_defect_density)   # n_d ∝ τ_Q^{-1/2}

        # Collapse-and-revival
        revival = client.evolve.quench(n_qubits=20, h=2.0, t_max=10.0)

        # Multi-asset 2D lattice
        lattice = client.evolve.lattice(n_rows=4, n_cols=4)
        print(lattice.bond_entropy_2d)  # 4×4 entanglement heatmap
    """

    # ── Real-time evolution ───────────────────────────────────────────────

    def run(
        self,
        n_qubits: int,
        hamiltonian: dict | HamiltonianSpec,
        t_max: float = 1.0,
        dt: float = 0.1,
        bond_dim: int = 64,
        observables: list[str] | None = None,
        record_every: int = 1,
        order: int = 2,
        initial_state: str = "zero",
        timeout: float = 300.0,
    ) -> EvolveResult:
        """
        Real-time TEBD evolution under a Pauli-sum Hamiltonian.

        Parameters
        ----------
        n_qubits : int
            System size (2–200).
        hamiltonian : dict or HamiltonianSpec
            Use a preset: ``{"preset": "ising_1d", "J": 1.0, "h": 1.0}``,
            or custom terms: ``{"terms": [{"sites": [0,1], "operator": "ZZ",
            "strength": -1.0}, ...]}``.
        t_max : float
            Total evolution time.
        dt : float
            Trotter step size.
        bond_dim : int
            MPS bond dimension χ (max 1024).
        observables : list[str]
            Any of: ``"entropy"``, ``"magnetization"``, ``"energy"``,
            ``"qfi"``, ``"correlators"``.
        record_every : int
            Record observables every N Trotter steps.
        order : int
            Trotter order: 1 (first) or 2 (second, default).
        initial_state : str
            ``"zero"`` (|0⟩^N), ``"ferromagnet"`` (|↑⟩^N), or
            ``"neel"`` (|↑↓↑↓…⟩).
        timeout : float
            Client-side request timeout in seconds.

        Returns
        -------
        EvolveResult
        """
        if observables is None:
            observables = ["entropy", "magnetization"]
        ham = hamiltonian.model_dump() if isinstance(hamiltonian, HamiltonianSpec) else hamiltonian
        body = dict(
            n_qubits=n_qubits,
            hamiltonian=ham,
            t_max=t_max,
            dt=dt,
            bond_dim=bond_dim,
            observables=observables,
            record_every=record_every,
            order=order,
            initial_state=initial_state,
            timeout_seconds=timeout,
        )
        data = self._post_timeout("/evolve", body, timeout=timeout + 10.0)
        return EvolveResult(**data)

    # ── Sudden quench ─────────────────────────────────────────────────────

    def quench(
        self,
        n_qubits: int,
        J: float = 1.0,
        h: float = 2.0,
        t_max: float = 10.0,
        dt: float = 0.02,
        bond_dim: int = 128,
        observables: list[str] | None = None,
        record_every: int = 5,
        frame: str = "SCHRODINGER",
        timeout: float = 300.0,
    ) -> EvolveResult:
        """
        Sudden quench from ferromagnetic |↑⟩^N into the TFIM.

        Produces collapse-and-revival oscillations in ⟨σᶻ⟩ and stepwise
        QFI growth (Section IV.1 of arXiv:2604.05032).
        Revival period T ≈ 2π/h (use h = 2*J for collapse-and-revival regime).

        Parameters
        ----------
        n_qubits : int
        J : float  Ising ZZ coupling.
        h : float  Transverse field (h = 2*J gives collapse-and-revival).
        t_max : float
        dt : float
        bond_dim : int
        observables : list[str]
        record_every : int
        frame : str  ``"SCHRODINGER"`` or ``"INTERACTION"`` (rotating frame).
        timeout : float
        """
        if observables is None:
            observables = ["entropy", "magnetization", "qfi"]
        body = dict(
            n_qubits=n_qubits,
            J=J, h=h, t_max=t_max, dt=dt,
            bond_dim=bond_dim,
            observables=observables,
            record_every=record_every,
            frame=frame,
            timeout_seconds=timeout,
        )
        data = self._post_timeout("/evolve/quench", body, timeout=timeout + 10.0)
        return EvolveResult(**data)

    # ── Ground state ──────────────────────────────────────────────────────

    def ground(
        self,
        n_qubits: int,
        hamiltonian: dict | HamiltonianSpec,
        dtau: float = 0.05,
        n_steps: int = 200,
        converge_tol: float = 1e-6,
        bond_dim: int = 64,
        timeout: float = 300.0,
    ) -> GroundStateResult:
        """
        Imaginary-time TEBD ground-state preparation.

        Applies exp(−H·dτ) iteratively until the Schmidt spectrum converges.
        Returns the ground-state energy and bond-entropy profile.

        Parameters
        ----------
        n_qubits : int
        hamiltonian : dict or HamiltonianSpec
        dtau : float  Imaginary time step.
        n_steps : int  Maximum imaginary-time steps.
        converge_tol : float  Convergence criterion on Schmidt spectrum change.
        bond_dim : int
        timeout : float
        """
        ham = hamiltonian.model_dump() if isinstance(hamiltonian, HamiltonianSpec) else hamiltonian
        body = dict(
            n_qubits=n_qubits,
            hamiltonian=ham,
            dtau=dtau,
            n_steps=n_steps,
            converge_tol=converge_tol,
            bond_dim=bond_dim,
            timeout_seconds=timeout,
        )
        data = self._post_timeout("/evolve/ground", body, timeout=timeout + 10.0)
        return GroundStateResult(**data)

    # ── QKZM protocol ─────────────────────────────────────────────────────

    def qkzm(
        self,
        n_qubits: int,
        J: float = 1.0,
        h0: float = 5.0,
        h_f: float = 0.2,
        t_ramp: float = 5.0,
        t_post: float = 2.0,
        dt: float = 0.05,
        bond_dim: int = 128,
        observables: list[str] | None = None,
        record_every: int = 5,
        dtau: float = 0.05,
        n_prep_steps: int = 200,
        converge_tol: float = 1e-5,
        timeout: float = 300.0,
    ) -> QKZMResult:
        """
        Full 3-phase Quantum Kibble-Zurek Mechanism protocol.

        Phase 1: Imaginary-time TEBD → ground state |GS(h₀)⟩.
        Phase 2: Linear ramp h(t) = h₀ → h_f crossing h_c = J.
        Phase 3: Post-quench dynamics at h_f.

        KZM prediction (TFIM, ν = z = 1):
            defect density n_d ∝ τ_Q^{−1/2}
            where τ_Q = (h₀ − h_f) / |dh/dt| is the ramp timescale.

        Parameters
        ----------
        n_qubits : int
        J : float  Ising ZZ coupling and critical field h_c = J.
        h0 : float  Initial (paramagnetic) transverse field (>> J).
        h_f : float  Final (ferromagnetic) transverse field (<< J).
        t_ramp : float  Duration of the linear ramp (controls τ_Q).
        t_post : float  Post-quench observation window.
        dt : float  Trotter step size.
        bond_dim : int
        observables : list[str]
        record_every : int
        dtau : float  Imaginary-time step for Phase 1.
        n_prep_steps : int  Max imaginary-time steps in Phase 1.
        converge_tol : float
        timeout : float
        """
        if observables is None:
            observables = ["entropy", "magnetization", "qfi"]
        body = dict(
            n_qubits=n_qubits,
            J=J, h0=h0, h_f=h_f,
            t_ramp=t_ramp, t_post=t_post, dt=dt,
            bond_dim=bond_dim,
            observables=observables,
            record_every=record_every,
            dtau=dtau,
            n_prep_steps=n_prep_steps,
            converge_tol=converge_tol,
            timeout_seconds=timeout,
        )
        data = self._post_timeout("/evolve/qkzm", body, timeout=timeout + 10.0)
        return QKZMResult(**data)

    # ── 2D Lattice ─────────────────────────────────────────────────────────

    def lattice(
        self,
        n_rows: int = 4,
        n_cols: int = 4,
        coupling_matrix: list[list[float]] | None = None,
        J_default: float = 1.0,
        h_transverse: float = 0.5,
        bond_dim: int = 128,
        dtau: float = 0.05,
        n_steps: int = 100,
        converge_tol: float = 1e-5,
        timeout: float = 300.0,
    ) -> LatticeResult:
        """
        Multi-asset 2D lattice regime classifier via snake-MPS.

        Arranges n_rows × n_cols assets on a grid, snakes to 1D, and computes
        the imaginary-time ground-state bond-entropy map.

        The 2D entropy heatmap is a proxy for cross-asset entanglement:
        - High-entropy bonds: strongly correlated asset pairs.
        - Sudden entropy collapse across the grid: regime change / market stress.

        Bond couplings can be set from a pairwise correlation matrix to encode
        the return structure of the asset universe.

        Parameters
        ----------
        n_rows : int  Grid rows (2–8).
        n_cols : int  Grid columns (2–8).
        coupling_matrix : list[list[float]], optional
            n × n pairwise coupling matrix J_ij. Uses J_default if not provided.
        J_default : float  Uniform coupling when coupling_matrix is None.
        h_transverse : float  Single-site transverse field.
        bond_dim : int  MPS bond dimension χ (max 512).
        dtau : float  Imaginary time step.
        n_steps : int  Max imaginary-time steps (max 2000).
        converge_tol : float
        timeout : float

        Returns
        -------
        LatticeResult
            Includes ``bond_entropy_2d`` (n_rows × n_cols heatmap) and
            ``ground_energy``.
        """
        body = dict(
            n_rows=n_rows,
            n_cols=n_cols,
            coupling_matrix=coupling_matrix,
            J_default=J_default,
            h_transverse=h_transverse,
            bond_dim=bond_dim,
            dtau=dtau,
            n_steps=n_steps,
            converge_tol=converge_tol,
            timeout_seconds=timeout,
        )
        data = self._post_timeout("/evolve/lattice", body, timeout=timeout + 10.0)
        return LatticeResult(**data)

    # ── Internal helper ───────────────────────────────────────────────────

    def _post_timeout(self, path: str, body: dict, timeout: float) -> dict:
        """POST with a longer httpx timeout to accommodate slow TEBD jobs."""
        import httpx
        with httpx.Client(
            base_url=self._api_url,
            headers=self._headers,
            timeout=timeout,
        ) as client:
            resp = client.post(path, json=body)
            self._raise_for_status(resp)
            return resp.json()
