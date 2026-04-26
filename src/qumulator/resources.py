"""
Resource clients — one class per engine endpoint group.
"""
from __future__ import annotations

import json
from typing import Optional

from qumulator._http import _BaseClient, QumulatorHTTPError
from qumulator.models import (
    HafnianResult,
    HomoResult,
    JobStatus,
    KLTResult,
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
