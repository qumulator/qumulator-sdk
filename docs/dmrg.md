# DMRG Ground-State Energy

Compute the ground-state energy of a molecular active space using
**two-site Density Matrix Renormalization Group (DMRG)** with Numba JIT acceleration.

No circuit ansatz is required. DMRG is a variational eigenvalue solver that
systematically approaches the exact ground state as bond dimension d_max increases.
At d_max = 64 it is **exact (machine precision) for n_orb ≤ 6**.

For larger active spaces (n_orb > 30) or multi-fragment molecules, use
[MPS/MPO](molecular-mps.md) instead.

---

## Quick start

```python
import numpy as np
from pyscf import gto, scf, mcscf, ao2mo
from qumulator import QumulatorClient

client = QumulatorClient()   # reads QUMULATOR_API_URL, QUMULATOR_API_KEY from env

# ── H2 CAS(2,2) ───────────────────────────────────────────────────────────
mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", spin=0, charge=0)
mf  = scf.RHF(mol).run()
mc  = mcscf.CASSCF(mf, ncas=2, nelecas=2).run()

h1e, e_core = mc.get_h1eff()
h2e = ao2mo.restore(1, mc.get_h2eff(), mc.ncas)
e_nuc = mc.energy_nuc() + e_core

result = client.dmrg.energy(
    h1e=h1e.tolist(),
    h2e=h2e.tolist(),
    n_elec=list(mc.nelecas),
    e_nuc=float(e_nuc),
    d_max=64,
)

print(f"E(DMRG) = {result.energy:.10f} Ha")   # −1.1372838 Ha (exact FCI)
print(f"converged = {result.converged}")
print(f"sweeps    = {result.n_sweeps_run}")
print(f"wall time = {result.wall_time_s:.2f} s")
```

---

## Water CAS(7,7)

```python
# H2O 6-31g* active space: 7 orbitals, 7 electrons
result = client.dmrg.energy(
    h1e=h1e.tolist(),    # 7×7
    h2e=h2e.tolist(),    # 7×7×7×7
    n_elec=[4, 3],
    e_nuc=e_nuc,
    d_max=64,
    n_sweeps=8,
)
print(f"E(DMRG/H2O) = {result.energy:.6f} Ha")
```

---

## Convergence guide

| n_orb | Recommended d_max | Typical time |
|---|---|---|
| ≤ 6 | 64 (exact) | < 1 s |
| 7–12 | 64–128 | 1–30 s |
| 13–20 | 128–256 | 30–300 s |
| 21–30 | 256–512 | 300–3600 s |

If `converged=False`, increase `d_max` or `n_sweeps` and resubmit.

---

## DMRG vs MPS/MPO

| Criterion | DMRG (`client.dmrg`) | MPS (`client.molecular`) |
|---|---|---|
| Active space size | ≤ 30 orbitals | ≤ 50 orbitals |
| Correlation type | 1D-like chains | Multi-fragment / branched |
| Circuit ansatz | Not required | Optional Givens circuit |
| Accuracy control | d_max + n_sweeps | MPO bond dimension |

---

## DMRGEnergyResult fields

| Field | Type | Description |
|---|---|---|
| `energy` | `float` | Total energy including e_nuc (Ha) |
| `converged` | `bool` | `True` when `|ΔE| < tol` before exhausting sweeps |
| `n_sweeps_run` | `int` | Number of sweeps performed |
| `d_max_used` | `int` | Bond dimension as requested |
| `n_orb` | `int` | Active molecular orbitals |
| `n_so` | `int` | Spin-orbitals = 2 × n_orb |
| `wall_time_s` | `float` | Wall-clock time in seconds |

---

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `h1e` | `list[list[float]]` | required | n_orb × n_orb 1-electron integrals |
| `h2e` | `list[list[list[list[float]]]]` | required | n_orb⁴ 2-electron integrals |
| `n_elec` | `list[int]` | required | `[n_alpha, n_beta]` active electrons |
| `e_nuc` | `float` | `0.0` | Nuclear repulsion + core energy (Ha) |
| `d_max` | `int` | `64` | Max MPS bond dimension (1–512) |
| `n_sweeps` | `int` | `8` | Max DMRG sweeps (1–50) |
| `tol` | `float` | `1e-10` | Convergence threshold `|ΔE|` per sweep |
| `timeout` | `float` | `900.0` | Client request timeout (s) |

---

## Limits

| Limit | Value |
|---|---|
| Max n_orb | 30 |
| Max d_max | 512 |
| Max n_sweeps | 50 |
| Rate limit | 60 req/min per API key |

---

## See also

- [MPS/MPO Molecular Energy](molecular-mps.md) — larger active spaces, multi-fragment
- [Molecular Orbitals (HOMO/LUMO)](molecular.md) — DFT frontier orbital energies from SMILES
