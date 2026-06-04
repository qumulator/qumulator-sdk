# Molecular Energy — MPS/MPO

Compute the ground-state energy of a molecular active space using
**Matrix Product State / Matrix Product Operator (MPS/MPO)**.

This endpoint accepts 1-electron and 2-electron integrals from any quantum
chemistry package (PySCF, ORCA, Gaussian) and evaluates ⟨ψ|H|ψ⟩ + e_nuc.

- **No circuit required** — returns the Hartree-Fock reference energy by default.
- **Optional Givens circuit** — rotate the reference MPS to include correlation.
- **Supports up to 50 orbitals** (100 spin-orbital qubits).
- **Multi-fragment molecules** — Aspirin, Ibuprofen, DNA bases, pharma fragments.

For smaller active spaces (n_orb ≤ 30) where you want pure variational accuracy
without a circuit ansatz, see [DMRG](dmrg.md) instead.

---

## Quick start

```python
import numpy as np
from pyscf import gto, scf, mcscf, ao2mo
from qumulator import QumulatorClient

client = QumulatorClient()   # reads QUMULATOR_API_URL, QUMULATOR_API_KEY from env

# ── Build molecule in PySCF ───────────────────────────────────────────────
mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", spin=0, charge=0)
mf  = scf.RHF(mol).run()
mc  = mcscf.CASSCF(mf, ncas=2, nelecas=2).run()

# ── Get active-space integrals ────────────────────────────────────────────
h1e, e_core = mc.get_h1eff()
h2e = ao2mo.restore(1, mc.get_h2eff(), mc.ncas)
e_nuc = mc.energy_nuc() + e_core

# ── MPS energy (Hartree-Fock reference, no circuit) ──────────────────────
result = client.molecular.energy(
    h1e=h1e.tolist(),
    h2e=h2e.tolist(),
    n_elec=list(mc.nelecas),
    e_nuc=float(e_nuc),
)
print(f"E(HF)   = {result.energy:.8f} Ha")
print(f"n_orb   = {result.n_orb}, n_qubits = {result.n_qubits}")
print(f"Fragments detected: {result.n_components}")
```

---

## Adding a Givens circuit

Givens gates are 2-qubit orbital rotations that evolve the reference MPS through
the correlation energy landscape.

```python
# Single Givens rotation between spin-orbitals 0 and 2
result = client.molecular.energy(
    h1e=h1e.tolist(),
    h2e=h2e.tolist(),
    n_elec=list(mc.nelecas),
    e_nuc=float(e_nuc),
    circuit=[
        {"qi": 0, "qj": 2, "theta": 0.15},
        {"qi": 1, "qj": 3, "theta": 0.15},
    ],
)
print(f"E(MPS) = {result.energy:.8f} Ha")
```

**Qubit convention (Jordan-Wigner)**:
- Qubit `2p` = α spin-orbital p
- Qubit `2p+1` = β spin-orbital p

---

## Pharmaceutical fragment example

```python
# Aspirin fragment — CAS(8,8)
result = client.molecular.energy(
    h1e=h1e.tolist(),   # 8×8
    h2e=h2e.tolist(),   # 8×8×8×8
    n_elec=[4, 4],
    e_nuc=e_nuc,
    circuit=givens_gates,  # Givens circuit from VQE-style optimizer
)
print(f"E(MPS/aspirin) = {result.energy:.6f} Ha")
print(f"ZZ correlators: {result.zz_correlators}")
```

---

## MolecularEnergyResult fields

| Field | Type | Description |
|---|---|---|
| `energy` | `float` | Total energy including e_nuc (Ha) |
| `n_qubits` | `int` | 2 × n_orb spin-orbital qubits |
| `n_orb` | `int` | Active molecular orbitals |
| `n_components` | `int` | Number of MPS fragment components discovered |
| `zz_correlators` | `list[list[float]]` or `None` | ⟨ZᵢZⱼ⟩ correlator matrix |

---

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `h1e` | `list[list[float]]` | required | n_orb × n_orb 1-electron integrals |
| `h2e` | `list[list[list[list[float]]]]` | required | n_orb⁴ 2-electron integrals (chemist notation) |
| `n_elec` | `list[int]` | required | `[n_alpha, n_beta]` active electrons |
| `e_nuc` | `float` | `0.0` | Nuclear repulsion + core energy (Ha) |
| `circuit` | `list[dict]` | `[]` | Givens gates `[{"qi": int, "qj": int, "theta": float}, ...]` |
| `coup_thr` | `float` | `1e-6` | Coupling threshold for fragment detection |
| `timeout` | `float` | `300.0` | Client request timeout (s) |

---

## Limits

| Limit | Value |
|---|---|
| Max n_orb | 50 |
| Max circuit gates | 2000 |
| Rate limit | 60 req/min per API key |

---

## See also

- [DMRG Ground-State Energy](dmrg.md) — variational DMRG without a circuit, n_orb ≤ 30
- [Molecular Orbitals (HOMO/LUMO)](molecular.md) — DFT frontier orbital energies from SMILES
