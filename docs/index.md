# Qumulator — API & SDK Reference

!!! success "The Qumulator statevector engine is now open-source"
    `LocalStatevectorEngine` ships inside the SDK — **no qubit limits**, no API key,
    no account, no network required. The breaking changes listed in the [README](https://github.com/qumulator/qumulator-sdk#readme) affect the cloud API client only — `LocalStatevectorEngine` is unaffected.
    no account, no network required. Now you can run the statevector engine locally,
    with full source code. The only ceiling is your own hardware.
    GPU acceleration (CuPy / JAX / PyTorch) is auto-detected at runtime.

    ```bash
    pip install qumulator-sdk          # CPU — unlimited qubits
    pip install "qumulator-sdk[gpu]"   # GPU — CuPy / JAX / PyTorch
    ```

    ```python
    from qumulator.local import LocalStatevectorEngine

    eng = LocalStatevectorEngine(n_qubits=28)   # no hard cap — use what your hardware has
    eng.apply('h', 0)
    for i in range(27):
        eng.apply('cx', [i, i + 1])
    result = eng.run(shots=4096, return_entropy_map=True)
    ```

Everything you need to run quantum circuits, spin systems, photonic amplitudes, and
molecular ground-state energies (MPS/MPO and DMRG) on classical hardware.
GPU optional.

---

## Quickstart

Run your first quantum circuit in under two minutes. No account, no credit card, no
hardware required.

### Step 1 — Get an API key

One POST request returns a key immediately. The key is displayed once — save it somewhere
safe.

```bash
# cURL
curl -s -X POST https://api.qumulator.com/keys \
  -H "Content-Type: application/json" \
  -d '{"name": "my-first-key"}'
```

```json
{
  "key":        "qum_xxxxxxxx_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
  "name":       "my-first-key",
  "created_at": "2026-04-20T12:00:00Z"
}
```

### Step 2 — Install the SDK

```bash
pip install qumulator-sdk
```

### Step 3 — Run a Bell-state circuit

```python
from qumulator import QumulatorClient

client = QumulatorClient(
    api_url="https://api.qumulator.com",
    api_key="qum_xxxxxxxx...",
)

# Build and run a 2-qubit Bell state
eng = client.circuit.engine(n_qubits=2)
eng.apply('h', 0).apply('cx', [0, 1])

counts = eng.sample(shots=1024)
print(counts)   # {'00': ~512, '11': ~512}
```

!!! tip
    Store your key in the environment variable `QUMULATOR_API_KEY` and read it with
    `os.environ["QUMULATOR_API_KEY"]` to keep it out of source code.

---

## What Qumulator can simulate

| Method | Client attribute | Active space / qubit limit | Best for |
|---|---|---|---|
| Gate-based circuits | `client.circuit` | 1,000 qubits (MPS) / 20 (exact statevector) | VQE, QAOA, benchmark circuits |
| Hamiltonian time evolution | `client.evolve` | 1,000 qubits | TEBD, quenches, Kibble-Zurek |
| Spin ground states | `client.hamiltonian` | 1,000 sites | Ising / Heisenberg Hamiltonians |
| Photonic amplitudes | `client.hafnian` | Any matrix size | Hafnian, permanent, GBS |
| Molecular HOMO/LUMO | `client.homo` | Any SMILES | Frontier orbital energies from DFT |
| **Molecular energy (MPS/MPO)** | **`client.molecular`** | **≤ 50 orbitals** | **Multi-fragment pharma molecules** |
| **DMRG ground-state energy** | **`client.dmrg`** | **≤ 30 orbitals** | **Exact FCI, strongly-correlated systems** |

### Molecular simulation quickstart

```python
from pyscf import gto, scf, mcscf, ao2mo
from qumulator import QumulatorClient

client = QumulatorClient()

# Prepare H₂ CAS(2,2) integrals with PySCF
mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", spin=0)
mf  = scf.RHF(mol).run()
mc  = mcscf.CASSCF(mf, ncas=2, nelecas=2).run()

h1e, e_core = mc.get_h1eff()
h2e  = ao2mo.restore(1, mc.get_h2eff(), mc.ncas)
e_nuc = mc.energy_nuc() + e_core

# MPS/MPO — up to 50 orbitals, optional Givens circuit
result = client.molecular.energy(
    h1e=h1e.tolist(), h2e=h2e.tolist(),
    n_elec=list(mc.nelecas), e_nuc=float(e_nuc),
)
print(f"E(MPS)   = {result.energy:.8f} Ha")

# DMRG — exact FCI at d_max=64, up to 30 orbitals
result = client.dmrg.energy(
    h1e=h1e.tolist(), h2e=h2e.tolist(),
    n_elec=list(mc.nelecas), e_nuc=float(e_nuc),
    d_max=64, n_sweeps=8,
)
print(f"E(DMRG)  = {result.energy:.10f} Ha")   # −1.1372838 Ha (≤ 10⁻¹⁰ Ha from FCI)
print(f"converged = {result.converged}")
```

See [Molecular Energy (MPS/MPO)](molecular-mps.md) and [DMRG Ground-State Energy](dmrg.md) for full documentation.

---

## Quantum Simulation

Gate-based circuit simulation up to 1,000 qubits. Build circuits with Python, submit OpenQASM
2/3 programs, or drop in as a Qiskit or Cirq backend.

- [Circuits & Gates](circuit.md) — `client.circuit.engine()`, `.run()`, `.sample()`
- [OpenQASM](qasm.md) — `client.circuit.run_qasm()`
- [Simulation Modes](modes.md) — `statevector`, `mps`, `cluster_mps`, `cluster_statevector`, `greens`, `gaussian`
- [Cost Estimator](cost-estimator.md) — estimate CU cost before running
- [Qiskit Integration](qiskit.md) — `QumulatorBackend`
- [Cirq Integration](cirq.md) — `QumulatorSimulator`

---

## Molecular Chemistry

Compute frontier orbital energies from SMILES, or ground-state energies from 1e/2e integrals
via MPS/MPO or DMRG.

- [Frontier Orbitals (HOMO/LUMO)](molecular.md) — `client.homo.run(smiles)`
- [MPS/MPO Ground-State Energy](molecular-mps.md) — `client.molecular.energy(h1e, h2e, n_elec, e_nuc)` — up to 50 orbitals
- [DMRG Ground-State Energy](dmrg.md) — `client.dmrg.energy(...)` — up to 30 orbitals, exact FCI

---

## Condensed Matter

Spin ground states and time evolution for 1D/2D lattice models — Ising, Heisenberg,
Kitaev chain, and custom Pauli-sum Hamiltonians.

- [Spin Ground States](spin.md) — `client.hamiltonian.run(J)`
- [Time Evolution (TEBD)](evolve.md) — `client.evolve.run()`, `.ground()`, `.qkzm()`, `.quench()`

---

## Photonic Computing

Hafnian and permanent amplitudes for photonic circuits and Gaussian Boson Sampling.

- [Hafnian & GBS Amplitudes](photonics.md) — `client.hafnian.run(A)`

---

## Notebooks

Remote notebook execution and a library of worked examples — [Notebooks](notebooks.md).
