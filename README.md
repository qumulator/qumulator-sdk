# Qumulator SDK

[![Test SDK](https://github.com/qumulator/qumulator-sdk/actions/workflows/test.yml/badge.svg)](https://github.com/qumulator/qumulator-sdk/actions/workflows/test.yml)
[![PyPI version](https://badge.fury.io/py/qumulator-sdk.svg)](https://pypi.org/project/qumulator-sdk/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-qumulator.github.io-7c6fff.svg)](https://qumulator.github.io/qumulator-sdk/)

> Simulate quantum circuits up to 1,000 qubits in the cloud — or **unlimited qubits locally**, GPU-accelerated, with no account required.

---

## Local statevector engine (open-source, run locally)

The full statevector simulation core ships inside `qumulator-sdk` as
`LocalStatevectorEngine` — open-source, no API key, no account, no network.
**No qubit limits**: the only ceiling is your own hardware. GPU acceleration
(CuPy / JAX / PyTorch) is auto-detected at runtime.

```bash
pip install qumulator-sdk          # CPU — unlimited qubits, pure NumPy
pip install "qumulator-sdk[gpu]"   # GPU — CuPy / JAX / PyTorch, auto-detected
```

```python
from qumulator.local import LocalStatevectorEngine

eng = LocalStatevectorEngine(n_qubits=28)   # no hard cap
eng.apply('h', 0)
for i in range(27):
    eng.apply('cx', [i, i + 1])

result = eng.run(shots=4096, return_entropy_map=True)
print(result.counts)       # GHZ measurement results
print(result.entropy_map)  # per-qubit entanglement entropy
```

For circuits beyond your local RAM, the cloud API routes to MPS / cluster backends
that scale to 1,000 qubits on a standard CPU.

---

## What is Qumulator?

Qumulator is a platform for simulating quantum circuits, spin systems, photonic amplitudes,
and molecular properties on classical hardware. It is organised around four domains:
**Quantum Simulation**, **Molecular Chemistry**, **Condensed Matter**, and **Photonic Computing**.

---

## Getting started

### Install

```bash
pip install qumulator-sdk
```

Optional extras:

```bash
pip install "qumulator-sdk[qiskit]"   # Qiskit drop-in backend
pip install "qumulator-sdk[cirq]"     # Cirq drop-in simulator
pip install "qumulator-sdk[all]"      # everything
```

### Get an API key

**Mac / Linux**
```bash
curl -s -X POST https://api.qumulator.com/keys \
     -H "Content-Type: application/json" \
     -d '{"name": "my-key"}'
```

**Windows (PowerShell)**
```powershell
Invoke-WebRequest `
  -Uri "https://api.qumulator.com/keys" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"name":"my-key"}' `
  -UseBasicParsing
```

Or via the CLI (after install):

```bash
qumulator key
```

### Quick start — 1,000-qubit circuit

```python
import os
from qumulator import QumulatorClient

client = QumulatorClient(
    api_url="https://api.qumulator.com",
    api_key=os.environ["QUMULATOR_API_KEY"],
)

# 500 parallel Bell pairs across 1,000 qubits (depth 1)
eng = client.circuit.engine(n_qubits=1000)
for i in range(0, 1000, 2):
    eng.apply("h", i)
for i in range(0, 1000, 2):
    eng.apply("cx", [i, i + 1])

result = eng.sample(shots=10)
print(result.counts)
print(result.most_probable)
```

Run the built-in demo against the live API:

```bash
qumulator demo           # 1,000-qubit Bell pairs (depth 1)
qumulator demo --willow  # 105-qubit Willow-layout RCS
```

---

## How Qumulator compares

MPS simulation scales with **entanglement depth**, not qubit count. Statevector simulators
scale with **qubit count**. These are complementary tools — Qumulator is not a drop-in
replacement for exact small-circuit simulation; it is the right tool when qubit count is
large and entanglement depth is bounded.

| Simulator | Max qubits (exact) | Requires GPU | Scales with | Best for |
|---|---|---|---|---|
| **Qumulator** | Unlimited (local) / 1,000 (cloud MPS) / 20 (cloud statevector) | Optional (local) | Entanglement depth | Large structured/variational circuits, VQE, QAOA |
| **Qumulator DMRG** | 30 active orbitals | No | Active-space size | Exact molecular ground states (FCI-quality) |
| Qumulator MPS/MPO | 50 active orbitals | No | Fragment count | Multi-fragment pharma molecules |
| Qiskit Aer | ~30 (statevector) | Optional | Qubit count | General small circuits, local use |
| BlueQubit | 34–36 | Optional | Qubit count | Exact small–mid circuits, GPU-accelerated |
| PennyLane | ~25 | Optional | Qubit count | Differentiable circuits, VQE, QAOA |

**Rule of thumb:** Qumulator handles the full spectrum — unlimited statevector locally (bounded only by your RAM/GPU), exact statevector up to 20 qubits via the cloud API, MPS up to 1,000 qubits at low entanglement depth, and cluster/Green's modes for exact results beyond that. The only narrow range where a multi-GPU statevector server has an edge is roughly 20–35 qubits at arbitrary depth; above ~35 qubits, MPS is the only practical option on any hardware, and Qumulator runs it on a standard CPU.

---

## Quantum Simulation

Gate-based quantum circuits up to 1,000 qubits — exact for structured circuits, MPS for
general depth-bounded circuits, with OpenQASM 2/3 input and Qiskit/Cirq drop-in backends.

### OpenQASM 2/3

```python
result = client.circuit.run_qasm("""
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q -> c;
""", shots=1024)

print(result.counts)       # {'00': ~512, '11': ~512}
print(result.entropy_map)  # [0.999, 0.999] — entanglement per qubit
```

### Drop into Qiskit — two lines of code

```python
from qumulator.backends.qiskit_backend import QumulatorBackend
from qiskit import transpile

backend = QumulatorBackend(client)               # replaces AerSimulator()
job     = backend.run(transpile(qc, backend), shots=1024)
counts  = job.result().get_counts()
```

Everything else in your Qiskit workflow is unchanged.

### Drop into Cirq — two lines of code

```python
from qumulator.backends.cirq_simulator import QumulatorSimulator

sim    = QumulatorSimulator(client)              # replaces cirq.Simulator()
result = sim.run(circuit, repetitions=1024)
```

### Simulation modes

Pass `mode=` to any `run()` call.

| Mode | Max qubits | Best for |
|---|---|---|
| `"statevector"` | 20 | Unconditionally exact; small N, any depth |
| `"cluster_statevector"` | 1,000 | Exact for any circuit; no 2ᴺ array; memory O(Σ 2^k_c); exact result |
| `"cluster_mps"` | 1,000 | VQE, QAOA, chemistry ansätze; MPS per cluster |
| `"mps"` | 1,000 | General circuits; low-entanglement, VQE, QAOA |
| `"hamiltonian"` | 1,000 | Hamiltonian simulation without gate decomposition |
| `"gaussian"` | unlimited | Clifford / Gaussian circuits; returns covariance certificate |
| `"greens"` | 1,000 | Free-fermion circuits; O(N²) memory, exact 1-RDM |
| `"local"` | unlimited | Local simulation — no API call, no billing |

> Modes with max 1,000 qubits are subject to the tier depth limit (max 7 entangling layers for N > 105).
> See the [circuit depth limits](#circuit-depth-limits) table. `statevector` mode: 20 qubits, any depth.

### Circuit depth limits

Bond dimension $\chi = 2^\text{depth}$; all tiers keep peak memory under 400 MB.

| Tier | Qubit range | Max entangling depth | χ | Peak memory | Notes |
|------|-------------|---------------------|---|-------------|-------|
| 1 | 1 – 20 | **20** | 1024 | 335 MB | Exact for structured circuits |
| 2 | 21 – 54 | **9** | 512 | 226 MB | Exact (2⁹ = 512) |
| 3 | 55 – 105 | **8** | 256 | 110 MB | Exact (2⁸ = 256) |
| 4 | 106 – 1,000 | **7** | 128 | 262 MB | Exact (2⁷ = 128) |

**Depth is counted in entangling layers only** — single-qubit gates (H, Rz, T, etc.) do not
count toward the depth limit and are not restricted. Requests exceeding the tier depth limit
return HTTP 422 with a self-documenting error message.

Statevector mode: max **20 qubits** at any depth.

### QS benchmarks

| Problem | Size | Result | Reference | Error | Time |
|---|---|---|---|---|---|
| CHSH Bell violation | N=2 | S = 2.828427 | 2√2 = 2.828427 | **< 0.0001%** | < 1 ms |
| RCS circuit (exact) | 12 q, depth 20 | XEB = 1.014 | exact statevector | **0.00%** | 15–23 ms |
| RCS circuit (exact) | 20 q, depth 20 | XEB = 1.024 | exact statevector | **0.00%** | 8.5–9.6 s |
| QUBO dense optimisation | N=100 | matches SA optimum | simulated annealing | 0% | ~3 s |
| Kuramoto BEC (large-scale) | N=500 oscillators, 2 MB | r=0.114 (Mott-like) | statevector: 2⁵⁰⁰ bytes | — | 3.22 s |
| Sparse GHZ state | N=500, K=2 | exact, fidelity=1.0 | dense statevector | **< 10⁻¹⁵** | 4 ms |

### QS notebooks

| Notebook | Description |
|---|---|
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qumulator/qumulator-sdk/blob/main/notebooks/willow_rcs_benchmark.ipynb) | **Willow RCS** — 105-qubit exact simulation, Willow-layout |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qumulator/qumulator-sdk/blob/main/notebooks/chsh_bell.ipynb) | **CHSH Bell inequality** — Bell state + CHSH correlator, S=2√2 Tsirelson bound |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qumulator/qumulator-sdk/blob/main/notebooks/cluster_statevector_demo.ipynb) | **Cluster engine** — exact simulation without 2^N state vector; memory O(Σ 2^k_c) |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qumulator/qumulator-sdk/blob/main/notebooks/greens_demo.ipynb) | **Green's function engine** — free-fermion circuits, O(N²) memory, entropy map |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qumulator/qumulator-sdk/blob/main/notebooks/qubo.ipynb) | **QUBO optimisation** — 100-variable dense combinatorial optimisation |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qumulator/qumulator-sdk/blob/main/notebooks/cluster_geometry.ipynb) | **Cluster geometry** — entanglement growth in random H/CX/Rz brickwork circuits |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qumulator/qumulator-sdk/blob/main/notebooks/prime_factoring.ipynb) | **Prime factorization** — factors N=35 via quantum-inspired energy landscape |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qumulator/qumulator-sdk/blob/main/notebooks/options_pricer.ipynb) | **European call option pricer** — quantum amplitude estimation vs Black-Scholes, <1% error |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qumulator/qumulator-sdk/blob/main/notebooks/fantasy_football.ipynb) | **Fantasy football lineup optimizer** — DraftKings QUBO solved with `client.hamiltonian` |

---

## Molecular Chemistry

Compute molecular frontier orbital energies from a SMILES string, or ground-state energies
from 1-electron/2-electron integrals using MPS/MPO (up to 50 orbitals) or DMRG (up to 30 orbitals).

### Frontier orbitals — `client.homo`

```python
homo = client.homo.run("Oc1ccc(/C=C/c2cc(O)cc(O)c2)cc1")
print(homo.homo_E_eV, homo.lumo_E_eV, homo.gap_eV)
```

### MPS/MPO ground-state energy — `client.molecular`

```python
from pyscf import gto, scf, mcscf, ao2mo
from qumulator import QumulatorClient

client = QumulatorClient()

mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", spin=0, charge=0)
mf  = scf.RHF(mol).run()
mc  = mcscf.CASSCF(mf, ncas=2, nelecas=2).run()

h1e, e_core = mc.get_h1eff()
h2e = ao2mo.restore(1, mc.get_h2eff(), mc.ncas)
e_nuc = mc.energy_nuc() + e_core

result = client.molecular.energy(
    h1e=h1e.tolist(),
    h2e=h2e.tolist(),
    n_elec=list(mc.nelecas),
    e_nuc=float(e_nuc),
)
print(f"E(MPS)  = {result.energy:.8f} Ha")
```

Full documentation: [Molecular Energy (MPS/MPO)](https://qumulator.github.io/qumulator-sdk/molecular-mps/)

### DMRG ground-state energy — `client.dmrg`

```python
result = client.dmrg.energy(
    h1e=h1e.tolist(),
    h2e=h2e.tolist(),
    n_elec=list(mc.nelecas),
    e_nuc=float(e_nuc),
    d_max=64,
    n_sweeps=8,
)
print(f"E(DMRG) = {result.energy:.10f} Ha")   # −1.1372838 Ha (exact FCI)
print(f"converged={result.converged}, t={result.wall_time_s:.2f} s")
```

**Choosing the right method:**

| | DMRG | MPS/MPO |
|---|---|---|
| Active space | ≤ 30 orb | ≤ 50 orb |
| Requires circuit | No | Optional |
| Accuracy control | d_max + sweeps | MPO bond dim |
| Best for | 1D-like, exact FCI | Multi-fragment, pharma |

Full documentation: [DMRG Ground-State Energy](https://qumulator.github.io/qumulator-sdk/dmrg/)

### MC benchmarks

| Problem | Size | Result | Reference | Error | Time |
|---|---|---|---|---|---|
| H₂ DMRG ground state | CAS(2,2) STO-3G, d_max=64 | −1.13728383 Ha | FCI exact | **< 10⁻¹⁰ Ha** | < 1 s |
| N₂ MPS/MPO | CAS(10,8) STO-6G | −107.6218 Ha | FCI exact | **< 1 mHa** | ~5 s |

### MC notebooks

| Notebook | Description |
|---|---|
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qumulator/qumulator-sdk/blob/main/notebooks/h2_ground_state.ipynb) | **H₂ ground state** — 4-qubit exact simulation, CASCI(2,2)/STO-3G, 100% correlation recovery |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qumulator/qumulator-sdk/blob/main/notebooks/lih_ground_state.ipynb) | **LiH ground state** — Pauli-Hamiltonian solver, 1.15 mHa error, chemical accuracy |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qumulator/qumulator-sdk/blob/main/notebooks/n2_ground_state.ipynb) | **N₂ ground state** — 12-qubit exact simulation, 79 kcal/mol correlation recovered, 100% |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qumulator/qumulator-sdk/blob/main/notebooks/molecular_mps_quickstart.ipynb) | **MPS/MPO quickstart** — molecular ground-state energy from PySCF CAS integrals; `client.molecular` |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qumulator/qumulator-sdk/blob/main/notebooks/dmrg_quickstart.ipynb) | **DMRG quickstart** — exact FCI via two-site DMRG sweeps; `client.dmrg`; H₂ → −1.13728383 Ha |

---

## Condensed Matter

Spin ground states and time evolution for 1D/2D lattice models — Ising, Heisenberg,
Kitaev chain, and custom Pauli-sum Hamiltonians.

### Spin ground states — `client.hamiltonian`

```python
import numpy as np
J = np.random.randn(8, 8); J = (J + J.T) / 2
result = client.hamiltonian.run(J.tolist())
print(result.energy)
```

### Hamiltonian time evolution (TEBD) — `client.evolve`

Real-time Suzuki–Trotter evolution, imaginary-time ground states, and
Kibble–Zurek quench protocols.

```python
# Real-time TEBD evolution — Ising TFIM
ev = client.evolve.run(
    n_qubits=20,
    hamiltonian={"preset": "ising_1d", "J": 1.0, "h": 1.0},
    t_max=2.0, dt=0.1,
    observables=["entropy", "magnetization", "qfi"],
)
for step in ev.trajectory:
    print(step["t"], step["magnetization"], step.get("qfi"))

# Imaginary-time ground state
gs = client.evolve.ground(
    n_qubits=20,
    hamiltonian={"preset": "ising_1d", "J": 1.0, "h": 0.5},
)
print(gs.energy, gs.bond_entropy)

# Kibble–Zurek quench
qkzm = client.evolve.qkzm(
    n_qubits=40, J=1.0, h0=5.0, h_f=0.2, t_ramp=10.0,
)
print(qkzm.kzm_defect_density, qkzm.kzm_prediction)
```

Hamiltonian presets: `"ising_1d"`, `"xx_model"`, `"heisenberg"`, `"kuramoto_ising"`.
Custom Pauli-sum terms are also supported via `hamiltonian={"terms": [...]}`.

#### MPS bond entanglement regime labels (`phase_labels`)

TEBD endpoints (`client.evolve.aklt()`, `client.evolve.run()`) return a `phase_labels` list — one label per MPS bond, derived from the von Neumann entropy S of each bond's Schmidt decomposition:

| Label | Entropy | Description |
|---|---|---|
| `"product_state"` | S < 0.1 bits | Unentangled — qubits on either side of the bond are independent |
| `"area_law"` | 0.1–0.6 bits | Bounded entanglement; MPS-tractable at any depth |
| `"topological_class"` | 0.6–1.2 bits | SPT / topological phase regime — the AKLT VBS inter-site bond has S = 1 bit and lands here exactly |
| `"near_volume_law"` | 1.2–2.5 bits | High entanglement, approaching volume-law |
| `"volume_law"` | S ≥ 2.5 bits | Near-maximal entanglement — Haar-random / deep circuit regime |

### CM benchmarks
| Non-Abelian anyon braiding | Fibonacci anyons (N=8) | ‖[σ₁,σ₂]‖ = 1.272 | SU(2)₃ exact | **< 0.001%** | < 1 ms |
| Kitaev chain BdG | L=1000 sites | W=−1, gap=2.000 | analytic (exact) | **< 10⁻¹²** | 0.84 s |
| MBL discrete time crystal | 8 q, 24 Floquet | autocorr = 0.827 | Google Sycamore 2021 | Consistent | ~1 s |
| Holographic wormhole | 2×6 SYK sites | fidelity 94.89% | Google Sycamore 2022 | — | ~5 s |

### CM notebooks

| Notebook | Description |
|---|---|
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qumulator/qumulator-sdk/blob/main/notebooks/wormhole.ipynb) | **Holographic wormhole** — traversable wormhole, matches Google 2022 |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qumulator/qumulator-sdk/blob/main/notebooks/anyon_braiding.ipynb) | **Anyon braiding** — Fibonacci anyons, SU(2)₃ Chern-Simons braiding, matches Microsoft topological target |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qumulator/qumulator-sdk/blob/main/notebooks/time_crystal.ipynb) | **Discrete time crystal** — MBL Floquet, matches Google Sycamore 2021 |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qumulator/qumulator-sdk/blob/main/notebooks/tebd_quench_demo.ipynb) | **Collapse & revival / QKZM** — TEBD Hamiltonian evolution, Kibble-Zurek scaling |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qumulator/qumulator-sdk/blob/main/notebooks/h12_vqe.ipynb) | **H₁₂ chain ground state** — 12-site Ising chain, `client.hamiltonian` |

---

## Photonic Computing

Compute hafnian and permanent amplitudes for photonic circuits and Gaussian Boson Sampling
(GBS) experiments — any matrix size, no qubit limit.

### Hafnian / GBS amplitudes — `client.hafnian`

```python
import numpy as np
A = np.random.randn(8, 8); A = (A + A.T) / 2
h = client.hafnian.run(A.tolist())
print(h.value)
```

Full documentation: [Photonic Amplitudes](https://qumulator.github.io/qumulator-sdk/photonics/)

### Photonics benchmarks

| Problem | Size | Result | Reference | Error | Time |
|---|---|---|---|---|---|
| Photonic hafnian (GBS) | 8×8 matrix | 0.2598−0.0078i | exact DP | **< 2×10⁻¹⁵** | 39 ms |
| Photonic hafnian (GBS) | 12×12 matrix | 0.0239+0.9947i | exact DP | **< 5×10⁻¹⁵** | 43 ms |

### Photonics notebooks

| Notebook | Description |
|---|---|
| [Boson Sampling Demo](https://colab.research.google.com/github/qumulator/qumulator-sdk/blob/main/notebooks/boson_sampling_xeb.ipynb) | GBS correctness baseline — hafnians 8×8–12×12, self-XEB, exact N=1,000 simulation |
| [Hafnian Benchmark](https://colab.research.google.com/github/qumulator/qumulator-sdk/blob/main/notebooks/hafnian_benchmark.ipynb) | 4×4 → 16×16 GBS matrices, scaling plot, verified against `thewalrus` |
| [GBS Output Distribution](https://colab.research.google.com/github/qumulator/qumulator-sdk/blob/main/notebooks/gbs_output_distribution.ipynb) | Full photon-number distribution for 4-mode GBS — normalisation and vacuum probability |
| [Permanent vs Hafnian](https://colab.research.google.com/github/qumulator/qumulator-sdk/blob/main/notebooks/permanent_vs_hafnian.ipynb) | Distinguishable vs identical photons — HOM dip: P(1,1)=0 confirmed |
| [Quantum Advantage Threshold](https://colab.research.google.com/github/qumulator/qumulator-sdk/blob/main/notebooks/quantum_advantage_threshold.ipynb) | Jiuzhang-style scaling — classical spoofability crossover, speedup certificate |

---

## Result diagnostics

Every circuit result returns a full diagnostics payload — no extra call needed.

```python
result = client.circuit.run(
    n_qubits=8,
    gates=[("h", 0), ("cx", [0, 1]), ("cx", [1, 2])],
    shots=1024,
)

# QFI-based entanglement certification (Tóth–Gühne 2012)
print(result.f_Q_density)        # f_Q > k → genuine (k+1)-partite entanglement
print(result.entanglement_depth) # floor(f_Q) — certified entanglement depth

# Accuracy bound and phase
print(result.predicted_tvd)      # TVD upper bound; 0.0 for exact modes
print(result.entropy_map)        # per-bond von Neumann entropy (bits)

# Entanglement degrees-of-freedom diagnostics (on the job object, not result)
status = client.jobs.get(job_id)
print(status.entanglement_dof)   # float — effective entanglement DOF; 0.0 = product state
print(status.dof_converged)      # bool  — True when DOF has stopped growing
```

The QFI density is the Tóth–Gühne (2012) multipartite entanglement witness computed
from the ZZ correlator matrix — an established, independently verifiable quantity:

- `f_Q_density = 0` — consistent with a product state
- `f_Q_density > k` — at least (k+1)-partite genuine entanglement certified
- `entanglement_depth = floor(f_Q_density)`

`predicted_tvd` is a model-based upper bound on the total variation distance to the
exact distribution. It is `0.0` for unconditionally exact modes (`"statevector"`, `"cluster_statevector"`).

`entanglement_dof` is the effective rank of the circuit's entanglement structure,
computed as `(Σσᵢ)² / Σσᵢ²` from the singular-value spectrum of the 1-RDM. It grows
with circuit depth and saturates when `dof_converged` becomes `True` — a signal that
additional depth adds no new entanglement correlation. Both fields are on the
`JobStatus` object returned by `client.jobs.get()`, not on the circuit `result` directly.

---

## API Pricing

**1 Compute Unit (CU) = 1 second of engine CPU time.**

A simple 2-qubit Bell circuit uses < 1 CU. A 100-qubit depth-5 MPS circuit uses ~2–5 CU.
A 1,000-qubit depth-3 circuit uses ~10–20 CU. A 20-qubit exact statevector at depth 20 uses ~9 CU.

| Plan | Price | CU / month | Notes |
|---|---|---|---|
| **Free** | $0 | 1,000 | Non-commercial & academic use. No account. No credit card. |
| **Commercial** | Contact us | Unlimited (fair-use) | Dedicated instance, SLA, custom engine parameters |

Full pricing: [qumulator.com/#pricing](https://qumulator.com/#pricing)

---

## API free tier limits

| Limit | Value |
|---|---|
| Compute Units / month | 1,000 CU (1 CU = 1 CPU-second of engine time) |
| Max qubits (statevector mode) | 20 |
| Max qubits (MPS mode) | 1,000 — see tier depth limits table |
| Rate limit | 1 request / minute |
| Daily limit | 100 requests / day |

---

## CLI

The `qumulator` command ships with the SDK:

```
qumulator demo               # 1000-qubit GHZ demo vs. the public API
qumulator demo --willow      # 105-qubit Willow-layout benchmark
qumulator demo --wormhole    # holographic wormhole
qumulator demo --anyon       # anyon braiding
qumulator key                # instructions to get a free API key
qumulator run circuit.qasm   # submit a QASM file and print the result
```

Set `QUMULATOR_API_KEY` in your environment, or pass `--key YOUR_KEY`.

---

## Documentation

Full SDK reference: [qumulator.github.io/qumulator-sdk](https://qumulator.github.io/qumulator-sdk/)

Full API reference and website docs: [qumulator.com](https://qumulator.com)

---

## License

MIT — see [LICENSE](LICENSE).
