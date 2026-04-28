# Qumulator SDK

[![Test SDK](https://github.com/qumulator/qumulator-sdk/actions/workflows/test.yml/badge.svg)](https://github.com/qumulator/qumulator-sdk/actions/workflows/test.yml)
[![PyPI version](https://badge.fury.io/py/qumulator-sdk.svg)](https://pypi.org/project/qumulator-sdk/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> Simulate 1,000-qubit quantum circuits on classical hardware. Exact results. No GPU. No quantum hardware required.

---

## What is this?

Qumulator is a cloud API — and this is its Python client — for simulating quantum circuits,
spin systems, photonic amplitudes, and molecular properties on standard classical hardware.
It does not require a quantum computer, a GPU, or any special hardware. It runs in the cloud
(Google Cloud Run, 4 vCPU, 16 GB RAM) and returns results over HTTP.

The key numbers: a **1,000-qubit circuit** at depth 3 runs in under **1 second** using **1 MB of memory**
— where the equivalent statevector would require $2^{1000}$ bytes (more atoms than exist in the
observable universe). A 105-qubit Willow-layout circuit at depth 5 completes in under 0.5 s.
Results are exact within the stated truncation error, not statistical estimates.

The simulation engine is built on the **KLT Engine**, a proprietary classical simulation
framework that routes each problem to the most efficient representation — tensor network,
cluster-exact, Gaussian covariance matrix, nexus graph, or full statevector — based on the
entanglement structure of the specific circuit. Callers select a mode with a single parameter;
the engine handles routing automatically.

---

## Benchmarks

Measured on a standard cloud CPU (4 vCPU, no GPU). "Exact" means output agrees with full
statevector simulation to double-precision floating point (< 10⁻¹⁴ L² error on the amplitude vector).

| Problem | Size | Result | Reference | Error | Time |
|---|---|---|---|---|---|
| CHSH Bell violation | N=2 | S = 2.828427 | 2√2 = 2.828427 | **< 0.0001%** | < 1 ms |
| H₁₂ Heisenberg chain | 12 sites | −11.000 | −11.000 (exact diag.) | **0.00%** | ~0.27 s |
| Photonic hafnian (GBS) | 8×8 matrix | 0.2598−0.0078i | exact DP | **< 2×10⁻¹⁵** | 39 ms |
| Photonic hafnian (GBS) | 12×12 matrix | 0.0239+0.9947i | exact DP | **< 5×10⁻¹⁵** | 43 ms |
| RCS circuit (exact) | 12 q, depth 20 | XEB = 1.014 | exact statevector | **0.00%** | 15–23 ms |
| RCS circuit (exact) | 20 q, depth 20 | XEB = 1.024 | exact statevector | **0.00%** | 8.5–9.6 s |
| MBL discrete time crystal | 8 q, 24 Floquet | autocorr = 0.827 | Google Sycamore 2021 | Consistent | ~1 s |
| Holographic wormhole | 2×6 SYK sites | fidelity 94.89% | Google Sycamore 2022 | — | ~5 s |
| Non-Abelian anyon braiding | Fibonacci anyons | ‖[σ₁,σ₂]‖ = 1.272 | SU(2)₃ exact | **< 0.001%** | < 1 ms |
| Kitaev chain BdG | L=1000 sites | W=−1, gap=2.000 | analytic (exact) | **< 10⁻¹²** | 0.84 s |
| QUBO dense optimisation | N=100 | matches SA optimum | simulated annealing | 0% | ~3 s |
| Kuramoto BEC (large-scale) | N=500 oscillators, 2 MB | r=0.114 (Mott-like) | statevector: 2⁵⁰⁰ bytes (impossible) | — | 3.22 s |

### Circuit depth limits (approximation modes)

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

---

## Install

```bash
pip install qumulator-sdk
```

Optional extras:

```bash
pip install "qumulator-sdk[qiskit]"   # Qiskit drop-in backend
pip install "qumulator-sdk[cirq]"     # Cirq drop-in simulator
pip install "qumulator-sdk[all]"      # everything
```

---

## Get a free API key

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

---

## Quick start — 1,000-qubit circuit in 30 seconds

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
    eng.apply("h", i)           # Hadamard on even qubits (parallel)
for i in range(0, 1000, 2):
    eng.apply("cx", [i, i + 1]) # entangle each pair (parallel)

result = eng.sample(shots=10)
print(result.counts)       # e.g. {'0101100110...': 1, '1010011001...': 1, ...}
print(result.most_probable)
# Exact result. No quantum hardware. No GPU. Standard cloud CPU.
```

Run the built-in demo against the live API:

```bash
qumulator demo           # 1000-qubit GHZ
qumulator demo --willow  # 105-qubit Willow-layout RCS
```

---

## OpenQASM 2/3

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

---

## Drop into Qiskit — two lines of code

```python
from qumulator.backends.qiskit_backend import QumulatorBackend
from qiskit import transpile

backend = QumulatorBackend(client)               # replaces AerSimulator()
job     = backend.run(transpile(qc, backend), shots=1024)
counts  = job.result().get_counts()
```

Everything else in your Qiskit workflow is unchanged.

---

## Drop into Cirq — two lines of code

```python
from qumulator.backends.cirq_simulator import QumulatorSimulator

sim    = QumulatorSimulator(client)              # replaces cirq.Simulator()
result = sim.run(circuit, repetitions=1024)
```

---

## Simulation modes

Pass `mode=` to any `run()` call. The server selects `auto` by default.

| User-facing mode | Internal mode | Max qubits | Best for |
|---|---|---|---|
| `auto` | (server-routed) | 1,000 | General circuits; server auto-routes |
| `exact` | (statevector) | 20 | Unconditionally correct; small N |
| `compressed` | (tensor network) | 1,000 | VQE, QAOA, chemistry ansätze |
| `tensor` | (MPS) | 1,000 | 1D-structured, low-entanglement circuits |
| `hamiltonian` | (operator algebra) | 1,000 | Hamiltonian simulation without gate decomposition |
| `gaussian` | (covariance matrix) | unlimited | Clifford circuits; returns Wigner negativity certificate |

---

## Other computation types

```python
# Molecular HOMO/LUMO frontier orbital energies (SMILES input)
homo = client.homo.run("Oc1ccc(/C=C/c2cc(O)cc(O)c2)cc1")
print(homo.homo_E_eV, homo.lumo_E_eV, homo.gap_eV)

# Ground-state energy of a spin Hamiltonian (Ising / Heisenberg / general)
import numpy as np
J = np.random.randn(8, 8); J = (J + J.T) / 2
result = client.klt.run(J.tolist())
print(result.energy)

# Hafnian / GBS photonic amplitude
A = np.random.randn(8, 8); A = (A + A.T) / 2
h = client.hafnian.run(A.tolist())
print(h.value)
```

---

## Free tier limits

| Limit | Value |
|---|---|
| Compute Units / month | 500 CU (1 CU = 1 CPU-second of engine time) |
| Max qubits (statevector mode) | 20 |
| Max qubits (MPS mode) | 1,000 (all tiers) |
| Rate limit | 1 request / minute |
| Daily limit | 100 requests / day |
| Free tier availability | Beta only — may be discontinued at any time |

Paid plans start at **$99/month** (10,000 CU). See [qumulator.com/#pricing](https://qumulator.com/#pricing).

---

## Demo notebooks

Click to open in Google Colab — no install required, just add your API key:

| Notebook | Description |
|---|---|
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qumulator/qumulator-sdk/blob/main/notebooks/willow_rcs_benchmark.ipynb) | **Willow RCS** — 105-qubit exact simulation, Willow-layout |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qumulator/qumulator-sdk/blob/main/notebooks/wormhole.ipynb) | **Holographic wormhole** — traversable wormhole, matches Google 2022 |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qumulator/qumulator-sdk/blob/main/notebooks/anyon_braiding.ipynb) | **Anyon braiding** — Fibonacci anyons, matches Microsoft topological target |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qumulator/qumulator-sdk/blob/main/notebooks/time_crystal.ipynb) | **Discrete time crystal** — MBL Floquet, matches Google Sycamore 2021 |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qumulator/qumulator-sdk/blob/main/notebooks/qubo.ipynb) | **QUBO optimisation** — 100-variable dense combinatorial optimisation |

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

Full API reference and examples: [qumulator.com](https://qumulator.com)

---

## License

MIT — see [LICENSE](LICENSE).
