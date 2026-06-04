# Installation

## Local statevector engine (open-source, no account required)

The Qumulator statevector engine ships inside the SDK — now available with full source
code. Run it locally with no API key, no account, and no network connection required.
**No qubit limits** — simulate as many qubits as your hardware supports.

```bash
pip install qumulator-sdk          # CPU — unlimited qubits, pure NumPy
pip install "qumulator-sdk[gpu]"   # + GPU acceleration (CuPy / JAX / PyTorch)
```

GPU backends are auto-detected at import time in this order: CuPy (NVIDIA CUDA),
JAX (Google XLA), PyTorch. If none are available the engine silently falls back to NumPy.

```python
from qumulator.local import LocalStatevectorEngine

eng = LocalStatevectorEngine(n_qubits=28)          # CPU
eng = LocalStatevectorEngine(n_qubits=28, device='gpu')  # GPU
```

## Cloud API client

```bash
pip install qumulator-sdk
```

Requires Python 3.10 or later. The core package has a single dependency: `httpx`.
Framework adapters (Qiskit, Cirq) require those packages installed separately.

```bash
# With Qiskit adapter
pip install qumulator-sdk qiskit

# With Cirq adapter
pip install qumulator-sdk cirq

# Everything
pip install "qumulator-sdk[all]"
```

---

## QumulatorClient

`QumulatorClient` is the entry point for all computation types.

```python
from qumulator import QumulatorClient

client = QumulatorClient(
    api_url="https://api.qumulator.com",
    api_key="qum_...",
)
```

| Attribute | Type | Purpose |
|---|---|---|
| `client.circuit` | `CircuitClient` | Quantum circuit simulation (gate builder, QASM, Hamiltonian evolution) |
| `client.hamiltonian` | `HamiltonianClient` | Spin ground states and Pauli Hamiltonian energy |
| `client.hafnian` | `HafnianClient` | Photonic amplitudes (hafnian, permanent, GBS) |
| `client.homo` | `HomoClient` | Molecular HOMO/LUMO frontier orbital energies |
| `client.notebook` | `NotebookClient` | Remote Jupyter notebook execution |
| `client.evolve` | `EvolveClient` | Hamiltonian time evolution (TEBD) |
