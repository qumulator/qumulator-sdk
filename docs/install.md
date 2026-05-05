# Installation

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
| `client.klt` | `KLTClient` | Spin ground states and Pauli Hamiltonian energy |
| `client.hafnian` | `HafnianClient` | Photonic amplitudes (hafnian, permanent, GBS) |
| `client.homo` | `HomoClient` | Molecular HOMO/LUMO frontier orbital energies |
| `client.notebook` | `NotebookClient` | Remote Jupyter notebook execution |
| `client.evolve` | `EvolveClient` | Hamiltonian time evolution (TEBD) |
