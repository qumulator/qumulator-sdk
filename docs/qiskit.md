# Qiskit Integration

`QumulatorBackend` is a drop-in replacement for any Qiskit simulator backend. Build your
circuit in Qiskit as normal; the adapter transpiles and runs it on Qumulator.

```python
from qumulator import QumulatorClient
from qumulator.backends.qiskit_backend import QumulatorBackend
from qiskit import QuantumCircuit

client  = QumulatorClient(api_url="https://api.qumulator.com", api_key="qum_...")
backend = QumulatorBackend(client)

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

job    = backend.run(qc, shots=2048)
result = job.result()
counts = result.get_counts()
print(counts)   # {'00': ~1024, '11': ~1024}
```

!!! info
    **Bit ordering:** Qiskit places qubit 0 as the least-significant bit. The adapter
    reverses bitstrings at output so `get_counts()` returns Qiskit-standard keys.
    No changes to your existing code are needed.

---

## Specifying a simulation mode

```python
# compressed mode for a large variational circuit
backend = QumulatorBackend(client, mode="compressed")
job     = backend.run(qc, shots=8192)
```

---

## Using with Qiskit VQE / QAOA

Any Qiskit circuit can be sent to Qumulator. The adapter automatically transpiles to the
supported gate set. Gates that expose `gate.to_matrix()` are submitted as arbitrary
unitaries and simulated exactly.

```python
# Run your parameterised ansatz on Qumulator instead of a local simulator
bound_circuit = ansatz.assign_parameters(theta_values)
bound_circuit.measure_all()

job = backend.run(bound_circuit, shots=4096)
counts = job.result().get_counts()
expectation_value = compute_energy(counts, hamiltonian)
```
