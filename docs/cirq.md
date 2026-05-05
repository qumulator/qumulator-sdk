# Cirq Integration

`QumulatorSimulator` implements Cirq's `Simulator` interface and supports both sampling
and statevector simulation.

```python
import cirq
from qumulator import QumulatorClient
from qumulator.backends.cirq_simulator import QumulatorSimulator

client = QumulatorClient(api_url="https://api.qumulator.com", api_key="qum_...")
sim    = QumulatorSimulator(client)

q0, q1 = cirq.LineQubit.range(2)
circuit = cirq.Circuit(
    cirq.H(q0),
    cirq.CNOT(q0, q1),
    cirq.measure(q0, q1, key='result'),
)

# Sampling
result = sim.run(circuit, repetitions=1024)
print(result.histogram(key='result'))  # Counter({0: ~512, 3: ~512})

# Statevector
sv = sim.simulate(circuit)
print(sv.final_state_vector)
# array([0.707+0j, 0+0j, 0+0j, 0.707+0j])
```

!!! info
    All Cirq gates are submitted via `cirq.unitary()`, so any gate with a unitary
    representation is supported — including custom operations and Sycamore-family gates.

---

## Google Sycamore / Willow circuits

```python
q = cirq.LineQubit.range(4)
circuit = cirq.Circuit(
    [cirq.H(qi) for qi in q],
    cirq.SYC(q[0], q[1]),
    cirq.SYC(q[2], q[3]),
    [cirq.measure(qi, key=str(i)) for i, qi in enumerate(q)],
)
result = sim.run(circuit, repetitions=2048)
```
