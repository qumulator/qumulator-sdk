# OpenQASM Input

Both OpenQASM 2 and OpenQASM 3 source strings are accepted. Pass the source to
`client.circuit.run_qasm()`.

```python
qasm_source = """
OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
bit[3]   c;

h  q[0];
cx q[0], q[1];
cx q[0], q[2];
measure q -> c;
"""

result = client.circuit.run_qasm(
    qasm=qasm_source,
    shots=2048,
    mode="statevector",
    return_entropy_map=True,
)
print(result.counts)    # {'000': ~1024, '111': ~1024}
```

!!! info
    QASM 2 circuits exported from Qiskit are fully supported. The server handles both
    formats transparently. Standard gate libraries including `stdgates.inc`, `qelib1.inc`,
    and custom gate definitions are all supported.

---

## Custom unitary gates

For gates with no standard QASM name (e.g. the SYC gate in Google Willow circuits), use
the `instructions` input with an explicit matrix:

```python
import numpy as np

# Arbitrary 2-qubit unitary (e.g. Google SYC gate)
my_gate = np.array([
    [1, 0,    0,   0                  ],
    [0, 0,   -1j,  0                  ],
    [0, -1j,  0,   0                  ],
    [0, 0,    0,   np.exp(1j * np.pi / 6)],
])

eng = client.circuit.engine(n_qubits=4)
eng.apply('unitary', [0, 1], params=my_gate.tolist())
   .apply('unitary', [2, 3], params=my_gate.tolist())

result = eng.run(shots=1024)
```

You can also submit circuits via the CLI:

```bash
qumulator run circuit.qasm
```

The CLI performs the same pre-flight depth check before submission and prints a clear
warning if the circuit exceeds a tier limit.
