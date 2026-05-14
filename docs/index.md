# Qumulator — API & SDK Reference

!!! success "The Qumulator statevector engine is now open-source"
    `LocalStatevectorEngine` ships inside the SDK — **no qubit limits**, no API key,
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
molecular orbitals on classical hardware. GPU optional. No quantum computer required.

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
