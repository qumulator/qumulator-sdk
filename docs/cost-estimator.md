# Cost Estimator

Estimate the compute-unit (CU) cost of a circuit **before submitting it** — no API call,
no credits consumed.

1 CU ≈ 1 second of engine wall-clock CPU time.

---

## Fluent builder: `estimated_cost()`

Call `estimated_cost()` on a `CircuitEngine` at any point to get a cost estimate based
on the gates accumulated so far.

```python
eng = client.circuit.engine(n_qubits=20, mode='exact')
for i in range(0, 20, 2):
    eng.apply('cx', [i, i + 1])

est = eng.estimated_cost(shots=4096)
print(f"Estimated cost: {est.total_cu:.2f} CU")
# Estimated cost: 45.71 CU

print(est.breakdown)
# {
#   'base': 0.3,
#   'depth_surcharge': 45.26,
#   'shots': 0.61,
#   'mode_multiplier': 1.0,
# }
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `shots` | `int` | `1024` | Number of measurement samples to assume |

Returns a [`CostEstimate`](#costestimate) object.

---

## Standalone: `client.circuit.estimate_cost()`

Use the standalone function when you have a gate list but no builder:

```python
est = client.circuit.estimate_cost(
    gates=[('h', 0), ('cx', [0, 1])],
    n_qubits=2,
    shots=1024,
)
print(est.total_cu)   # e.g. 0.45
```

Both tuple and dict gate formats are accepted:

```python
# Tuple format
gates_tuple = [('h', 0), ('cx', [0, 1]), ('rz', 0, [1.5708])]

# Dict format
gates_dict = [
    {'gate': 'h',  'qubits': [0]},
    {'gate': 'cx', 'qubits': [0, 1]},
]

# Mixed is also fine
client.circuit.estimate_cost(gates_tuple + gates_dict, n_qubits=2)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `gates` | `list` | required | Gate list in tuple or dict format |
| `n_qubits` | `int` | required | Number of qubits |
| `shots` | `int` | `1024` | Number of measurement samples to assume |
| `mode` | `str` | `"auto"` | Execution mode (see [Simulation Modes](modes.md)) |
| `bond_dim` | `int` | `None` | Bond-dimension cap for MPS modes (default: 16) |

---

## `CostEstimate`

| Field | Type | Description |
|---|---|---|
| `total_cu` | `float` | Total estimated CU cost (rounded to 4 decimal places) |
| `breakdown` | `dict[str, float]` | Per-component cost split (see below) |

### `breakdown` keys

| Key | Description |
|---|---|
| `base` | Fixed submission overhead, scaled by the mode multiplier |
| `depth_surcharge` | Compute cost from 2-qubit gate count and qubit/bond-dim scaling, scaled by mode multiplier |
| `shots` | Per-shot sampling cost (linear in `shots`; mode-independent) |
| `mode_multiplier` | The multiplier applied to `base` + `depth_surcharge` for this mode |

---

## Cost model

The estimator uses two formulae, selected by mode and qubit count:

**Statevector** (`mode='exact'`, or any mode with N ≤ 20):

$$\text{CU} = (C_\text{base} + k_\text{sv} \cdot 2^N \cdot G) \times m + s \cdot \text{shots}$$

**MPS / tensor-network** (N > 20, `mode='auto'` / `'tensor'` / `'compressed'`):

$$\text{CU} = (C_\text{base} + k_\text{mps} \cdot N \cdot \chi^3 \cdot G) \times m + s \cdot \text{shots}$$

Where:

| Symbol | Value | Meaning |
|---|---|---|
| $C_\text{base}$ | 0.30 CU | Fixed per-submission overhead |
| $k_\text{sv}$ | 2.14 × 10⁻⁷ | Statevector kernel constant |
| $k_\text{mps}$ | 1.62 × 10⁻⁶ | MPS kernel constant |
| $G$ | — | Number of 2-qubit gates in the circuit |
| $\chi$ | 16 (default) | MPS bond dimension |
| $m$ | see table | Mode multiplier |
| $s$ | 1.5 × 10⁻⁴ | Per-shot cost (CU / shot) |

### Mode multipliers

| Mode | Multiplier |
|---|---|
| `'exact'` / `'gaussian'` | 1.0× |
| `'compressed'` / `'auto'` / `'hamiltonian'` | 1.5× |
| `'tensor'` | 2.0× |
| `'cluster'` | 3.0× |
| `'greens'` | 1.2× |
| `'local'` | 0.0× (no server billing) |

### Calibration points

The constants above are fit to these observed benchmark results:

| Circuit | Observed | Formula |
|---|---|---|
| 20-qubit depth-20 statevector (~200 CX gates) | ~45 CU | ✓ |
| 54-qubit χ=16 MPS depth-6 (~162 CX gates) | ~58 CU | ✓ |
| 105-qubit χ=16 MPS depth-5 (~262 CX gates) | ~46 CU | ✓ |
| 1000-qubit χ=16 MPS depth-3 (~1000 CX gates) | ~112 CU | ✓ |

!!! note
    The estimator is accurate to within a factor of ~2 for typical circuits.
    Circuits with unusual entanglement structure (very deep or highly connected)
    may deviate more. Use `estimated_cost()` as a planning tool, not a billing guarantee.

---

## Example: compare modes before submitting

```python
eng = client.circuit.engine(n_qubits=54, mode='auto')
for depth in range(6):
    for i in range(0, 53, 2):
        eng.apply('cx', [i, i + 1])

for mode in ('exact', 'compressed', 'tensor', 'cluster'):
    eng.mode = mode
    est = eng.estimated_cost(shots=2048)
    print(f"  {mode:12s}: {est.total_cu:6.2f} CU")

# exact       :  58.31 CU
# compressed  :  87.46 CU
# tensor      : 116.62 CU
# cluster     : 174.93 CU
```
