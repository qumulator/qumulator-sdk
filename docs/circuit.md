# Circuit Simulation

There are two ways to construct a circuit: the **fluent gate builder** and the direct
**gate-list API**. Both are equivalent and produce identical results. No simulation runs
locally — all execution is server-side.

---

## Fluent gate builder

Obtain a `CircuitEngine` from `client.circuit.engine()`, accumulate gates with `.apply()`,
then submit with `.run()` or `.sample()`.

```python
eng = client.circuit.engine(n_qubits=3)

eng.apply('h',  0             )  # Hadamard on qubit 0
   .apply('cx', [0, 1]        )  # CNOT: control 0, target 1
   .apply('cx', [0, 2]        )  # CNOT: control 0, target 2
   .apply('rz', 2, params=[1.5708])  # Rz(pi/2) on qubit 2

result = eng.run(
    shots=4096,
    return_statevector=True,
    return_entropy_map=True,
)

print(result.counts)        # {'000': ~2048, '111': ~2048}
print(result.entropy_map)   # [1.0, 0.0, 0.0]  — qubit 0 maximally entangled
print(result.statevector)   # complex ndarray, length 8
```

---

## Gate-list API

```python
result = client.circuit.run(
    gates=[
        ('h',  0),
        ('cx', [0, 1]),
    ],
    n_qubits=2,
    shots=2048,
)
```

---

## Supported gates

| Gate(s) | Qubits | Params | Description |
|---|---|---|---|
| `h`, `x`, `y`, `z` | 1 | — | Pauli and Hadamard |
| `s`, `t`, `sdg`, `tdg` | 1 | — | Phase / T gates and their adjoints |
| `rx`, `ry`, `rz` | 1 | `[θ]` | Rotation gates; angle in radians |
| `u` | 1 | `[θ, φ, λ]` | General single-qubit unitary (IBM U gate) |
| `cx`, `cnot` | 2 | — | CNOT (control, target) |
| `cz` | 2 | — | Controlled-Z |
| `swap` | 2 | — | SWAP |
| `iswap` | 2 | — | iSWAP |
| `fsim` | 2 | `[θ, φ]` | fSim gate (Google Sycamore / Willow family) |
| `syc` | 2 | — | Sycamore gate |
| `ecr` | 2 | — | Echoed cross-resonance (IBM native gate) |
| `ccx`, `toffoli` | 3 | — | Toffoli (CCNOT) |
| `cswap`, `fredkin` | 3 | — | Fredkin |
| `unitary` | 1–N | matrix | Arbitrary unitary; pass a 2ᴺ×2ᴺ complex matrix via `params` |

!!! info
    Qubits are 0-indexed. For two-qubit gates, pass a list: `eng.apply('cx', [0, 1])`.
    Bitstrings in results are **MSB-first**: qubit 0 is the leftmost character.

---

## `engine()` parameters

| Parameter | Type | | Description |
|---|---|---|---|
| `n_qubits` | `int` | required | Number of qubits in the register |
| `mode` | `str` | | Simulation mode. Default: `"auto"`. See [Simulation Modes](modes.md). |
| `bond_dim` | `int` | | Bond dimension cap for `"compressed"` and `"tensor"` modes |

---

## `run()` parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `shots` | `int` | 1024 | Measurement samples (1 to 1,000,000) |
| `seed` | `int` | None | RNG seed for reproducible sampling |
| `return_statevector` | `bool` | False | Include complex amplitude vector (requires N ≤ 20, `"exact"` mode only) |
| `return_probabilities` | `bool` | False | Include probability vector (requires N ≤ 20, `"exact"` mode only) |
| `return_entropy_map` | `bool` | False | Include per-qubit entanglement entropy values |
| `dry_run` | `bool` | False | Validate the circuit client-side without submitting to the API |

---

## Pre-flight validation

`CircuitEngine.validate()` checks the circuit's qubit count and entangling-layer depth
against the published tier limits before submitting — no API call, no compute cost.

```python
eng = client.circuit.engine(n_qubits=200)
for i in range(200):
    eng.apply('h', i)
for i in range(0, 199, 2):
    eng.apply('cx', [i, i + 1])
# ... 10 entangling layers total

eng.validate()  # raises ValueError: depth 10 exceeds Tier 4 limit of 7
```

Use `dry_run=True` to validate silently without submitting:

```python
result = eng.run(shots=1024, dry_run=True)  # returns None if valid; raises if not
```
