# Spin Ground States

Compute ground-state energies for interacting spin systems or arbitrary Pauli-string
Hamiltonians. Returns the minimum energy configuration plus a per-site entanglement
entropy map.

---

## Pauli Hamiltonian (recommended)

Pass a dictionary of Pauli-string → coefficient entries. Each key is an N-character
string over `{I, X, Y, Z}`, one character per site.

```python
# H2 molecule — STO-3G, Jordan-Wigner mapping (2 sites)
result = client.klt.run(
    pauli_hamiltonian={
        "II": -1.8572750,   # constant / nuclear repulsion
        "ZI": -0.3979374,
        "IZ": -0.3979374,
        "ZZ":  0.3980202,
        "XX":  0.1809270,
        "YY":  0.1809270,
    },
)
print(result.energy)       # ground-state energy in Hartree
print(result.entropy_list) # per-site entanglement entropy
```

---

## Ising / Heisenberg J-matrix

```python
import numpy as np

N = 16
J = np.random.randn(N, N)
J = (J + J.T) / 2  # symmetrize
np.fill_diagonal(J, 0)

result = client.klt.run(interaction_matrix=J.tolist())
print(result.energy)   # ground-state energy
print(result.states)   # per-site spin expectation values
print(result.max_S)    # max bipartite entanglement entropy
```

---

## KLTResult fields

| Field | Type | Description |
|---|---|---|
| `energy` | `float` | Ground-state energy |
| `states` | `list[float]` | Per-site expectation values |
| `entropy_list` | `list[float]` | Per-site entanglement entropy |
| `max_S` | `float` | Maximum bipartite entanglement entropy |
| `mean_S` | `float` | Mean entanglement entropy across all bipartitions |
