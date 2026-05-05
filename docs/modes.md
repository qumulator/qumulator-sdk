# Simulation Modes

Pass `mode=` to `engine()` or `run()`. The default `"auto"` lets the server select the
optimal backend for your circuit.

| Mode | Best for | N limit |
|---|---|---|
| `"auto"` | General use. Server selects the backend based on circuit size and structure. | ≤ 1,000 |
| `"exact"` | Small circuits requiring full amplitude precision. Statevector and probability arrays available. | ≤ 20 |
| `"compressed"` | Structured circuits with moderate entanglement — VQE, QAOA, chemistry ansätze, shallow random circuits. | ≤ 1,000 |
| `"tensor"` | Circuits with a 1D or near-1D connectivity pattern. | ≤ 1,000 |
| `"hamiltonian"` | Direct time evolution under a Pauli-string Hamiltonian. Use with `evolve_hamiltonian()`. | ≤ 1,000 |
| `"gaussian"` | Clifford-heavy circuits. Returns a `GaussianCertificate` classifying non-Clifford content. Memory scales as O(N²). | ≤ 1,000 |
| `"cluster"` | Exact cluster-factorisation engine. No 2ᴺ state vector is ever allocated. Memory O(Σ 2^k_c) where k_c is the size of each entangled cluster. Exact for *all* circuits (TVD = 0). | ≤ 1,000 |
| `"greens"` | Green's function / Bloch encoding. Exact within the free-fermion (Gaussian) subspace. O(N²) memory. Returns per-qubit marginals and von Neumann entropy map. | ≤ 1,000 |

!!! info
    Depth limits apply depending on qubit count. Exceeding a tier limit returns HTTP 422
    with a self-documenting error. See [Simulation Limits](limits.md).

---

## Hamiltonian evolution

Use `mode="hamiltonian"` and call `evolve_hamiltonian()` to apply e^(−iHt) directly from
a Pauli-string Hamiltonian, without requiring a gate decomposition.

```python
eng = client.circuit.engine(n_qubits=4, mode="hamiltonian")

eng.apply('h', 0).apply('h', 1)  # initial state preparation

eng.evolve_hamiltonian(
    pauli_terms=[
        ( 0.5,  'ZZII'),   # ZZ coupling on sites 0-1
        ( 0.5,  'IZZI'),   # ZZ coupling on sites 1-2
        (-1.0, 'XIIX'),    # transverse field
    ],
    t=1.0,  # evolution time
)

result = eng.run(shots=2048, return_entropy_map=True)
```

Pauli strings use `I X Y Z` per qubit; the leftmost character is qubit 0.

---

## Gaussian mode and the simulation certificate

```python
eng = client.circuit.engine(n_qubits=20, mode="gaussian")

for i in range(20):
    eng.apply('h', i)
for i in range(0, 19, 2):
    eng.apply('cz', [i, i+1])
eng.apply('t', 5)   # non-Clifford gate

result = eng.run(shots=1024)
cert   = result.gaussian_certificate

print(cert.rcs_certificate)            # "LIKELY_GAUSSIAN"
print(cert.entanglement_regime)        # "area_law"
print(cert.gaussian_fidelity)          # 0.991
print(cert.wigner_negativity_estimate) # small positive float
```

| Certificate label | Meaning |
|---|---|
| `GAUSSIAN_SIMULABLE` | Purely Clifford circuit; Gaussian approximation is exact. |
| `LIKELY_GAUSSIAN` | Non-Clifford content is small; high-fidelity approximation. |
| `NON_GAUSSIAN_CORRECTION_NEEDED` | Substantial non-Clifford content; switch to `"exact"` or `"compressed"` for full accuracy. |
