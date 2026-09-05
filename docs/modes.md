# Simulation Modes

Pass `mode=` to `engine()` or `run()`. The default `"mps"` uses the MPS backend and
scales to 1,000 qubits.

| Mode | Best for | N limit |
| --- | --- | --- |
| `"auto"` | Let the engine choose. Analyses the circuit's entanglement graph and selects the optimal mode automatically. Resolved mode and routing diagnostics returned in `result.resolved_mode` and `result.preflight_report`. | ≤ 1,000 |
| `"statevector"` | Small circuits requiring full amplitude precision. Statevector and probability arrays available. | ≤ 20 |
| `"mps"` | General circuits; low-entanglement depth, VQE, QAOA, 1D/near-1D connectivity. | ≤ 1,000 |
| `"geometric_mps"` | **Exact, non-truncated MPS.** Bond dimension derived from the circuit's qubit-coupling graph (chain/ring/grid/general — not a spatial embedding) before running — no SVD truncation, no guessed χ. Automatically selected by `mode="auto"` for particle-number-conserving circuits (Givens/iSWAP-family ansätze); actual bond dimension stays C(n,k)-bounded for the true excitation count regardless of apparent graph density. | ≤ 1,000 |
| `"cluster_mps"` | Cluster-factorised MPS. VQE, QAOA, chemistry ansätze, shallow random circuits. | ≤ 1,000 |
| `"cluster_statevector"` | Exact cluster-factorisation engine. No 2ᴺ state vector is ever allocated. Memory O(Σ 2^k_c). Exact for *all* circuits (TVD = 0). | ≤ 1,000 |
| `"cluster_exact_graph"` | Near-volume-law circuits with graph entanglement topology. | ≤ 50 |
| `"hamiltonian"` | Direct time evolution under a Pauli-string Hamiltonian. Use with `evolve_hamiltonian()`. | ≤ 1,000 |
| `"gaussian"` | Clifford-heavy circuits. Returns a `GaussianCertificate` classifying non-Clifford content. Memory scales as O(N²). | ≤ 1,000 |
| `"greens"` | Green's function / Bloch encoding. Exact within the free-fermion (Gaussian) subspace. O(N²) memory. Returns per-qubit marginals and von Neumann entropy map. | ≤ 1,000 |
| `"cluster_gaussian"` | **Cluster-factorised exact probability engine.** Never builds 2^N; O(Σ 2^k_c) memory per cluster. | ≤ 1,000 |
| `"fibonacci_anyon"` | **Topological quantum computing.** SU(2)₃ Chern-Simons Fibonacci anyons. Hilbert space dimension F_{N+2}. Native braid gates: `fibonacci_f`, `fibonacci_r`, `fibonacci_b`, `fibonacci_bdg`. Returns topological XEB score. | ≤ 1,000 |
| `"kuramoto"` | **Bose-Hubbard BEC / Kuramoto synchronisation.** O(N²) phase-oscillator engine. Superfluid order parameter r = \|⟨e^{iθ}⟩\|. Exact for genuine bosonic/BEC problems; an approximate heuristic when applied to general qubit circuits. Ideal for large-N BEC and synchronisation studies. Use `kuramoto_diagnostics()` for full phase-space output. | ≤ 10,000 |
| `"sparse"` | **Adaptive sparse exact simulation.** O(K log K) per gate, K = number of active basis states. Ideal for particle-conserving circuits, GHZ/cluster states, structured chemistry. Up to 78× faster than dense statevector at N=20 for sparse circuits. | Unlimited (K ≪ 2^N) |

!!! info
    Depth limits apply depending on qubit count. Exceeding a tier limit returns HTTP 422
    with a self-documenting error. See [Simulation Limits](limits.md).

!!! info
    Two names above are re-used from other subfields and mean something specific here:
    `"cluster_*"` modes refer to connected components of the circuit's entanglement graph,
    not the measurement-based-quantum-computing "cluster state" resource; `"gaussian"`
    adapts the continuous-variable Gaussian-state/covariance-matrix formalism to qubit
    Clifford circuits, it is not a continuous-variable photonic mode.

---

## Auto-mode routing

When `mode="auto"`, the engine analyses the circuit before simulation using the
**Kaplan-Yorke dimension** (D_KY), which characterises the fractal dimension of the
circuit's entanglement structure. The routing thresholds are:

| Condition | Resolved mode | Regime |
| --- | --- | --- |
| Tree entanglement graph | `"mps"` | Treewidth = 1; MPS is exact |
| No non-Clifford (T) gates | `"gaussian"` | Clifford-only; O(N²), exact |
| Particle-number-conserving (Givens/iSWAP-family, diagonal-only entangling gates) | `"geometric_mps"` | Exact regardless of apparent graph density; cost is C(n,k)-bounded for the true excitation count |
| D_KY < 2.1 | `"cluster_mps"` | Area-law (low entanglement) |
| 2.1 ≤ D_KY < 2.5 | `"cluster_mps"` | Intermediate entanglement |
| 2.5 ≤ D_KY < 2.9 | `"cluster_exact_graph"` | 3D-like, near volume-law |
| D_KY ≥ 2.9, N ≤ 20 | `"statevector"` | Volume-law (dense), exact |
| D_KY ≥ 2.9, N > 20 | `"statevector"` ⚠ | Classically hard; simulation may be slow |

The resolved mode is returned in `result.resolved_mode`. Routing diagnostics
(`d_ky`, `entanglement_regime`, `reasoning`, `is_tree`, `edge_density`, etc.) are
returned in `result.preflight_report`.

```python
eng = client.circuit.engine(n_qubits=20, mode="auto")
for i in range(19):
    eng.apply('cx', [i, i + 1])
    eng.apply('rz', i, params=[0.3])

result = eng.run(shots=2048)
print(result.resolved_mode)                      # e.g. 'mps'
print(result.preflight_report["d_ky"])           # e.g. 2.07
print(result.preflight_report["reasoning"])      # 'Entanglement graph is a tree...'
print(result.preflight_report["is_tree"])        # True
```

### Dry-run preflight (zero CU)

Use `client.circuit.preflight()` to get the routing recommendation without running
the simulation — no compute units are consumed:

```python
report = client.circuit.preflight(qasm_source)
print(report["recommended_mode"])  # e.g. 'cluster_mps'
print(report["reasoning"])         # one-line explanation
print(report["d_ky"])              # Kaplan-Yorke dimension
print(report["is_tree"])           # bool
```

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
| --- | --- |
| `GAUSSIAN_SIMULABLE` | Purely Clifford circuit; Gaussian approximation is exact. |
| `LIKELY_GAUSSIAN` | Non-Clifford content is small; high-fidelity approximation. |
| `NON_GAUSSIAN_CORRECTION_NEEDED` | Substantial non-Clifford content; switch to `"statevector"` or `"cluster_mps"` for full accuracy. |
