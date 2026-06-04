# Hamiltonian Time Evolution (TEBD)

The `client.evolve` sub-client provides Hamiltonian time evolution via TEBD
(Time-Evolving Block Decimation). Five endpoints are available:

| Method | Description |
|---|---|
| `client.evolve.run()` | Real-time TEBD evolution (Suzuki-Trotter 1st/2nd order). Returns a `trajectory` list of observables (entropy, magnetization, QFI) at each timestep. |
| `client.evolve.quench()` | Sudden quench / collapse-and-revival protocol. Returns trajectory + optional `C_tR` ZZ-correlator heatmap. |
| `client.evolve.ground()` | Imaginary-time evolution to the ground state. Returns `energy`, `bond_entropy`, and `converged`. |
| `client.evolve.qkzm()` | Kibble-Zurek quench protocol. Returns `kzm_defect_density`, `kzm_prediction`, and `tau_Q`. |
| `client.evolve.lattice()` | 2D lattice regime classifier. Returns `bond_entropy_2d` heatmap and `phase_label`. |

---

## Real-time evolution

```python
# 10-site transverse-field Ising model, critical point J=h=1
result = client.evolve.run(
    n_qubits=10,
    hamiltonian={"preset": "ising_1d", "J": 1.0, "h": 1.0},
    t_max=2.0,
    dt=0.1,
    bond_dim=64,
    observables=["entropy", "magnetization", "qfi"],
    initial_state="ferromagnet",
    order=2,
)

for pt in result.trajectory:
    print(pt["t"], pt.get("max_entropy"), pt.get("f_Q_density"))
```

---

## Ground state preparation (imaginary time)

```python
gs = client.evolve.ground(
    n_qubits=10,
    hamiltonian={"preset": "ising_1d", "J": 1.0, "h": 1.0},
    bond_dim=64,
)
print(gs.energy)        # ground-state energy
print(gs.bond_entropy)  # per-bond von Neumann entropy
print(gs.converged)     # True if imaginary-time evolution converged
```

---

## QKZM Kibble-Zurek quench

```python
qkzm = client.evolve.qkzm(
    n_qubits=20, J=1.0, h0=5.0, h_f=0.2, t_ramp=5.0,
)
print(qkzm.kzm_defect_density)   # n_d ∝ τ_Q^{−1/2}
print(qkzm.kzm_prediction)
```

---

## Collapse-and-revival quench

```python
revival = client.evolve.quench(n_qubits=20, h=2.0, t_max=10.0)
```

---

## Hamiltonian presets

| Preset | Model | Parameters |
|---|---|---|
| `"ising_1d"` | Transverse-field Ising chain | `J`, `h` |
| `"xx_model"` | XX free-fermion chain | `t` |
| `"heisenberg"` | XXX Heisenberg chain | `J` |
| `"kuramoto_ising"` | Hamiltonian (DM coupling) | `J`, `K` |

Custom Pauli-sum Hamiltonians are also supported via `terms`: each term is
`{"sites": [i, j], "operator": "ZZ", "strength": -1.0}`.
