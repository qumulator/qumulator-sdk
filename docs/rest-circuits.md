# REST API — Circuits

## Submit a circuit

**`POST /circuits`** — Returns `job_id` immediately. Simulation runs asynchronously.

| Field | Type | | Description |
|---|---|---|---|
| `qasm` | `string` | | OpenQASM 2 or 3 source. Provide *either* `qasm` or `instructions`. |
| `instructions` | `array` | | JSON gate instruction list. Requires `n_qubits`. |
| `n_qubits` | `int` | | Register width. Required when using `instructions`. |
| `shots` | `int` | | Measurement samples. Default: 1024. Max: 1,000,000. |
| `mode` | `string` | | Simulation mode. One of: `"auto"`, `"statevector"`, `"mps"`, `"cluster_mps"`, `"cluster_statevector"`, `"cluster_exact_graph"`, `"gaussian"`, `"greens"`, `"hamiltonian"`, `"dyson"`. Default: `"statevector"`. Pass `"auto"` to let the engine select the optimal mode based on circuit analysis. |
| `seed` | `int` | | RNG seed for reproducible sampling. |
| `return_statevector` | `bool` | | Include complex amplitude vector in result. |
| `return_entropy_map` | `bool` | | Include per-qubit entropy values in result. |
| `bond_dim` | `int` | | Bond dimension cap for tensor-network modes. |

```bash
# Submit via cURL
curl -X POST https://api.qumulator.com/circuits \
  -H "X-API-Key: qum_..." \
  -H "Content-Type: application/json" \
  -d '{
    "qasm": "OPENQASM 3.0; include \"stdgates.inc\"; qubit[2] q; h q[0]; cx q[0],q[1]; bit[2] c; measure q->c;",
    "shots": 1024,
    "mode": "auto"
  }'

# Response 202
{ "job_id": "01HX8M7PVKTA..." }
```

### Result fields (when `mode="auto"`)

When the job completes, the result body includes two additional fields:

| Field | Type | Description |
|---|---|---|
| `resolved_mode` | `string` | The simulation mode actually used (e.g. `"mps"`, `"cluster_mps"`). |
| `preflight_report` | `object` | Routing diagnostics: `d_ky`, `entanglement_regime`, `reasoning`, `is_tree`, `edge_density`, `d_s`, `n_2q_gates`, `n_t_gates`, `ky_gp_consistent`. |

---

## Preflight analysis (zero cost)

**`POST /circuits/preflight`** — Analyse a circuit and get a mode recommendation without
running a simulation. No job is created and no compute units are deducted.

| Field | Type | | Description |
|---|---|---|---|
| `qasm` | `string` | | OpenQASM 2 or 3 source. Provide *either* `qasm` or `instructions`. |
| `instructions` | `array` | | JSON gate instruction list. Requires `n_qubits`. |
| `n_qubits` | `int` | | Register width. Required when using `instructions`. |

**Response:**

| Field | Type | Description |
|---|---|---|
| `n_qubits` | `int` | Qubit count parsed from the circuit. |
| `recommended_mode` | `string` | Optimal simulation mode for this circuit. |
| `reasoning` | `string` | One-line explanation of the routing decision. |
| `d_ky` | `float` | Kaplan-Yorke dimension estimate (null for very small circuits). |
| `entanglement_regime` | `string` | `"area_law"`, `"transitional"`, or `"volume_law"`. |
| `d_s` | `float` | Spectral dimension of the entanglement graph (secondary check). |
| `is_tree` | `bool` | True if the entanglement graph is a tree (treewidth = 1). |
| `edge_density` | `float` | Fraction of possible qubit pairs connected by 2Q gates. |
| `n_2q_gates` | `int` | Number of 2-qubit gates. |
| `n_t_gates` | `int` | Number of non-Clifford gates. |
| `ky_gp_consistent` | `bool` | Whether D_KY ≥ d_S/2 (null if either measure unavailable). |

```bash
curl -X POST https://api.qumulator.com/circuits/preflight \
  -H "X-API-Key: qum_..." \
  -H "Content-Type: application/json" \
  -d '{"qasm": "OPENQASM 3.0; include \"stdgates.inc\"; qubit[2] q; h q[0]; cx q[0],q[1];"}'

# Response 200
{
  "n_qubits": 2,
  "recommended_mode": "statevector",
  "reasoning": "No 2Q gates in non-Clifford context; statevector trivial",
  "d_ky": null,
  "is_tree": true,
  "edge_density": 0.5,
  "n_2q_gates": 1,
  "n_t_gates": 0
}
```

---

## List recent circuits

**`GET /circuits`** — Returns the most recent circuit jobs for your API key.

---

## Get a circuit job

**`GET /circuits/{job_id}`** — Returns the job status and result when complete.
