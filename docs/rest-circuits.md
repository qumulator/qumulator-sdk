# REST API — Circuits

## Submit a circuit

**`POST /circuits`** — Returns `job_id` immediately. Simulation runs asynchronously.

| Field | Type | | Description |
|---|---|---|---|
| `qasm` | `string` | | OpenQASM 2 or 3 source. Provide *either* `qasm` or `instructions`. |
| `instructions` | `array` | | JSON gate instruction list. Requires `n_qubits`. |
| `n_qubits` | `int` | | Register width. Required when using `instructions`. |
| `shots` | `int` | | Measurement samples. Default: 1024. Max: 1,000,000. |
| `mode` | `string` | | Simulation mode. Default: `statevector`. |
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
    "mode": "statevector"
  }'

# Response 202
{ "job_id": "01HX8M7PVKTA..." }
```

---

## List recent circuits

**`GET /circuits`** — Returns the most recent circuit jobs for your API key.

---

## Get a circuit job

**`GET /circuits/{job_id}`** — Returns the job status and result when complete.
