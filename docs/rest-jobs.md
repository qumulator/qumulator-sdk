# REST API — Jobs & Polling

All simulation endpoints return a `job_id` immediately. Poll until `status` is
`"completed"` or `"failed"`.

```bash
# Poll a job
curl https://api.qumulator.com/circuits/01HX8M7PVKTA \
  -H "X-API-Key: qum_..."

# While running
{ "job_id": "01HX...", "status": "running", "result": null }

# When complete
{
  "job_id":  "01HX...",
  "status":  "completed",
  "result": {
    "counts":      { "00": 512, "11": 512 },
    "n_qubits":    2,
    "shots":       1024,
    "entropy_map": [1.0, 1.0]
  }
}
```

| Status | Meaning |
|---|---|
| `queued` | Accepted and waiting for a worker |
| `running` | Simulation in progress |
| `completed` | Result available in `result` field |
| `failed` | Error message in `error` field |

## Entanglement diagnostics on job results

Circuit jobs expose two additional top-level fields alongside `result`:

| Field | Type | Description |
|---|---|---|
| `entanglement_dof` | `float\|null` | Effective entanglement degrees of freedom at job completion. Derived from the singular-value spectrum of the off-diagonal 1-RDM: `dof = (Σ σᵢ)² / Σ σᵢ²`. A product state gives `0.0`; a maximally entangled N-qubit state gives `N/2`. |
| `dof_converged` | `bool\|null` | `true` when entanglement DOF has stopped growing (relative change < 1 % between layers). Useful for detecting that additional circuit depth will not increase correlation. |

```bash
# Example: completed circuit job with entanglement diagnostics
{
  "job_id":            "01HX...",
  "status":            "completed",
  "entanglement_dof":  3.8124,
  "dof_converged":     true,
  "result": {
    "counts":      { "00": 512, "11": 512 },
    "n_qubits":    2,
    "shots":       1024,
    "entropy_map": [1.0, 1.0]
  }
}
```

Both fields are `null` for non-circuit jobs (HOMO, hafnian, molecular).
