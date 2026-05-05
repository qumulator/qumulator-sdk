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
