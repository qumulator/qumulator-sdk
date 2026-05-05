# Errors

All error responses use the shape `{"detail": "...message..."}`.

| Status | Meaning |
|---|---|
| 400 | Malformed JSON or missing required field. |
| 401 | Missing or invalid `X-API-Key` header. |
| 422 | Validation error — circuit exceeds a tier limit, unknown gate, or unsupported mode. The response body includes a self-documenting message with the full tier table. |
| 429 | Rate limit exceeded. Check the `Retry-After` header. |
| 500 | Internal server error. Retry after a moment; [contact us](https://qumulator.com/#contact) if it persists. |

```json
// Example 422 response (circuit exceeds tier limit)
{
  "detail": "Limit Exceeded: 30 qubits at depth 15. Tier 2 (21-54 q) max depth is 9. Tiers: T1(1-20q,d≤20) T2(21-54q,d≤9) T3(55-105q,d≤8) T4(106-1000q,d≤7)"
}
```

!!! tip
    Use `circuit.validate()` or `run(dry_run=True)` to catch tier-limit errors
    client-side before any API call is made. See [Circuit Simulation](circuit.md).
