# Simulation Limits

Circuit depth, not qubit count, determines memory usage. Requests exceeding a tier limit
return HTTP 422 with a self-documenting error message.

| Tier | Qubit range | Max entangling depth | Peak memory | Notes |
|---|---|---|---|---|
| 1 | 1–20 | **20 layers** | ~335 MB | Full-depth VQE, QAOA, variational ansätze at any structure. |
| 2 | 21–54 | **9 layers** | ~216 MB | Covers IBM Eagle (27 q) and Osprey (54 q) layouts. |
| 3 | 55–105 | **8 layers** | ~105 MB | Covers Willow-scale (105 q) near-term superconducting layouts. |
| 4 | 106–1,000 | **7 layers** | ~250 MB | At depth 7, a 1,000-qubit circuit uses ~250 MB — within the 350 MB cap. |
| N > 1,000 | — | — | — | Rejected (HTTP 422). Contact us for large-scale access. |

**Depth is counted in entangling layers only** — single-qubit gates (H, Rz, T, etc.) do
not count toward the depth limit and are not restricted.

!!! warning
    `"exact"` mode (statevector) is limited to N ≤ 20 (Tier 1 only). All modes follow
    the tier depth limits above.

---

## Client-side validation

Use `circuit.validate()` or `run(dry_run=True)` to catch limit violations before
submitting. See [Circuit Simulation — Pre-flight validation](circuit.md#pre-flight-validation).
