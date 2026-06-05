# Result Types

## CircuitResult

| Field | Type | Description |
|---|---|---|
| `counts` | `dict[str, int]` | Measurement outcome counts. Keys are MSB-first bitstrings (qubit 0 leftmost). |
| `n_qubits` | `int` | Register width |
| `shots` | `int` | Total measurement samples |
| `statevector` | `ndarray\|None` | Complex amplitude vector of length 2ᴺ. Populated when `return_statevector=True`. |
| `probabilities` | `ndarray\|None` | Probability vector. Populated when `return_probabilities=True`. |
| `entropy_map` | `list[float]\|None` | Per-qubit entanglement entropy in bits. 1.0 = maximally entangled; 0.0 = product state. |
| `f_Q_density` | `float\|None` | QFI density (Tóth–Gühne 2012). `f_Q > k` certifies genuine (k+1)-partite entanglement. |
| `entanglement_depth` | `int\|None` | Certified entanglement depth = ⌊f_Q⌋. 0 = separable. |
| `predicted_tvd` | `float\|None` | TVD upper bound to the exact distribution. `0.0` for unconditionally exact modes. |
| `phase_label` | `str\|None` | MPS bond entanglement regime label. One of: `"product_state"` (S < 0.1 bits), `"area_law"` (0.1–0.6), `"topological_class"` (0.6–1.2, SPT/Haldane/AKLT regime), `"near_volume_law"` (1.2–2.5), `"volume_law"` (≥ 2.5 bits). Returned by `client.evolve.aklt()` and related TEBD endpoints. `None` for circuit simulation jobs. |
| `gaussian_certificate` | `GaussianCertificate\|None` | Populated when `mode="gaussian"`. |
| `most_probable` | `str` (property) | Bitstring with the highest count |

---

## Phase Labels (MPS bond entanglement regimes)

`phase_labels` is a list of per-bond entanglement regime identifiers returned by `client.evolve.aklt()` and other TEBD endpoints. Each entry characterises the quantum entanglement across one MPS bond, derived from the von Neumann entropy S of the bond's Schmidt decomposition.

| Label | Entropy range | Physical description |
|---|---|---|
| `"product_state"` | S < 0.1 bits | Unentangled — qubits on either side of the bond are independent. |
| `"area_law"` | 0.1 – 0.6 bits | Area-law entanglement — bounded by O(1) per bond; MPS-tractable for any depth. |
| `"topological_class"` | 0.6 – 1.2 bits | SPT / topological phase regime. The AKLT Valence Bond Solid has S = 1 bit per inter-site bond and lands here exactly. |
| `"near_volume_law"` | 1.2 – 2.5 bits | High entanglement, approaching the volume-law limit. |
| `"volume_law"` | S ≥ 2.5 bits | Near-maximal entanglement — Haar-random / volume-law circuit regime. |

Thresholds are calibrated so that canonical quantum states land at their exact positions:
- Bell state |Φ⁺⟩: S = 1 bit exactly → mid `"area_law"` / `"topological_class"` boundary
- AKLT inter-site bond: S = 1 bit exactly → `"topological_class"`
- GHZ chain bond: S = 1 bit → `"topological_class"`
- Random unitary (volume-law): S → log₂(χ) → `"volume_law"`
- Product state: S = 0 → `"product_state"`

---

## GaussianCertificate

| Field | Type | Description |
|---|---|---|
| `rcs_certificate` | `str` | `GAUSSIAN_SIMULABLE` \| `LIKELY_GAUSSIAN` \| `NON_GAUSSIAN_CORRECTION_NEEDED` |
| `entanglement_regime` | `str\|None` | `area_law` \| `transitional` \| `volume_law` |
| `wigner_negativity_estimate` | `float\|None` | Estimated non-Clifford correction from T-gate content |
| `gaussian_fidelity` | `float\|None` | Estimated fidelity of the Gaussian approximation (0–1) |
| `xeb_lower_bound` | `float\|None` | Cross-entropy benchmark lower bound |

---

## JobStatus

| Field | Type | Description |
|---|---|---|
| `job_id` | `str` | Job identifier |
| `status` | `str` | `queued` \| `running` \| `completed` \| `failed` |
| `result` | `dict\|None` | Engine result when completed |
| `error` | `str\|None` | Error message when failed |
| `is_done` | `bool` | Property: True when completed or failed |
| `ok` | `bool` | Property: True when completed |
| `entanglement_dof` | `float\|None` | Effective entanglement degrees of freedom of the circuit at job completion. Derived from the singular-value spectrum of the off-diagonal 1-RDM block: `dof = (Σ σᵢ)² / Σ σᵢ²`. `None` for jobs that do not run a circuit engine (e.g. HOMO, hafnian). |
| `dof_converged` | `bool\|None` | `True` when the entanglement DOF has stopped growing between layers (convergence criterion: relative change < 1 %). Indicates the circuit has reached its entanglement saturation point. `None` when `entanglement_dof` is not available. |
