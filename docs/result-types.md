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
| `phase_label` | `str\|None` | KLT entanglement-phase label (Z1–Z5) returned when using `mode="klt_phase"`. |
| `gaussian_certificate` | `GaussianCertificate\|None` | Populated when `mode="gaussian"`. |
| `most_probable` | `str` (property) | Bitstring with the highest count |

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
