# Changelog

All notable changes to the Qumulator SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

> **A note on breaking changes in the upcoming release**
>
> During early development, several internal codenames crept into the public API
> (`KLT`, `klt`, `Vortex`, `GMPS`, `GESS`, `Nexus`). These names meant something
> internally but were opaque and confusing to anyone reading the code or docs for
> the first time. Rather than carry that confusion forward to v1.0, we decided to
> rename everything clearly while the SDK is still in beta and the user base is small.
> We know breaking changes are painful — we kept the list tight and the new names
> are straightforward. Please update to the latest version and replace the old names
> using the table in the `[Unreleased]` section below.
>
> **`pip install --upgrade qumulator-sdk`**

---

## [Unreleased]

### Changed (Breaking — Renamed)

- **`KLTClient`** renamed to **`HamiltonianClient`**; access via `client.hamiltonian` (was `client.klt`).
- **`KLTResult`** renamed to **`SpinGroundStateResult`**.
- **`AKLTResult.klt_labels`** renamed to **`AKLTResult.phase_labels`**.
- Circuit mode strings renamed: `"exact"` → `"statevector"`, `"tensor"` → `"mps"`,
  `"compressed"` → `"cluster_mps"`, `"cluster"` → `"cluster_statevector"`.

### Added

#### Auto-routing mode (`mode="auto"`) + Preflight API

- **`mode="auto"`** — New first-class simulation mode. The engine analyses the circuit’s
  entanglement graph, estimates the Kaplan–Yorke dimension (D_KY), and routes to the
  optimal backend before simulation begins. Resolved mode and diagnostics are returned
  in `CircuitResult.resolved_mode` and `CircuitResult.preflight_report`.

- **`CircuitResult.resolved_mode`** (`Optional[str]`) — The mode actually used after
  auto-routing. Populated for `mode="auto"` jobs; mirrors the requested mode for explicit
  mode selections.

- **`CircuitResult.preflight_report`** (`Optional[Dict]`) — Routing diagnostics for
  `mode="auto"` jobs: `d_ky`, `entanglement_regime`, `reasoning`, `is_tree`,
  `edge_density`, `d_s`, `n_2q_gates`, `n_t_gates`, `ky_gp_consistent`.

- **`CircuitClient.preflight(qasm_source)`** — Zero-cost pre-flight analysis. Builds
  the entanglement graph and returns the routing recommendation without running any
  simulation. No compute units are deducted.

- **`CircuitClient.preflight_instructions(n_qubits, gates)`** — Same as `preflight()`
  but accepts a gate-instruction list instead of QASM.

### Changed (Breaking — Round 2 Renamed)

- Circuit mode `"cluster"` more precisely: `"cluster"` now refers to the O(N)
  pair-product ClusterState backend; the exact-statevector-per-cluster engine uses
  `"cluster_statevector"`. If you were using `mode="cluster"` expecting exact results,
  switch to `mode="cluster_statevector"`.
- Internal wire-protocol strings: all `klt_*` mode strings replaced with clean names.
  These were never public SDK API but are documented here for completeness:
  `"klt_phase"` → `"phase"`, `"klt_stone"` → `"hamiltonian"`, `"klt_mps"` → `"mps"`,
  `"klt_cluster_mps"` → `"cluster_mps"`, `"klt_greens"` → `"greens"`,
  `"klt_gaussian"` → `"gaussian"`, `"klt_matrix_engine"` / `"klt_t4"` → `"dyson"`.
- Engine class `ExactClusterEngine` → `ClusterStatevectorEngine` (internal; not in public SDK).
- Engine class `KitaevChainEngine` replaces `KLTKitaevChainEngine` (internal only).

### Changed (Notebooks)

- `molecular_gmps_quickstart.ipynb` renamed to `molecular_mps_quickstart.ipynb`.
- `klt_cluster_demo.ipynb` renamed to `cluster_statevector_demo.ipynb`.
- `klt_greens_demo.ipynb` renamed to `greens_demo.ipynb`.
- `vortex_geometry.ipynb` renamed to `cluster_geometry.ipynb`.

### Added

- **`JobStatus.entanglement_dof`** (`Optional[float]`) — Effective entanglement
  degrees of freedom of the circuit simulation at job completion. Derived from the
  singular-value spectrum of the off-diagonal 1-RDM block:
  `dof = (Σσᵢ)² / Σσᵢ²`. `None` for non-circuit jobs (HOMO, hafnian, molecular).

- **`JobStatus.dof_converged`** (`Optional[bool]`) — `True` when `entanglement_dof`
  has stopped growing between layers (relative change < 1 %). Signals entanglement
  saturation: additional circuit depth will not increase correlation.
  `None` when `entanglement_dof` is not available.

- **Docs** — `result-types.md`, `rest-jobs.md`, and `README.md` updated with field
  descriptions and a polling JSON example showing the new fields.

### Fixed (Docs)

- **`README.md`** — Corrected the breaking-changes rename table: `mode="auto"` was
  incorrectly listed as `*(removed)* — use mode="mps"`. `mode="auto"` was re-added in
  v0.4.0 as a first-class auto-routing mode; the table now correctly shows it as
  **re-added** with a description of its preflight / D_KY routing behaviour.
- **`docs/circuit.md`** — `bond_dim` parameter description referenced the old mode
  names `"compressed"` and `"tensor"`; corrected to `"cluster_mps"` and `"mps"`.
  `return_statevector` and `return_probabilities` described as requiring `"exact"` mode;
  corrected to `"statevector"`.
- **`docs/qasm.md`** — Code example used `mode="exact"`; corrected to `mode="statevector"`.
- **`docs/qiskit.md`** — Code comment and `mode=` argument used old name `"compressed"`;
  corrected to `"cluster_mps"`.
- **`docs/limits.md`** — Warning admonition referred to `"exact"` mode; corrected to
  `"statevector"`.
- **`docs/index.md`** — Removed duplicate sentence in the open-source admonition.
- **`docs/notebooks.md`** — Removed duplicate `## Demo notebooks` section heading;
  renamed second occurrence to `## Quantum Computing Demos`.

---

## [0.5.1] — 2026-05-28

### Fixed

- **`notebooks/dmrg_quickstart.ipynb`** — Corrected the H₂O larger-active-space
  example: `CAS(7,7)` with `ncas=7, nelecas=(4,3)` was invalid (PySCF computes a
  non-integer `ncore` and aborts); replaced with `CAS(4,6)`: `ncas=4,
  nelecas=(3,3)`, `n_elec=[3, 3]`. Also replaced the hardcoded
  `ao2mo.restore(..., 7)` with `ao2mo.restore(..., mc2.ncas)` for correctness.
- **`notebooks/molecular_gmps_quickstart.ipynb`** — Fixed Givens gate format in the
  circuit example: `{"type": "givens", "i": 0, "j": 1, "theta": 0.3}` →
  `{"qi": 0, "qj": 1, "theta": 0.3}` to match the `GivensGate` backend model
  (old format caused HTTP 422).

---

## [0.5.0] — 2026-05-20

### Added

- **`MolecularClient` (`client.molecular`)** — SDK client for exact molecular
  active-space energy via GMPS/MPO (Geometric MPS with Matrix Product Operators).
  Accepts 1e/2e integrals and an optional Givens orbital-rotation circuit; supports
  up to 50 active orbitals (100 spin-orbital qubits).
  - `client.molecular.energy(h1e, h2e, n_elec, e_nuc, circuit=None, coup_thr=1e-6)`
  - Returns `MolecularEnergyResult(energy, n_qubits, n_orb, n_components, zz_correlators)`
  - Multi-fragment drug molecules (e.g. aspirin 4+5-orbital system) achieve up to
    7282× memory compression vs. the full statevector.

- **`DMRGClient` (`client.dmrg`)** — SDK client for two-site DMRG ground-state energy.
  No circuit ansatz required; variational sweeps with configurable bond dimension.
  - `client.dmrg.energy(h1e, h2e, n_elec, e_nuc, d_max=64, n_sweeps=8, tol=1e-10)`
  - Returns `DMRGEnergyResult(energy, converged, n_sweeps_run, d_max_used, n_orb, n_so, wall_time_s)`
  - H₂ CAS(2,2) exact at `d_max=64` (energy = −1.13728383 Ha); see `dmrg_quickstart.ipynb`.

- **`MolecularEnergyResult` model** — Pydantic v2 response model with `energy`,
  `n_qubits`, `n_orb`, `n_components`, and optional `zz_correlators`.

- **`DMRGEnergyResult` model** — Pydantic v2 response model with `energy`, `converged`,
  `n_sweeps_run`, `d_max_used`, `n_orb`, `n_so`, `wall_time_s`.

- **`QumulatorClient.molecular`** and **`QumulatorClient.dmrg`** — new attributes on the
  top-level client exposing `MolecularClient` and `DMRGClient` respectively.

- **SDK unit tests** — `tests/test_molecular_gmps.py` (13 tests) and `tests/test_dmrg.py`
  (13 tests) covering payload serialisation, response deserialisation, and HTTP error
  propagation.  Full suite: 98 tests, 0 failures.

---

## [0.4.2] — 2026-05-14

### Fixed

- **`src/qumulator/resources.py` (`NotebookClient.submit`)** — Fixed broken notebook
  submission: was sending raw bytes with `Content-Type: application/octet-stream`,
  but the backend requires `multipart/form-data` with the file in a `notebook` field.
  All calls to `client.notebook.run()` / `client.notebook.submit()` returned HTTP 422
  `Field required` before this fix.

---

## [0.4.1] — 2026-05-14

### Fixed

- **`notebooks/h2_ground_state.ipynb`** — Removed hardcoded WSL engine path
  (`/mnt/c/Projects/qumulator/engine/engines`); added `QUMULATOR_API_KEY` /
  `QumulatorClient` setup cell; embedded verified JW Pauli Hamiltonian (15 terms, 4
  qubits); corrected STO-3G reference energies to pyscf values (HF: −1.1168 Ha,
  FCI: −1.1373 Ha); added `client.klt.run()` cloud fallback so the notebook runs
  without a local engine.
- **`notebooks/n2_ground_state.ipynb`** — Same set of fixes as H₂: removed WSL path,
  added API key + client setup, embedded N₂ JW Pauli Hamiltonian (383 terms, 12
  qubits), added `client.klt.run()` cloud fallback.
- **`notebooks/willow_rcs_benchmark.ipynb`** — Cleared stale cell outputs from a
  partial test run (cells contained a saved HTTP 500 error).

---

## [0.4.0] — 2026-05-14

> **The Qumulator statevector engine is now open-source.**
> Now you can run the statevector engine locally, with full source code.
>
> `LocalStatevectorEngine` ships inside the SDK — no API key, no account, no network
> connection required. **No qubit limits** — simulate as many qubits as your hardware
> can hold. GPU acceleration (CuPy / JAX / PyTorch) is auto-detected and zero-config.
> Install once, run anywhere.
>
> ```bash
> pip install qumulator-sdk          # CPU — unlimited qubits, pure NumPy
> pip install "qumulator-sdk[gpu]"   # GPU — CuPy / JAX / PyTorch, auto-detected
> ```

### Added

- **`LocalStatevectorEngine`** — a complete local statevector simulator shipped as part
  of `qumulator-sdk`. No API key, no cloud account, no network connection required.
  Simulates arbitrary quantum circuits up to any qubit count using pure NumPy; optional
  GPU acceleration via CuPy, JAX, or PyTorch (`pip install qumulator-sdk[gpu]`).

  ```python
  from qumulator.local import LocalStatevectorEngine

  eng = LocalStatevectorEngine(n_qubits=10)
  eng.apply('h', 0)
  for i in range(9):
      eng.apply('cx', [i, i+1])
  result = eng.run(shots=4096, return_entropy_map=True)
  print(result.counts)       # {'0000000000': ~2048, '1111111111': ~2048}
  print(result.entropy_map)  # [~1.0, ~1.0, ...]  per-qubit entanglement
  ```

- **Analytic fast paths** — four circuit patterns execute in O(1) time without
  allocating or evolving a statevector:

  | Pattern | Detection | Output |
  |---|---|---|
  | Bell state | `H(0) + CNOT(0,1)` on 2 qubits | 50% `|00⟩`, 50% `|11⟩` |
  | Bernstein-Vazirani | `H⊗n + oracle + H⊗n` | hidden bitstring *s* with prob 1 |
  | QFT | H + controlled-phase ladder | uniform over all `2^n` states |
  | Grover | H init + oracle/diffusion × k | marked state with `sin²((2k+1)θ)` |

- **`entropy_map`** — per-qubit von Neumann entanglement entropy (log-base-2) via
  reduced density matrix SVD; available via `return_entropy_map=True` on both
  `LocalStatevectorEngine.run()` and the cloud `CircuitEngine.run()`.

- **`[gpu]` optional dependency group** — `pip install qumulator-sdk[gpu]` installs
  CuPy (NVIDIA CUDA), JAX (Google XLA), and PyTorch. The engine auto-detects availability
  in that order and falls back to NumPy silently.

- **`mode='local'` on `CircuitClient.engine()`** — existing cloud clients can opt into
  local simulation for development / offline use:

  ```python
  client = QumulatorClient(api_key="...")
  eng = client.circuit.engine(n_qubits=5, mode="local")
  eng.apply("h", 0).apply("cx", [0, 1])
  result = eng.run(shots=1024)
  ```

- **Large-qubit warning** — `UserWarning` emitted at `n_qubits > 20` with the expected
  memory footprint; simulation continues normally.

- **25 new unit tests** in `tests/test_local_engine.py` covering: Bell/GHZ states,
  Bernstein-Vazirani, QFT and Grover fast paths, norm conservation, entropy map (Bell,
  product state, GHZ), GPU smoke test, parametric gates (Rx, Rz, SWAP, Toffoli).
- **`client.evolve.aklt(n_sites, observables, string_order_pairs)`** — new `EvolveClient`
  method for exact AKLT Valence Bond Solid state preparation via `/evolve/aklt`:
  - Returns `AKLTResult` with `bond_entropy`, `mean_bond_entropy` (≈0.50 bits; inter-site
    bonds = 1.0 bit, intra-site bonds = 0.0 bit), `max_bond_dim` (=2 exact),
    `klt_labels` (Z3 on inter-site bonds, Z1 on intra-site), `string_order`
    map (values = −0.250 = −1/4 for all site pairs), and optional QFI/correlators
  - Example: `vbs = client.evolve.aklt(n_sites=10, observables=["entropy","string_order"])`
- **`"aklt"` preset in `client.evolve.run()` and `client.evolve.ground()`**:
  - `n_qubits` must equal `2 × n_sites`; parameters `J_AF` (default 1.0) and `J_FM` (default 2.0)
  - **Important**: pass `initial_state="neel"` to `client.evolve.ground()` — the default
    `|0…0⟩` state is a fixed point of the AKLT propagator and will never converge
- **`initial_state` parameter on `client.evolve.ground()`** — new optional `str` argument
  (`"zero"` | `"neel"` | `"ferromagnet"`, default `"zero"`).  Required for AKLT.
- `CircuitEngine.validate()` — client-side pre-flight check that counts entangling
  layers and compares against the published tier depth limits; raises `ValueError` with
  a descriptive message (qubit count, actual depth, tier limit, recommended alternative)
  before making any API call
- `dry_run: bool = False` parameter on `CircuitEngine.run()` — validates the circuit
  client-side and returns `None` without submitting; raises `ValueError` if invalid
- `CircuitEngine.run()` now calls `validate()` automatically on every submission so
  tier violations are caught locally before any network round-trip
- MkDocs-based documentation site (`mkdocs.yml`, `docs/`, 21 Markdown pages) covering
  all SDK features, REST API endpoints, and reference types; hosted at
  `qumulator.github.io/qumulator-sdk`
- GitHub Actions workflow (`.github/workflows/docs.yml`) auto-deploys the docs site to
  the `gh-pages` branch on every push to `main` that touches `docs/` or `mkdocs.yml`
- **`notebooks/options_pricer.ipynb`** — European call option pricer using quantum
  amplitude estimation on a 5-qubit circuit (4 register + 1 ancilla); prices within
  <1% of Black-Scholes for ATM calls; includes Greeks sweep and dark-theme visualisation;
  opens in Google Colab
- **`notebooks/fantasy_football.ipynb`** — DraftKings daily fantasy football lineup
  optimizer; encodes the 9-player, $50,000 salary-cap roster selection as a 40-variable
  QUBO and solves with `client.klt`; includes classical greedy baseline comparison and
  KLT confidence-score chart; opens in Google Colab
- **`notebooks/h2_ground_state.ipynb`** — H₂ ground state via 4-qubit exact simulation,
  CASCI(2,2)/STO-3G, 100% correlation energy recovery; opens in Google Colab
- **`notebooks/lih_ground_state.ipynb`** — LiH ground state via KLT Pauli-Hamiltonian
  solver, 1.15 mHa error, chemical accuracy; opens in Google Colab
- **`notebooks/n2_ground_state.ipynb`** — N₂ ground state via 12-qubit exact simulation,
  79 kcal/mol correlation recovered (100%), 3.2 s runtime; opens in Google Colab

### Changed

- `sdk` version bumped to **0.4.0**.
- `LocalStatevectorEngine` exported from `qumulator` top-level package.
- **`GaussianCertificate.kaplan_yorke_dim`** renamed to **`spectral_complexity_dim`**;
  **`GaussianCertificate.koopman_mode_count`** renamed to **`spectral_mode_count`**. The
  experimental Kaplan-Yorke and Koopman-mode fields have been replaced with neutral spectral
  complexity equivalents. Values and semantics are unchanged — only the field names differ.
- README tagline updated to reflect unlimited local simulation and cloud MPS up to 1,000 qubits
- README: added "How Qumulator compares" competitive table and Rule of thumb paragraph
- README: added inline Pricing section; free-tier table and CLI demo comment updated
- README: Documentation section links GitHub Pages as primary reference; `[![Docs]` badge added

### Removed

- Experimental Z1–Z5 correction layers removed from the KLT simulation engine. Phase labels
  (Z1–Z5) are still returned as entanglement analysis fields in every result; the correction
  calls themselves were speculative and could degrade accuracy. No changes to the SDK API or
  response models.

---

## [0.3.0] - 2026-05-03

### Added
- `EvolveClient` (`client.evolve`) — real-time TEBD Hamiltonian evolution, imaginary-time
  ground-state preparation, QKZM quench, and 2-D lattice evolution
- Circuit execution modes `'cluster'` (exact cluster-factorization, O(Σ 2^k_c) memory,
  TVD = 0) and `'greens'` (free-fermion Green's function encoder, O(N²) memory)
- CLI `--evolve` demo: 10-site TFIM real-time quench with QFI and entropy trajectory output
- `CircuitResult.f_Q_density` — Quantum Fisher Information density (Tóth–Gühne 2012)
  certifying genuine multipartite entanglement; `f_Q > k` certifies (k+1)-partite entanglement
- `CircuitResult.entanglement_depth` — certified entanglement depth `floor(f_Q_density)`
- `CircuitResult.predicted_tvd` — model-calibrated total variation distance bound per
  entanglement phase; `0.0` for unconditionally exact modes (`'exact'`, `'cluster'`)
- `CircuitResult.phase_label` — KLT entanglement-phase label (`Z1`–`Z5`) returned from the engine
- Colab demo notebooks: `klt_cluster_demo`, `klt_greens_demo`, `tebd_quench_demo`,
  `chsh_bell`, `h12_vqe`, `prime_factoring`, `vortex_geometry`

### Changed
- README expanded with `cluster`, `greens`, and `evolve` API examples
- README Colab notebook table updated with all new notebooks

### Fixed
- PyRight / mypy type errors resolved across `circuit.py`, `models.py`, and `resources.py`

---

## [0.2.1] - 2026-05-01

### Fixed
- `return_statevector=True` now returns a correct `numpy.ndarray` — the engine encodes
  amplitudes as separate `statevector_real` / `statevector_imag` arrays; the client
  now reassembles them into a single complex vector

### Changed
- CI publish workflow switched from OpenID Connect trusted publishing to `PYPI_API_TOKEN`
  secret for broader environment compatibility

---

## [0.2.0] - 2026-04-29

### Added
- `QumulatorClient()` zero-argument constructor: reads `QUMULATOR_API_URL` and
  `QUMULATOR_API_KEY` from the environment automatically — no keyword args required
- CLI `qumulator run <file.qasm>` — submit an OpenQASM 2.0 file and print the result
- Windows PowerShell instructions for API key generation added to README
- Tier 2 and Tier 3 entangling-depth limits documented in the README free-tier table

### Changed
- README quickstart updated to use parallel Bell pairs (depth-1 circuit, always within
  free-tier limits) instead of a linear GHZ chain
- Qiskit and Cirq backend constructors now accept zero arguments and read credentials
  from the environment, matching `QumulatorClient`

### Fixed
- Statevector mode max-qubit limit corrected from 25 to 20 in README
- Tier depth limits corrected: Tier 2 → 9, Tier 3 → 8, Tier 4 → 7
- 54-qubit MPS cap removed from free-tier table (no such cap applies)
- `willow_rcs_benchmark` notebook field names corrected for `GaussianCertificate` model
- Updated README pricing from placeholder estimates to calculated cloud-hosting costs; corrected free-tier rate limits

---

## [0.1.0] - 2026-04-18

### Added
- Initial release of the Qumulator Python SDK
- `QumulatorClient` with sub-clients: `circuit`, `homo` (DFT HOMO/LUMO), `klt`
  (Ising/spin Hamiltonian ground states), `hafnian` (photonic GBS), `majorana`, `vortex`
- `CircuitEngine` fluent builder API (`eng.apply('h', 0).apply('cx', [0, 1])`) plus
  gate-list API and `run()` / `sample()` / `evolve_hamiltonian()` methods
- Circuit execution modes: `'auto'`, `'exact'`, `'compressed'`, `'tensor'`,
  `'hamiltonian'`, `'gaussian'`
- `GaussianCertificate` model classifying circuits as `GAUSSIAN_SIMULABLE`,
  `LIKELY_GAUSSIAN`, or `NON_GAUSSIAN_CORRECTION_NEEDED`, with spectral complexity
  metric fields
- Qiskit drop-in backend (`qumulator-sdk[qiskit]`) and Cirq simulator (`qumulator-sdk[cirq]`)
- CLI: `qumulator demo` (50-qubit Bell, 105-qubit Willow, wormhole, anyon braiding demos),
  `qumulator key` (prints key-generation instructions)
- `JobStatus` model with `is_done` / `ok` helpers for async job polling
- CI workflows: pytest on every push; automatic PyPI publish on version tag

---

[Unreleased]: https://github.com/qumulator/qumulator-sdk/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/qumulator/qumulator-sdk/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/qumulator/qumulator-sdk/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/qumulator/qumulator-sdk/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/qumulator/qumulator-sdk/releases/tag/v0.1.0
