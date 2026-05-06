# Changelog

All notable changes to the Qumulator SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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

### Changed
- README tagline updated: *"Simulate quantum circuits up to 1,000 qubits at low
  entanglement depth on classical hardware"* (was "Simulate 1,000-qubit quantum circuits")
- README: added "How Qumulator compares" section with a four-row competitive table
  (Qumulator, Qiskit Aer, BlueQubit, PennyLane) and a Rule of thumb paragraph covering
  Qumulator's full simulation spectrum
- README: added footnote to the simulation modes table clarifying that MPS modes are
  subject to the tier depth limit (max 7 entangling layers for N > 105)
- README: added inline Pricing section with CU definition and plan table (Free / Starter
  / Professional) so pricing is discoverable from the repo without following an external link
- README: free tier table updated — "Beta only — may be discontinued at any time" replaced
  with "Public beta"; "1,000 (all tiers)" clarified to "1,000 — see tier depth limits table"
- README: CLI demo comment updated from "1000-qubit GHZ" to "1,000-qubit Bell pairs (depth 1)"
- README: Documentation section now links GitHub Pages (`qumulator.github.io/qumulator-sdk`)
  as the primary reference alongside the existing `qumulator.com` link
- README: added `[![Docs]` badge to the header badge row

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
- `CircuitResult.predicted_tvd` — model-calibrated total variation distance bound per KLT
  chaos phase; `0.0` for unconditionally exact modes (`'exact'`, `'cluster'`)
- `CircuitResult.phase_label` — KLT chaos-regime label (`Z1`–`Z5`) returned from the engine
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
  `LIKELY_GAUSSIAN`, or `NON_GAUSSIAN_CORRECTION_NEEDED`, with Kaplan-Yorke dimension
  and Koopman mode count fields
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
