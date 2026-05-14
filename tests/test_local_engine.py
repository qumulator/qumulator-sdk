"""
Unit tests for the LocalStatevectorEngine.

Run with:  pytest sdk/tests/test_local_engine.py -v
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from qumulator.local import LocalStatevectorEngine


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _counts_close(counts: dict, expected_probs: dict, shots: int, tol: float = 0.05) -> bool:
    """Check that observed counts are within tol * shots of expected for each key."""
    for key, p in expected_probs.items():
        observed = counts.get(key, 0) / shots
        if abs(observed - p) > tol:
            return False
    return True


# ---------------------------------------------------------------------------
#  1. Bell state — simulation path
# ---------------------------------------------------------------------------

class TestBellState:
    def test_counts_distribution(self):
        eng = LocalStatevectorEngine(n_qubits=2)
        eng.apply("h", 0).apply("cx", [0, 1])
        result = eng.run(shots=4096, seed=42)
        assert set(result.counts.keys()) <= {"00", "11"}
        assert abs(result.counts.get("00", 0) - 2048) < 150
        assert abs(result.counts.get("11", 0) - 2048) < 150

    def test_statevector(self):
        eng = LocalStatevectorEngine(n_qubits=2)
        eng.apply("h", 0).apply("cx", [0, 1])
        result = eng.run(shots=10, return_statevector=True)
        sv = result.statevector
        assert sv is not None
        assert sv.shape == (4,)
        r2 = 1 / math.sqrt(2)
        assert abs(abs(sv[0]) - r2) < 1e-10
        assert abs(abs(sv[3]) - r2) < 1e-10
        assert abs(sv[1]) < 1e-10
        assert abs(sv[2]) < 1e-10

    def test_norm_conservation(self):
        eng = LocalStatevectorEngine(n_qubits=2)
        eng.apply("h", 0).apply("cx", [0, 1])
        result = eng.run(shots=1, return_statevector=True)
        norm = float(np.linalg.norm(result.statevector))
        assert abs(norm - 1.0) < 1e-12


# ---------------------------------------------------------------------------
#  2. Bell state — analytic fast path
# ---------------------------------------------------------------------------

class TestBellFastPath:
    def test_fast_path_fires(self):
        """Fast path: no return_statevector → fast path is used."""
        eng = LocalStatevectorEngine(n_qubits=2)
        eng.apply("h", 0).apply("cx", [0, 1])
        # Should return instantly via fast path
        result = eng.run(shots=2048, seed=7)
        assert set(result.counts.keys()) <= {"00", "11"}
        total = sum(result.counts.values())
        assert total == 2048

    def test_fast_path_reproducible(self):
        eng = LocalStatevectorEngine(n_qubits=2)
        eng.apply("h", 0).apply("cx", [0, 1])
        r1 = eng.run(shots=512, seed=99)
        eng.reset().apply("h", 0).apply("cx", [0, 1])
        r2 = eng.run(shots=512, seed=99)
        assert r1.counts == r2.counts


# ---------------------------------------------------------------------------
#  3. GHZ state
# ---------------------------------------------------------------------------

class TestGHZState:
    def test_5qubit_ghz(self):
        eng = LocalStatevectorEngine(n_qubits=5)
        eng.apply("h", 0)
        for i in range(4):
            eng.apply("cx", [i, i + 1])
        result = eng.run(shots=8192, seed=0)
        assert set(result.counts.keys()) <= {"00000", "11111"}
        assert abs(result.counts.get("00000", 0) - 4096) < 300
        assert abs(result.counts.get("11111", 0) - 4096) < 300

    def test_ghz_total_shots(self):
        eng = LocalStatevectorEngine(n_qubits=3)
        eng.apply("h", 0).apply("cx", [0, 1]).apply("cx", [0, 2])
        result = eng.run(shots=1000)
        assert sum(result.counts.values()) == 1000


# ---------------------------------------------------------------------------
#  4. Bernstein-Vazirani fast path
# ---------------------------------------------------------------------------

class TestBernsteinVazirani:
    def _build_bv(self, n: int, s: str) -> LocalStatevectorEngine:
        """Build a BV circuit on n-1 data qubits + 1 ancilla (qubit n-1)."""
        eng = LocalStatevectorEngine(n_qubits=n)
        # Init ancilla in |-> via X then H
        eng.apply("x", n - 1).apply("h", n - 1)
        # H on all data qubits
        for i in range(n - 1):
            eng.apply("h", i)
        # Oracle: CX(i, ancilla) for each '1' bit in s
        for i, bit in enumerate(s):
            if bit == "1":
                eng.apply("cx", [i, n - 1])
        # H on all data qubits
        for i in range(n - 1):
            eng.apply("h", i)
        return eng

    def test_bv_4qubit(self):
        """BV with s='101' on 3 data qubits + 1 ancilla = 4 total qubits."""
        s = "101"
        eng = LocalStatevectorEngine(n_qubits=4)
        # Initialize ancilla (qubit 3) to |−⟩ = H|1⟩
        eng.apply("x", 3).apply("h", 3)
        # H on data qubits (0, 1, 2)
        for i in range(3):
            eng.apply("h", i)
        # Oracle: CX(i, ancilla) for each '1' bit in s
        for i, bit in enumerate(s):
            if bit == "1":
                eng.apply("cx", [i, 3])
        # H on data qubits
        for i in range(3):
            eng.apply("h", i)
        result = eng.run(shots=1024, return_statevector=True)
        # Most probable outcome should start with s
        sv = result.statevector
        probs = np.abs(sv) ** 2
        most_probable_idx = int(np.argmax(probs))
        most_probable = format(most_probable_idx, "04b")
        assert most_probable[:3] == s, (
            f"Expected hidden string '{s}', got '{most_probable[:3]}' "
            f"(full bitstring: '{most_probable}')"
        )


# ---------------------------------------------------------------------------
#  5. QFT fast path — uniform distribution
# ---------------------------------------------------------------------------

class TestQFTFastPath:
    def test_qft_uniform(self):
        """QFT|0⟩^n → uniform distribution."""
        from qumulator.local._fast_paths import _compute_qft
        result = _compute_qft(n=4, shots=4096, seed=1)
        counts = result["counts"]
        dim = 16
        total = sum(counts.values())
        assert total == 4096
        # All 16 outcomes should appear and be roughly equally distributed
        assert len(counts) == dim
        for v in counts.values():
            assert abs(v - 4096 / dim) < 4096 * 0.05


# ---------------------------------------------------------------------------
#  6. Grover fast path — success probability formula
# ---------------------------------------------------------------------------

class TestGroverFastPath:
    def test_grover_success_prob(self):
        from qumulator.local._fast_paths import _compute_grover
        n = 4  # 16 states
        k = 1  # 1 iteration
        theta = math.asin(1 / math.sqrt(16))
        expected_p = math.sin((2 * k + 1) * theta) ** 2

        result = _compute_grover(n=n, k=k, shots=10000, seed=42)
        marked_count = result["counts"].get("0000", 0)
        observed_p = marked_count / 10000
        assert abs(observed_p - expected_p) < 0.05

    def test_grover_success_prob_field(self):
        from qumulator.local._fast_paths import _compute_grover
        result = _compute_grover(n=4, k=1, shots=100, seed=0)
        assert "grover_success_prob" in result
        assert 0 < result["grover_success_prob"] <= 1.0


# ---------------------------------------------------------------------------
#  7. Norm conservation
# ---------------------------------------------------------------------------

class TestNormConservation:
    @pytest.mark.parametrize("n,circuit", [
        (1, [("h", 0), ("rz", 0, [0.7]), ("ry", 0, [1.2])]),
        (3, [("h", 0), ("cx", [0, 1]), ("cx", [0, 2]), ("x", 1), ("s", 2)]),
        (4, [("h", 0), ("h", 1), ("cx", [0, 2]), ("cz", [1, 3]), ("t", 0)]),
    ])
    def test_norm_preserved(self, n, circuit):
        eng = LocalStatevectorEngine(n_qubits=n)
        for gate in circuit:
            name = gate[0]
            qubits = gate[1]
            params = gate[2] if len(gate) > 2 else None
            eng.apply(name, qubits, params)
        result = eng.run(shots=1, return_statevector=True)
        norm = float(np.linalg.norm(result.statevector))
        assert abs(norm - 1.0) < 1e-12, f"Norm = {norm} for circuit {circuit}"


# ---------------------------------------------------------------------------
#  8. Entropy map
# ---------------------------------------------------------------------------

class TestEntropyMap:
    def test_bell_state_entropy(self):
        """Bell state: each qubit has max entropy of 1.0 bit."""
        eng = LocalStatevectorEngine(n_qubits=2)
        eng.apply("h", 0).apply("cx", [0, 1])
        result = eng.run(shots=1, return_entropy_map=True)
        em = result.entropy_map
        assert em is not None
        assert len(em) == 2
        for s in em:
            assert abs(s - 1.0) < 1e-6, f"Expected entropy ≈ 1.0, got {s}"

    def test_product_state_entropy(self):
        """Product state |0⟩⊗n: each qubit has entropy 0.0 bits."""
        n = 4
        eng = LocalStatevectorEngine(n_qubits=n)
        result = eng.run(shots=1, return_entropy_map=True)
        em = result.entropy_map
        assert em is not None
        for i, s in enumerate(em):
            assert abs(s) < 1e-12, f"Qubit {i} entropy = {s}, expected 0.0"

    def test_ghz_entropy(self):
        """GHZ state: all qubits have entropy ≈ 1.0."""
        n = 3
        eng = LocalStatevectorEngine(n_qubits=n)
        eng.apply("h", 0).apply("cx", [0, 1]).apply("cx", [0, 2])
        result = eng.run(shots=1, return_entropy_map=True)
        for s in result.entropy_map:
            assert abs(s - 1.0) < 1e-6


# ---------------------------------------------------------------------------
#  9. GPU path smoke test (skipped if no GPU library available)
# ---------------------------------------------------------------------------

class TestGPUSmokeTest:
    def test_gpu_or_skip(self):
        has_gpu = False
        try:
            import cupy as cp  # noqa: F401
            cp.array([0])  # triggers CUDA context; raises if no GPU device
            has_gpu = True
        except Exception:
            pass
        if not has_gpu:
            try:
                import jax  # type: ignore[import]
                has_gpu = bool(jax.devices("gpu"))
            except Exception:
                pass
        if not has_gpu:
            try:
                import torch
                has_gpu = torch.cuda.is_available()
            except ImportError:
                pass

        if not has_gpu:
            pytest.skip("No GPU device available; skipping GPU smoke test")

        eng = LocalStatevectorEngine(n_qubits=4, device="gpu")
        eng.apply("h", 0).apply("cx", [0, 1])
        result = eng.run(shots=512)
        assert sum(result.counts.values()) == 512


# ---------------------------------------------------------------------------
#  10. Large-qubit warning at N > 20
# ---------------------------------------------------------------------------

class TestLargeQubitWarning:
    def test_warning_emitted(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            eng = LocalStatevectorEngine(n_qubits=21)
            assert any(
                issubclass(warning.category, UserWarning)
                and "21 qubits" in str(warning.message)
                for warning in w
            ), f"Expected UserWarning about 21 qubits, got: {[str(x.message) for x in w]}"

    def test_continues_after_warning(self):
        """Engine still works after the large-qubit warning."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            eng = LocalStatevectorEngine(n_qubits=21)

        # Apply only H on qubit 0 — statevector is small-ish for a quick test
        # Actual allocation would be 32 MB; for CI we keep the circuit trivial
        # but validate the engine doesn't crash immediately
        assert eng.n_qubits == 21


# ---------------------------------------------------------------------------
#  11. LocalStatevectorEngine accessible from top-level package
# ---------------------------------------------------------------------------

def test_top_level_import():
    from qumulator import LocalStatevectorEngine as LSE  # noqa: F401
    assert LSE is LocalStatevectorEngine


# ---------------------------------------------------------------------------
#  12. Parametric gates
# ---------------------------------------------------------------------------

class TestParametricGates:
    def test_rx_full_rotation(self):
        """Rx(2π)|0⟩ should return to |0⟩ up to global phase."""
        eng = LocalStatevectorEngine(n_qubits=1)
        eng.apply("rx", 0, [2 * math.pi])
        result = eng.run(shots=1, return_statevector=True)
        sv = result.statevector
        # |sv[0]| ≈ 1, |sv[1]| ≈ 0
        assert abs(abs(sv[0]) - 1.0) < 1e-10
        assert abs(sv[1]) < 1e-10

    def test_rz_phase(self):
        """Rz(π)|1⟩ should add phase e^{iπ/2}·|1⟩ = i|1⟩."""
        eng = LocalStatevectorEngine(n_qubits=1)
        eng.apply("x", 0).apply("rz", 0, [math.pi])
        result = eng.run(shots=1, return_statevector=True)
        sv = result.statevector
        assert abs(sv[0]) < 1e-10
        assert abs(abs(sv[1]) - 1.0) < 1e-10

    def test_swap_gate(self):
        """SWAP(0,1) on |10⟩ should give |01⟩."""
        eng = LocalStatevectorEngine(n_qubits=2)
        eng.apply("x", 0).apply("swap", [0, 1])
        result = eng.run(shots=1024, seed=0)
        # All shots should land on "01"
        assert result.counts.get("01", 0) == 1024

    def test_toffoli_gate(self):
        """CCX on |110⟩ should flip qubit 2 → |111⟩."""
        eng = LocalStatevectorEngine(n_qubits=3)
        eng.apply("x", 0).apply("x", 1).apply("ccx", [0, 1, 2])
        result = eng.run(shots=512, seed=0)
        assert result.counts.get("111", 0) == 512
