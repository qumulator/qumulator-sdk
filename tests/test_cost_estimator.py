"""
Unit tests for the client-side CU cost estimator.

No API calls are made — everything is purely client-side formula evaluation.
Run with:  pytest sdk/tests/test_cost_estimator.py -v
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from qumulator.circuit import (
    CircuitEngine,
    CircuitClient,
    CostEstimate,
    _compute_cu_estimate,
    _normalise_gate_list,
)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _mock_engine(n_qubits: int, mode: str = "auto", bond_dim=None) -> CircuitEngine:
    """Return a CircuitEngine backed by a MagicMock client (no HTTP)."""
    client = MagicMock(spec=CircuitClient)
    return CircuitEngine(client, n_qubits, mode=mode, bond_dim=bond_dim)


def _bell_engine(mode: str = "exact") -> CircuitEngine:
    eng = _mock_engine(2, mode=mode)
    eng.apply("h", 0).apply("cx", [0, 1])
    return eng


# ---------------------------------------------------------------------------
#  1. CostEstimate dataclass structure
# ---------------------------------------------------------------------------

class TestCostEstimateFields:
    def test_total_cu_positive(self):
        est = _bell_engine().estimated_cost()
        assert isinstance(est, CostEstimate)
        assert est.total_cu > 0.0

    def test_breakdown_keys_present(self):
        est = _bell_engine().estimated_cost()
        for key in ("base", "depth_surcharge", "shots", "mode_multiplier"):
            assert key in est.breakdown, f"Missing breakdown key: {key}"

    def test_breakdown_sums_to_total(self):
        est = _bell_engine().estimated_cost(shots=1024)
        components = est.breakdown["base"] + est.breakdown["depth_surcharge"] + est.breakdown["shots"]
        assert abs(components - est.total_cu) < 1e-4

    def test_mode_multiplier_stored(self):
        exact_est = _bell_engine(mode="statevector").estimated_cost()
        assert exact_est.breakdown["mode_multiplier"] == 1.0

        tensor_est = _bell_engine(mode="mps").estimated_cost()
        assert tensor_est.breakdown["mode_multiplier"] == 2.0


# ---------------------------------------------------------------------------
#  2. Scaling behaviour
# ---------------------------------------------------------------------------

class TestCostScaling:
    def test_more_gates_costs_more(self):
        small = _mock_engine(5, mode="statevector")
        small.apply("h", 0).apply("cx", [0, 1])

        large = _mock_engine(5, mode="statevector")
        large.apply("h", 0).apply("cx", [0, 1])
        for i in range(10):
            large.apply("cx", [i % 5, (i + 1) % 5])

        assert large.estimated_cost().total_cu > small.estimated_cost().total_cu

    def test_more_qubits_costs_more_sv(self):
        # statevector cost grows as 2^N
        small = _mock_engine(5, mode="statevector")
        large = _mock_engine(15, mode="statevector")
        for eng in (small, large):
            eng.apply("h", 0).apply("cx", [0, 1])

        assert large.estimated_cost().total_cu > small.estimated_cost().total_cu

    def test_more_shots_costs_more(self):
        eng = _bell_engine()
        low = eng.estimated_cost(shots=100)
        high = eng.estimated_cost(shots=100_000)
        assert high.total_cu > low.total_cu

    def test_mps_costs_more_than_statevector(self):
        exact = _bell_engine(mode="statevector")
        tensor = _bell_engine(mode="mps")
        assert tensor.estimated_cost().total_cu > exact.estimated_cost().total_cu

    def test_local_mode_zero(self):
        # Local mode = no server billing
        est = _compute_cu_estimate(2, [{"gate": "h", "qubits": [0]}], "local", None, 1024)
        # depth_surcharge and base are multiplied by 0.0
        assert est.breakdown["mode_multiplier"] == 0.0
        assert est.breakdown["base"] == 0.0
        assert est.breakdown["depth_surcharge"] == 0.0


# ---------------------------------------------------------------------------
#  3. Ballpark calibration checks
# ---------------------------------------------------------------------------

class TestCalibration:
    def test_20q_depth20_statevector_ballpark(self):
        """
        Calibration point: 20-qubit depth-20 statevector ≈ 45 CU.
        Accept within factor-of-2 range [20, 90].
        """
        eng = _mock_engine(20, mode="statevector")
        # 20 entangling layers of ~10 cx gates each = 200 cx gates total
        for _ in range(20):
            for i in range(0, 20, 2):
                eng.apply("cx", [i, i + 1])
        est = eng.estimated_cost(shots=1024)
        assert 20.0 <= est.total_cu <= 90.0, (
            f"20q depth-20 statevector cost out of range: {est.total_cu:.2f} CU "
            f"(expected ~45 CU)"
        )

    def test_simple_circuit_low_cost(self):
        """A 2-qubit Bell state circuit should cost well under 5 CU."""
        est = _bell_engine().estimated_cost(shots=1024)
        assert est.total_cu < 5.0, (
            f"Simple 2-qubit circuit unexpectedly costly: {est.total_cu:.2f} CU"
        )

    def test_no_gates_returns_base_only(self):
        """Circuit with no gates still incurs base overhead."""
        eng = _mock_engine(5, mode="statevector")
        est = eng.estimated_cost(shots=0)
        # With 0 shots and no meaningful 2Q gates, only base (possibly multiplied) + surcharge for 1 gate
        assert est.total_cu > 0.0


# ---------------------------------------------------------------------------
#  4. Standalone CircuitClient.estimate_cost()
# ---------------------------------------------------------------------------

class TestStandaloneEstimateCost:
    def _client(self) -> CircuitClient:
        """Return a real CircuitClient (no HTTP calls needed for estimate_cost)."""
        return CircuitClient.__new__(CircuitClient)  # skip __init__

    def test_returns_cost_estimate(self):
        client = self._client()
        gates = [("h", 0), ("cx", [0, 1])]
        est = client.estimate_cost(gates, n_qubits=2, shots=1024)
        assert isinstance(est, CostEstimate)
        assert est.total_cu > 0.0

    def test_accepts_dict_format(self):
        client = self._client()
        gates = [{"gate": "h", "qubits": [0]}, {"gate": "cx", "qubits": [0, 1]}]
        est = client.estimate_cost(gates, n_qubits=2, shots=1024)
        assert isinstance(est, CostEstimate)

    def test_accepts_mixed_format(self):
        client = self._client()
        gates = [("h", 0), {"gate": "cx", "qubits": [0, 1]}]
        est = client.estimate_cost(gates, n_qubits=2, shots=1024)
        assert isinstance(est, CostEstimate)

    def test_mode_affects_cost(self):
        client = self._client()
        gates = [("h", 0), ("cx", [0, 1])]
        exact = client.estimate_cost(gates, n_qubits=2, shots=1024, mode="statevector")
        tensor = client.estimate_cost(gates, n_qubits=2, shots=1024, mode="mps")
        assert tensor.total_cu > exact.total_cu

    def test_bond_dim_affects_mps_cost(self):
        """Higher bond dimension raises MPS cost."""
        client = self._client()
        # Build a gate list with 2Q gates so surcharge is non-zero
        gates = [("cx", [i, i + 1]) for i in range(49)]  # 50q
        low_chi = client.estimate_cost(gates, n_qubits=50, shots=1024, mode="mps", bond_dim=8)
        hi_chi  = client.estimate_cost(gates, n_qubits=50, shots=1024, mode="mps", bond_dim=64)
        assert hi_chi.total_cu > low_chi.total_cu


# ---------------------------------------------------------------------------
#  5. compute_cu_estimate internal function
# ---------------------------------------------------------------------------

class TestComputeCuEstimate:
    def test_deterministic(self):
        gates = [{"gate": "cx", "qubits": [0, 1]}]
        a = _compute_cu_estimate(5, gates, "exact", None, 1024)
        b = _compute_cu_estimate(5, gates, "exact", None, 1024)
        assert a.total_cu == b.total_cu

    def test_empty_gates(self):
        # Empty circuit still has base cost
        est = _compute_cu_estimate(3, [], "exact", None, 1024)
        assert est.total_cu > 0.0

    def test_single_qubit_gates_ignored_from_surcharge(self):
        """Single-qubit gates don't contribute to 2Q gate count."""
        h_gates = [{"gate": "h", "qubits": [i]} for i in range(10)]
        # Many CX gates so their count clearly exceeds the max(n_2q, 1) floor
        cx_gates = [{"gate": "cx", "qubits": [i % 10, (i + 1) % 10]} for i in range(20)]
        only_h = _compute_cu_estimate(10, h_gates, "exact", None, 1024)
        h_cx   = _compute_cu_estimate(10, h_gates + cx_gates, "exact", None, 1024)
        assert h_cx.total_cu > only_h.total_cu
