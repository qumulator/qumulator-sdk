"""
Qumulator SDK — client unit tests (no live API required).

All tests use respx to mock HTTP calls and verify that:
  1. The correct API payloads are sent.
  2. Responses are correctly deserialised into SDK types.
"""
import math
from typing import Any, Dict

import httpx
import numpy as np
import pytest
import respx

from qumulator import QumulatorClient, CircuitResult
from qumulator.circuit import CircuitClient, CircuitEngine, _normalise_gate_list, _gate_to_instruction


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------

API_URL = "https://test.qumulator.com"
API_KEY = "test-key"

_SUBMIT_RESPONSE = {"job_id": "test-job-01", "status": "queued"}
_RESULT_RESPONSE = {
    "job_id": "test-job-01",
    "status": "completed",
    "created_at": "2026-04-18T10:00:00",
    "started_at": "2026-04-18T10:00:01",
    "completed_at": "2026-04-18T10:00:02",
    "result": {
        "counts": {"00": 512, "11": 512},
        "n_qubits": 2,
        "shots": 1024,
    },
    "error": None,
}

_BELL_RESPONSE = _RESULT_RESPONSE["result"]


def _mock_circuit_run(result_override: dict | None = None) -> None:
    """Set up respx mocks for the submit-and-poll circuit flow."""
    result = dict(_RESULT_RESPONSE)
    if result_override:
        result = dict(result)
        result["result"] = {**result["result"], **result_override}
    respx.post(f"{API_URL}/circuits").mock(
        return_value=httpx.Response(202, json=_SUBMIT_RESPONSE)
    )
    respx.get(f"{API_URL}/circuits/test-job-01").mock(
        return_value=httpx.Response(200, json=result)
    )


@pytest.fixture
def client() -> CircuitClient:
    return CircuitClient(api_url=API_URL, api_key=API_KEY)


@pytest.fixture
def qclient() -> QumulatorClient:
    return QumulatorClient(api_url=API_URL, api_key=API_KEY)


# ---------------------------------------------------------------------------
#  CircuitResult
# ---------------------------------------------------------------------------


class TestCircuitResult:
    def test_most_probable_single(self):
        r = CircuitResult(counts={"00": 1024}, n_qubits=2, shots=1024)
        assert r.most_probable == "00"

    def test_most_probable_multiple(self):
        r = CircuitResult(
            counts={"00": 800, "11": 150, "01": 50, "10": 24},
            n_qubits=2,
            shots=1024,
        )
        assert r.most_probable == "00"

    def test_optional_fields_default_none(self):
        r = CircuitResult(counts={}, n_qubits=1, shots=0)
        assert r.statevector is None
        assert r.probabilities is None
        assert r.entropy_map is None


# ---------------------------------------------------------------------------
#  CircuitEngine — gate accumulation
# ---------------------------------------------------------------------------


class TestCircuitEngine:
    def test_apply_returns_self_for_chaining(self, client):
        eng = client.engine(n_qubits=2)
        assert eng.apply("h", 0).apply("cx", [0, 1]) is eng

    def test_single_qubit_int(self, client):
        eng = client.engine(n_qubits=2)
        eng.apply("h", 0)
        assert eng._gates == [{"gate": "h", "qubits": [0]}]

    def test_two_qubit_list(self, client):
        eng = client.engine(n_qubits=2)
        eng.apply("cx", [0, 1])
        assert eng._gates == [{"gate": "cx", "qubits": [0, 1]}]

    def test_parametric_gate(self, client):
        eng = client.engine(n_qubits=1)
        eng.apply("rx", 0, [math.pi / 2])
        gate = eng._gates[0]
        assert gate["gate"] == "rx"
        assert math.isclose(gate["params"][0], math.pi / 2)

    def test_reset_clears_all_gates(self, client):
        eng = client.engine(n_qubits=2)
        eng.apply("h", 0).apply("cx", [0, 1]).reset()
        assert eng._gates == []

    def test_unitary_matrix_serialised(self, client):
        eng = client.engine(n_qubits=1)
        H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
        eng.apply("unitary", [0], [H])
        params = eng._gates[0]["params"][0]
        # Should be a list of [real, imag] pairs (4 entries for 2x2 matrix)
        assert len(params) == 4
        assert all(len(p) == 2 for p in params)

    def test_evolve_hamiltonian_gate_entry(self, client):
        eng = client.engine(n_qubits=2, mode="hamiltonian")
        eng.evolve_hamiltonian([(0.5, "ZZ"), (-0.3, "XI")], t=1.0)
        g = eng._gates[0]
        assert g["gate"] == "hamiltonian_evolution"
        assert g["t"] == 1.0
        assert len(g["pauli_terms"]) == 2


# ---------------------------------------------------------------------------
#  CircuitClient — HTTP payload and response parsing
# ---------------------------------------------------------------------------


_BELL_RESPONSE: Dict[str, Any] = {
    "counts": {"00": 512, "11": 512},
    "n_qubits": 2,
    "shots": 1024,
}


@respx.mock
class TestCircuitClientRun:
    def test_run_returns_circuit_result(self, client):
        _mock_circuit_run()
        result = client.run(
            gates=[{"gate": "h", "qubits": [0]}, {"gate": "cx", "qubits": [0, 1]}],
            n_qubits=2,
            shots=1024,
        )
        assert isinstance(result, CircuitResult)
        assert result.counts == {"00": 512, "11": 512}
        assert result.n_qubits == 2
        assert result.shots == 1024

    def test_run_sends_n_qubits_and_mode(self, client):
        captured: Dict = {}

        def _capture(request: httpx.Request, **_):
            import json
            captured.update(json.loads(request.content))
            return httpx.Response(202, json=_SUBMIT_RESPONSE)

        respx.post(f"{API_URL}/circuits").mock(side_effect=_capture)
        respx.get(f"{API_URL}/circuits/test-job-01").mock(
            return_value=httpx.Response(200, json=_RESULT_RESPONSE)
        )
        client.run(
            gates=[("h", 0)],
            n_qubits=2,
            mode="exact",
            shots=512,
        )
        assert captured["n_qubits"] == 2
        assert captured["mode"] == "statevector"   # alias resolved
        assert captured["shots"] == 512

    def test_run_omits_false_flags(self, client):
        captured: Dict = {}

        def _capture(request: httpx.Request, **_):
            import json
            captured.update(json.loads(request.content))
            return httpx.Response(202, json=_SUBMIT_RESPONSE)

        respx.post(f"{API_URL}/circuits").mock(side_effect=_capture)
        respx.get(f"{API_URL}/circuits/test-job-01").mock(
            return_value=httpx.Response(200, json=_RESULT_RESPONSE)
        )
        client.run(gates=[], n_qubits=1, return_statevector=False)
        assert "return_statevector" not in captured

    def test_statevector_deserialised(self, client):
        sv_override = {
            "statevector": [[0.7071, 0.0], [0.0, 0.0], [0.0, 0.0], [0.7071, 0.0]],
        }
        _mock_circuit_run(sv_override)
        result = client.run(gates=[], n_qubits=2, return_statevector=True)
        assert result.statevector is not None
        assert result.statevector.dtype == complex
        assert result.statevector.shape == (4,)
        assert math.isclose(abs(result.statevector[0]), 0.7071, abs_tol=1e-4)

    def test_entropy_map_deserialised(self, client):
        _mock_circuit_run({"entropy_map": [1.0, 1.0]})
        result = client.run(gates=[], n_qubits=2, return_entropy_map=True)
        assert result.entropy_map == [1.0, 1.0]

    def test_engine_run_delegates_to_client(self, client):
        _mock_circuit_run()
        eng = client.engine(n_qubits=2)
        eng.apply("h", 0).apply("cx", [0, 1])
        result = eng.run(shots=1024)
        assert result.counts == {"00": 512, "11": 512}

    def test_engine_sample_returns_counts(self, client):
        _mock_circuit_run()
        counts = client.engine(n_qubits=2).apply("h", 0).apply("cx", [0, 1]).sample()
        assert set(counts) == {"00", "11"}


# ---------------------------------------------------------------------------
#  QumulatorClient — top-level bundled client
# ---------------------------------------------------------------------------


class TestQumulatorClient:
    def test_attributes_exist(self, qclient):
        assert hasattr(qclient, "circuit")
        assert hasattr(qclient, "homo")
        assert hasattr(qclient, "klt")
        assert hasattr(qclient, "hafnian")
        assert hasattr(qclient, "notebook")

    def test_circuit_is_circuit_client(self, qclient):
        from qumulator.circuit import CircuitClient
        assert isinstance(qclient.circuit, CircuitClient)

    def test_notebook_client_exists(self, qclient):
        from qumulator.resources import NotebookClient
        assert isinstance(qclient.notebook, NotebookClient)


# ---------------------------------------------------------------------------
#  Gate-list normalisation
# ---------------------------------------------------------------------------


class TestNormaliseGateList:
    def test_dict_passthrough(self):
        gates = [{"gate": "h", "qubits": [0]}]
        assert _normalise_gate_list(gates) == gates

    def test_tuple_two_element(self):
        result = _normalise_gate_list([("h", 0)])
        assert result == [{"gate": "h", "qubits": [0]}]

    def test_tuple_list_qubits(self):
        result = _normalise_gate_list([("cx", [0, 1])])
        assert result == [{"gate": "cx", "qubits": [0, 1]}]

    def test_tuple_with_params(self):
        result = _normalise_gate_list([("rx", 0, [math.pi])])
        assert result[0]["gate"] == "rx"
        assert math.isclose(result[0]["params"][0], math.pi)


# ---------------------------------------------------------------------------
#  _gate_to_instruction — unitary matrix conversion
# ---------------------------------------------------------------------------


class TestGateToInstruction:
    def test_non_unitary_passthrough(self):
        g = {"gate": "h", "qubits": [0]}
        assert _gate_to_instruction(g) is g

    def test_unitary_converts_matrix(self):
        """2x2 Hadamard in SDK params format → matrix_real / matrix_imag."""
        s = 1 / math.sqrt(2)
        g = {
            "gate": "unitary",
            "qubits": [0],
            "params": [[[s, 0.0], [s, 0.0], [s, 0.0], [-s, 0.0]]],
        }
        out = _gate_to_instruction(g)
        assert "params" not in out
        assert out["matrix_real"] == [[s, s], [s, -s]]
        assert out["matrix_imag"] == [[0.0, 0.0], [0.0, 0.0]]
