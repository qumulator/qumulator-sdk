"""
Unit tests for MolecularClient (MPS/MPO engine).

No live API required — all HTTP calls are intercepted by respx.
Verifies:
  1. Correct JSON payload serialisation (h1e, h2e, n_elec, e_nuc, circuit, coup_thr).
  2. Response correctly deserialised into MolecularEnergyResult.
  3. Optional fields (circuit, zz_correlators) handled properly.
  4. HTTP errors surface as QumulatorHTTPError.
Run with:  pytest sdk/tests/test_molecular_mps.py -v
"""
from __future__ import annotations

import json

import httpx
import numpy as np
import pytest
import respx

from qumulator import QumulatorClient
from qumulator._http import QumulatorHTTPError
from qumulator.models import MolecularEnergyResult

# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------

API_URL = "https://test.qumulator.com"
API_KEY  = "test-key"

# Minimal H2 CAS(2,2) STO-3G integrals (analytically derived)
_H1E = [[-1.12396678, 0.0], [0.0, -0.59597322]]
_H2E = [
    [[[0.67409512, 0.0], [0.0, 0.18129475]],
     [[0.0, 0.18129475], [0.0, 0.0]]],
    [[[0.0, 0.0], [0.18129475, 0.0]],
     [[0.18129475, 0.0], [0.0, 0.66341843]]],
]
_N_ELEC  = [1, 1]
_E_NUC   = 0.71375399

_MOCK_RESPONSE = {
    "energy":        -1.13728383,
    "n_qubits":      4,
    "n_orb":         2,
    "n_components":  1,
    "zz_correlators": None,
}


@pytest.fixture
def client() -> QumulatorClient:
    return QumulatorClient(api_url=API_URL, api_key=API_KEY)


# ---------------------------------------------------------------------------
#  1. Payload serialisation
# ---------------------------------------------------------------------------


class TestMolecularPayload:
    @respx.mock
    def test_required_fields_sent(self, client: QumulatorClient) -> None:
        """h1e, h2e, n_elec, e_nuc, empty circuit, default coup_thr are all present."""
        route = respx.post(f"{API_URL}/molecular/energy").mock(
            return_value=httpx.Response(200, json=_MOCK_RESPONSE)
        )
        client.molecular.energy(
            h1e=_H1E, h2e=_H2E, n_elec=_N_ELEC, e_nuc=_E_NUC
        )
        assert route.called
        body = json.loads(route.calls[0].request.content)
        assert body["h1e"]     == _H1E
        assert body["h2e"]     == _H2E
        assert body["n_elec"]  == _N_ELEC
        assert body["e_nuc"]   == pytest.approx(_E_NUC)
        assert body["circuit"] == []
        assert "coup_thr" in body

    @respx.mock
    def test_e_nuc_default_zero(self, client: QumulatorClient) -> None:
        """Omitting e_nuc sends 0.0."""
        route = respx.post(f"{API_URL}/molecular/energy").mock(
            return_value=httpx.Response(200, json=_MOCK_RESPONSE)
        )
        client.molecular.energy(h1e=_H1E, h2e=_H2E, n_elec=_N_ELEC)
        body = json.loads(route.calls[0].request.content)
        assert body["e_nuc"] == 0.0

    @respx.mock
    def test_circuit_forwarded(self, client: QumulatorClient) -> None:
        """An explicit Givens circuit is forwarded in the payload."""
        circuit = [{"qi": 0, "qj": 2, "theta": 0.15}]
        route = respx.post(f"{API_URL}/molecular/energy").mock(
            return_value=httpx.Response(200, json=_MOCK_RESPONSE)
        )
        client.molecular.energy(
            h1e=_H1E, h2e=_H2E, n_elec=_N_ELEC, e_nuc=_E_NUC,
            circuit=circuit,
        )
        body = json.loads(route.calls[0].request.content)
        assert body["circuit"] == circuit

    @respx.mock
    def test_coup_thr_custom(self, client: QumulatorClient) -> None:
        """Custom coup_thr is forwarded."""
        route = respx.post(f"{API_URL}/molecular/energy").mock(
            return_value=httpx.Response(200, json=_MOCK_RESPONSE)
        )
        client.molecular.energy(
            h1e=_H1E, h2e=_H2E, n_elec=_N_ELEC, coup_thr=1e-4
        )
        body = json.loads(route.calls[0].request.content)
        assert body["coup_thr"] == pytest.approx(1e-4)


# ---------------------------------------------------------------------------
#  2. Response deserialisation
# ---------------------------------------------------------------------------


class TestMolecularResponse:
    @respx.mock
    def test_returns_molecular_energy_result(self, client: QumulatorClient) -> None:
        respx.post(f"{API_URL}/molecular/energy").mock(
            return_value=httpx.Response(200, json=_MOCK_RESPONSE)
        )
        result = client.molecular.energy(h1e=_H1E, h2e=_H2E, n_elec=_N_ELEC)
        assert isinstance(result, MolecularEnergyResult)

    @respx.mock
    def test_energy_value(self, client: QumulatorClient) -> None:
        respx.post(f"{API_URL}/molecular/energy").mock(
            return_value=httpx.Response(200, json=_MOCK_RESPONSE)
        )
        result = client.molecular.energy(h1e=_H1E, h2e=_H2E, n_elec=_N_ELEC)
        assert result.energy == pytest.approx(-1.13728383)

    @respx.mock
    def test_n_qubits_is_twice_n_orb(self, client: QumulatorClient) -> None:
        respx.post(f"{API_URL}/molecular/energy").mock(
            return_value=httpx.Response(200, json=_MOCK_RESPONSE)
        )
        result = client.molecular.energy(h1e=_H1E, h2e=_H2E, n_elec=_N_ELEC)
        assert result.n_qubits == 2 * result.n_orb

    @respx.mock
    def test_zz_correlators_none(self, client: QumulatorClient) -> None:
        """zz_correlators is optional and defaults to None."""
        respx.post(f"{API_URL}/molecular/energy").mock(
            return_value=httpx.Response(200, json=_MOCK_RESPONSE)
        )
        result = client.molecular.energy(h1e=_H1E, h2e=_H2E, n_elec=_N_ELEC)
        assert result.zz_correlators is None

    @respx.mock
    def test_zz_correlators_present(self, client: QumulatorClient) -> None:
        """zz_correlators is parsed when returned by the server."""
        mock = dict(_MOCK_RESPONSE)
        mock["zz_correlators"] = [[1.0, -0.5], [-0.5, 1.0]]
        mock["n_qubits"] = 2
        respx.post(f"{API_URL}/molecular/energy").mock(
            return_value=httpx.Response(200, json=mock)
        )
        result = client.molecular.energy(h1e=_H1E, h2e=_H2E, n_elec=_N_ELEC)
        assert result.zz_correlators is not None
        assert len(result.zz_correlators) == 2

    @respx.mock
    def test_n_components_positive(self, client: QumulatorClient) -> None:
        respx.post(f"{API_URL}/molecular/energy").mock(
            return_value=httpx.Response(200, json=_MOCK_RESPONSE)
        )
        result = client.molecular.energy(h1e=_H1E, h2e=_H2E, n_elec=_N_ELEC)
        assert result.n_components >= 1


# ---------------------------------------------------------------------------
#  3. Error handling
# ---------------------------------------------------------------------------


class TestMolecularErrors:
    @respx.mock
    def test_422_raises_http_error(self, client: QumulatorClient) -> None:
        respx.post(f"{API_URL}/molecular/energy").mock(
            return_value=httpx.Response(422, json={"detail": "n_orb exceeds limit"})
        )
        with pytest.raises(QumulatorHTTPError) as exc_info:
            client.molecular.energy(h1e=_H1E, h2e=_H2E, n_elec=_N_ELEC)
        assert exc_info.value.status_code == 422

    @respx.mock
    def test_500_raises_http_error(self, client: QumulatorClient) -> None:
        respx.post(f"{API_URL}/molecular/energy").mock(
            return_value=httpx.Response(500, json={"detail": "internal error"})
        )
        with pytest.raises(QumulatorHTTPError):
            client.molecular.energy(h1e=_H1E, h2e=_H2E, n_elec=_N_ELEC)

    @respx.mock
    def test_401_raises_http_error(self, client: QumulatorClient) -> None:
        respx.post(f"{API_URL}/molecular/energy").mock(
            return_value=httpx.Response(401, json={"detail": "unauthorized"})
        )
        with pytest.raises(QumulatorHTTPError) as exc_info:
            client.molecular.energy(h1e=_H1E, h2e=_H2E, n_elec=_N_ELEC)
        assert exc_info.value.status_code == 401
