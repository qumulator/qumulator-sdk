"""
Unit tests for DMRGClient.

No live API required — all HTTP calls are intercepted by respx.
Verifies:
  1. Correct JSON payload serialisation (h1e, h2e, n_elec, e_nuc, d_max, n_sweeps, tol).
  2. Response correctly deserialised into DMRGEnergyResult.
  3. The ``converged`` flag is correctly parsed (True / False).
  4. HTTP errors surface as QumulatorHTTPError.
Run with:  pytest sdk/tests/test_dmrg.py -v
"""
from __future__ import annotations

import json

import httpx
import pytest
import respx

from qumulator import QumulatorClient
from qumulator._http import QumulatorHTTPError
from qumulator.models import DMRGEnergyResult

# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------

API_URL = "https://test.qumulator.com"
API_KEY  = "test-key"

# Minimal H2 CAS(2,2) STO-3G integrals
_H1E = [[-1.12396678, 0.0], [0.0, -0.59597322]]
_H2E = [
    [[[0.67409512, 0.0], [0.0, 0.18129475]],
     [[0.0, 0.18129475], [0.0, 0.0]]],
    [[[0.0, 0.0], [0.18129475, 0.0]],
     [[0.18129475, 0.0], [0.0, 0.66341843]]],
]
_N_ELEC = [1, 1]
_E_NUC  = 0.71375399

_MOCK_CONVERGED = {
    "energy":       -1.13728383,
    "converged":    True,
    "n_sweeps_run": 5,
    "d_max_used":   64,
    "n_orb":        2,
    "n_so":         4,
    "wall_time_s":  0.42,
}

_MOCK_NOT_CONVERGED = {
    "energy":       -1.13000000,
    "converged":    False,
    "n_sweeps_run": 8,
    "d_max_used":   4,
    "n_orb":        2,
    "n_so":         4,
    "wall_time_s":  0.18,
}


@pytest.fixture
def client() -> QumulatorClient:
    return QumulatorClient(api_url=API_URL, api_key=API_KEY)


# ---------------------------------------------------------------------------
#  1. Payload serialisation
# ---------------------------------------------------------------------------


class TestDMRGPayload:
    @respx.mock
    def test_required_fields_sent(self, client: QumulatorClient) -> None:
        """h1e, h2e, n_elec, e_nuc, d_max, n_sweeps, tol all present in body."""
        route = respx.post(f"{API_URL}/molecular/dmrg").mock(
            return_value=httpx.Response(200, json=_MOCK_CONVERGED)
        )
        client.dmrg.energy(
            h1e=_H1E, h2e=_H2E, n_elec=_N_ELEC, e_nuc=_E_NUC,
            d_max=64, n_sweeps=8, tol=1e-10,
        )
        assert route.called
        body = json.loads(route.calls[0].request.content)
        assert body["h1e"]      == _H1E
        assert body["h2e"]      == _H2E
        assert body["n_elec"]   == _N_ELEC
        assert body["e_nuc"]    == pytest.approx(_E_NUC)
        assert body["d_max"]    == 64
        assert body["n_sweeps"] == 8
        assert body["tol"]      == pytest.approx(1e-10)

    @respx.mock
    def test_default_d_max(self, client: QumulatorClient) -> None:
        """Default d_max is 64."""
        route = respx.post(f"{API_URL}/molecular/dmrg").mock(
            return_value=httpx.Response(200, json=_MOCK_CONVERGED)
        )
        client.dmrg.energy(h1e=_H1E, h2e=_H2E, n_elec=_N_ELEC)
        body = json.loads(route.calls[0].request.content)
        assert body["d_max"] == 64

    @respx.mock
    def test_default_n_sweeps(self, client: QumulatorClient) -> None:
        """Default n_sweeps is 8."""
        route = respx.post(f"{API_URL}/molecular/dmrg").mock(
            return_value=httpx.Response(200, json=_MOCK_CONVERGED)
        )
        client.dmrg.energy(h1e=_H1E, h2e=_H2E, n_elec=_N_ELEC)
        body = json.loads(route.calls[0].request.content)
        assert body["n_sweeps"] == 8

    @respx.mock
    def test_custom_d_max_forwarded(self, client: QumulatorClient) -> None:
        """Custom d_max value is forwarded."""
        route = respx.post(f"{API_URL}/molecular/dmrg").mock(
            return_value=httpx.Response(200, json=_MOCK_CONVERGED)
        )
        client.dmrg.energy(h1e=_H1E, h2e=_H2E, n_elec=_N_ELEC, d_max=128)
        body = json.loads(route.calls[0].request.content)
        assert body["d_max"] == 128

    @respx.mock
    def test_e_nuc_default_zero(self, client: QumulatorClient) -> None:
        """Omitting e_nuc sends 0.0."""
        route = respx.post(f"{API_URL}/molecular/dmrg").mock(
            return_value=httpx.Response(200, json=_MOCK_CONVERGED)
        )
        client.dmrg.energy(h1e=_H1E, h2e=_H2E, n_elec=_N_ELEC)
        body = json.loads(route.calls[0].request.content)
        assert body["e_nuc"] == 0.0


# ---------------------------------------------------------------------------
#  2. Response deserialisation
# ---------------------------------------------------------------------------


class TestDMRGResponse:
    @respx.mock
    def test_returns_dmrg_energy_result(self, client: QumulatorClient) -> None:
        respx.post(f"{API_URL}/molecular/dmrg").mock(
            return_value=httpx.Response(200, json=_MOCK_CONVERGED)
        )
        result = client.dmrg.energy(h1e=_H1E, h2e=_H2E, n_elec=_N_ELEC)
        assert isinstance(result, DMRGEnergyResult)

    @respx.mock
    def test_energy_value(self, client: QumulatorClient) -> None:
        respx.post(f"{API_URL}/molecular/dmrg").mock(
            return_value=httpx.Response(200, json=_MOCK_CONVERGED)
        )
        result = client.dmrg.energy(h1e=_H1E, h2e=_H2E, n_elec=_N_ELEC)
        assert result.energy == pytest.approx(-1.13728383)

    @respx.mock
    def test_converged_true(self, client: QumulatorClient) -> None:
        respx.post(f"{API_URL}/molecular/dmrg").mock(
            return_value=httpx.Response(200, json=_MOCK_CONVERGED)
        )
        result = client.dmrg.energy(h1e=_H1E, h2e=_H2E, n_elec=_N_ELEC)
        assert result.converged is True

    @respx.mock
    def test_converged_false(self, client: QumulatorClient) -> None:
        """converged=False is correctly deserialised (d_max too small)."""
        respx.post(f"{API_URL}/molecular/dmrg").mock(
            return_value=httpx.Response(200, json=_MOCK_NOT_CONVERGED)
        )
        result = client.dmrg.energy(h1e=_H1E, h2e=_H2E, n_elec=_N_ELEC, d_max=4)
        assert result.converged is False

    @respx.mock
    def test_n_so_is_twice_n_orb(self, client: QumulatorClient) -> None:
        respx.post(f"{API_URL}/molecular/dmrg").mock(
            return_value=httpx.Response(200, json=_MOCK_CONVERGED)
        )
        result = client.dmrg.energy(h1e=_H1E, h2e=_H2E, n_elec=_N_ELEC)
        assert result.n_so == 2 * result.n_orb

    @respx.mock
    def test_d_max_used_matches_request(self, client: QumulatorClient) -> None:
        respx.post(f"{API_URL}/molecular/dmrg").mock(
            return_value=httpx.Response(200, json=_MOCK_CONVERGED)
        )
        result = client.dmrg.energy(h1e=_H1E, h2e=_H2E, n_elec=_N_ELEC, d_max=64)
        assert result.d_max_used == 64

    @respx.mock
    def test_n_sweeps_run_non_negative(self, client: QumulatorClient) -> None:
        respx.post(f"{API_URL}/molecular/dmrg").mock(
            return_value=httpx.Response(200, json=_MOCK_CONVERGED)
        )
        result = client.dmrg.energy(h1e=_H1E, h2e=_H2E, n_elec=_N_ELEC)
        assert result.n_sweeps_run >= 0

    @respx.mock
    def test_wall_time_non_negative(self, client: QumulatorClient) -> None:
        respx.post(f"{API_URL}/molecular/dmrg").mock(
            return_value=httpx.Response(200, json=_MOCK_CONVERGED)
        )
        result = client.dmrg.energy(h1e=_H1E, h2e=_H2E, n_elec=_N_ELEC)
        assert result.wall_time_s >= 0.0


# ---------------------------------------------------------------------------
#  3. Error handling
# ---------------------------------------------------------------------------


class TestDMRGErrors:
    @respx.mock
    def test_422_raises_http_error(self, client: QumulatorClient) -> None:
        """n_orb > 30 should be rejected by the backend with 422."""
        respx.post(f"{API_URL}/molecular/dmrg").mock(
            return_value=httpx.Response(422, json={"detail": "n_orb exceeds limit of 30"})
        )
        with pytest.raises(QumulatorHTTPError) as exc_info:
            client.dmrg.energy(h1e=_H1E, h2e=_H2E, n_elec=_N_ELEC)
        assert exc_info.value.status_code == 422

    @respx.mock
    def test_500_raises_http_error(self, client: QumulatorClient) -> None:
        respx.post(f"{API_URL}/molecular/dmrg").mock(
            return_value=httpx.Response(500, json={"detail": "internal error"})
        )
        with pytest.raises(QumulatorHTTPError):
            client.dmrg.energy(h1e=_H1E, h2e=_H2E, n_elec=_N_ELEC)

    @respx.mock
    def test_401_raises_http_error(self, client: QumulatorClient) -> None:
        respx.post(f"{API_URL}/molecular/dmrg").mock(
            return_value=httpx.Response(401, json={"detail": "unauthorized"})
        )
        with pytest.raises(QumulatorHTTPError) as exc_info:
            client.dmrg.energy(h1e=_H1E, h2e=_H2E, n_elec=_N_ELEC)
        assert exc_info.value.status_code == 401

    @respx.mock
    def test_429_rate_limit_raises_http_error(self, client: QumulatorClient) -> None:
        """Rate limiting (HTTP 429) surfaces as QumulatorHTTPError."""
        respx.post(f"{API_URL}/molecular/dmrg").mock(
            return_value=httpx.Response(429, json={"detail": "rate limit exceeded"})
        )
        with pytest.raises(QumulatorHTTPError) as exc_info:
            client.dmrg.energy(h1e=_H1E, h2e=_H2E, n_elec=_N_ELEC)
        assert exc_info.value.status_code == 429
