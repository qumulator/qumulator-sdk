"""
Low-level HTTP client — shared by all resource clients.
"""
import time

import httpx

from qumulator.models import JobStatus

_DEFAULT_POLL_INTERVAL = 2.0   # seconds between status polls
_DEFAULT_TIMEOUT = 600.0       # max seconds to wait for a job to complete


class QumulatorHTTPError(Exception):
    """Raised when the API returns a non-2xx response."""
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


class _BaseClient:
    """Shared sync HTTP client with submit-and-poll helper."""

    def __init__(self, api_url: str, api_key: str):
        self._api_url = api_url.rstrip("/")
        self._headers = {"X-API-Key": api_key, "Content-Type": "application/json"}

    def _post(self, path: str, body: dict) -> dict:
        with httpx.Client(base_url=self._api_url, headers=self._headers,
                          timeout=30.0) as client:
            resp = client.post(path, json=body)
            self._raise_for_status(resp)
            return resp.json()

    def _get(self, path: str) -> dict:
        with httpx.Client(base_url=self._api_url, headers=self._headers,
                          timeout=30.0) as client:
            resp = client.get(path)
            self._raise_for_status(resp)
            return resp.json()

    @staticmethod
    def _raise_for_status(resp: httpx.Response) -> None:
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise QumulatorHTTPError(resp.status_code, detail)

    def _submit_and_wait(
        self,
        engine_path: str,
        body: dict,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> JobStatus:
        """Submit a job and block until it is completed or failed."""
        submit_data = self._post(f"/jobs{engine_path}", body)
        job_id = submit_data["job_id"]

        deadline = time.monotonic() + timeout
        while True:
            status = JobStatus(**self._get(f"/jobs/{job_id}"))
            if status.is_done:
                return status
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Job {job_id} did not complete within {timeout:.0f}s"
                )
            time.sleep(poll_interval)
