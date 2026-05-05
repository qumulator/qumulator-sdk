# Authentication

Every request to the API (except `POST /keys`) must include your API key in the
`X-API-Key` header.

```bash
curl https://api.qumulator.com/circuits \
  -H "X-API-Key: qum_xxxxxxxx_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

The Python SDK handles this automatically once you pass `api_key` to `QumulatorClient`.

!!! warning
    Keys are **not recoverable** after creation. If you lose a key, generate a new one
    via `POST /keys`. There is no limit on how many keys you can create.

---

## Base URL & Versioning

| Property | Value |
|---|---|
| Base URL | `https://api.qumulator.com` |
| Protocol | HTTPS only |
| Request format | JSON — `Content-Type: application/json` |
| Interactive docs | [api.qumulator.com/docs](https://api.qumulator.com/docs) (Swagger UI) |

The API is currently unversioned (no `/v1/` prefix). Breaking changes will be announced
in advance.
