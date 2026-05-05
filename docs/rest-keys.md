# REST API — API Keys

## Create a key

**`POST /keys`** — No authentication required.

| Field | Type | Description |
|---|---|---|
| `name` | `string` | Human-readable label |

```json
// Response 201
{
  "key":        "qum_xxxxxxxx_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
  "name":       "my-key",
  "created_at": "2026-04-20T12:00:00Z"
}
```

```bash
# cURL example
curl -s -X POST https://api.qumulator.com/keys \
  -H "Content-Type: application/json" \
  -d '{"name": "my-key"}'
```

```powershell
# PowerShell example
Invoke-WebRequest `
  -Uri "https://api.qumulator.com/keys" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"name":"my-key"}' `
  -UseBasicParsing
```
