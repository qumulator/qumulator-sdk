# Rate Limits

| Limit | Value |
|---|---|
| Circuit simulations per key per minute | 1 |
| Circuit simulations per key per day | 100 |
| Window type | Rolling 60-second window |
| HTTP status when exceeded | 429 with `Retry-After` header |

The `Retry-After` header tells you exactly how many seconds to wait. The SDK surfaces
this as `QumulatorHTTPError(status_code=429)`.

```python
from qumulator import QumulatorHTTPError
import time

try:
    result = eng.run(shots=1024)
except QumulatorHTTPError as e:
    if e.status_code == 429:
        retry_after = int(e.response.headers.get("Retry-After", 10))
        time.sleep(retry_after)
        result = eng.run(shots=1024)   # retry once
```
