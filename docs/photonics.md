# Photonic Amplitudes

Compute the hafnian, permanent, or Gaussian boson sampling amplitudes from a symmetric
coupling matrix. The result is a complex number computed to configurable precision.

```python
import numpy as np

n = 8
A = np.random.randn(n, n)
A = (A + A.T) / 2
np.fill_diagonal(A, 0)

result = client.hafnian.run(matrix_real=A.tolist())

print(result.value)    # complex: haf_real + 1j*haf_imag
print(result.elapsed)  # server compute time in seconds
```

---

## Complex matrices and precision

```python
# Complex coupling matrix, 50 decimal digits of precision
result = client.hafnian.run(
    matrix_real=A_real.tolist(),
    matrix_imag=A_imag.tolist(),
    mp_dps=50,   # default: 34
)
```

---

## HafnianResult fields

| Field | Type | Description |
|---|---|---|
| `haf_real` | `float` | Real part of the hafnian |
| `haf_imag` | `float` | Imaginary part |
| `value` | `complex` | Property: `haf_real + 1j*haf_imag` |
| `elapsed` | `float` | Server-side compute time (seconds) |
| `max_S` | `float` | Maximum coupling entropy |
| `n_edges` | `int` | Number of non-trivial coupling pairs |
| `est_matchings` | `float` | Estimated number of perfect matchings |
