# Notebooks

## Remote notebook execution

Submit a Jupyter notebook (`.ipynb`) for remote execution. The notebook runs in an
isolated sandbox environment with `qumulator-sdk` and the full scientific Python stack
pre-installed.

```python
# Submit and wait for completion
with open("my_experiment.ipynb", "rb") as f:
    nb_bytes = f.read()

status = client.notebook.run(nb_bytes)
print(status.status)   # "completed"
# status.result contains the executed notebook with outputs
```

### Async: submit then poll

```python
import time

with open("my_experiment.ipynb", "rb") as f:
    job_id = client.notebook.submit(f.read())

print("Job submitted:", job_id)

while True:
    status = client.notebook.status(job_id)
    if status.is_done:
        break
    time.sleep(3)

if status.ok:
    executed_notebook = status.result
else:
    print("Failed:", status.error)
```

!!! info
    **Pre-installed packages:** `numpy`, `scipy`, `matplotlib`, `pandas`, `qiskit`,
    `cirq`, and `qumulator-sdk`. Additional packages can be installed at the top of your
    notebook with `!pip install ...`.

---

## Demo notebooks

Open any notebook directly in Google Colab — no install required, just add your API key
in the second cell.

| Notebook | Description |
|---|---|
| [Willow RCS](https://colab.research.google.com/github/qumulator/qumulator-sdk/blob/main/notebooks/willow_rcs_benchmark.ipynb) | 105-qubit exact simulation on a Willow-layout SYC grid |
| [Holographic wormhole](https://colab.research.google.com/github/qumulator/qumulator-sdk/blob/main/notebooks/wormhole.ipynb) | Traversable wormhole protocol — matches Google Sycamore 2022 |
| [Anyon braiding](https://colab.research.google.com/github/qumulator/qumulator-sdk/blob/main/notebooks/anyon_braiding.ipynb) | Fibonacci anyons, non-Abelian braiding — matches Microsoft topological target |
| [Discrete time crystal](https://colab.research.google.com/github/qumulator/qumulator-sdk/blob/main/notebooks/time_crystal.ipynb) | MBL Floquet dynamics — matches Google Sycamore 2021 |
| [QUBO optimisation](https://colab.research.google.com/github/qumulator/qumulator-sdk/blob/main/notebooks/qubo.ipynb) | 100-variable dense combinatorial optimisation |
| [Cluster engine](https://colab.research.google.com/github/qumulator/qumulator-sdk/blob/main/notebooks/klt_cluster_demo.ipynb) | Exact simulation without 2ᴺ state vector — `mode="cluster"` |
| [Green's function engine](https://colab.research.google.com/github/qumulator/qumulator-sdk/blob/main/notebooks/klt_greens_demo.ipynb) | Free-fermion circuits, O(N²) memory, entropy map — `mode="greens"` |
| [Collapse & revival / QKZM](https://colab.research.google.com/github/qumulator/qumulator-sdk/blob/main/notebooks/tebd_quench_demo.ipynb) | TEBD Hamiltonian time evolution, Kibble-Zurek scaling — `client.evolve` |
| [Prime factorization](https://colab.research.google.com/github/qumulator/qumulator-sdk/blob/main/notebooks/prime_factoring.ipynb) | Factors N=35 via quantum-inspired energy landscape — pure scipy, no API call required |
| [CHSH Bell inequality](https://colab.research.google.com/github/qumulator/qumulator-sdk/blob/main/notebooks/chsh_bell.ipynb) | Bell state + CHSH correlator, S=2√2 Tsirelson bound |
| [H₁₂ chain ground state](https://colab.research.google.com/github/qumulator/qumulator-sdk/blob/main/notebooks/h12_vqe.ipynb) | 12-site Ising chain solved with KLT solver — `client.klt` |
| [Vortex geometry](https://colab.research.google.com/github/qumulator/qumulator-sdk/blob/main/notebooks/vortex_geometry.ipynb) | Entanglement growth in random H/CX/Rz brickwork circuits |
