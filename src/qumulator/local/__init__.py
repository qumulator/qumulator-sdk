"""
qumulator.local — local statevector simulator (no API key required).

Usage::

    from qumulator.local import LocalStatevectorEngine

    eng = LocalStatevectorEngine(n_qubits=2)
    eng.apply('h', 0).apply('cx', [0, 1])
    print(eng.sample(shots=1024))          # {'00': ~512, '11': ~512}

    result = eng.run(shots=2048, return_entropy_map=True)
    print(result.entropy_map)              # [~1.0, ~1.0]

GPU acceleration (optional)::

    eng = LocalStatevectorEngine(n_qubits=20, device='gpu')
    # auto-detects CuPy → JAX → PyTorch; falls back to CPU silently
"""

from qumulator.local.engine import LocalStatevectorEngine

__all__ = ["LocalStatevectorEngine"]
