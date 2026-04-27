"""
Qumulator circuit simulation backends.

Optional adapters for Qiskit and Cirq.  Install the relevant package
to activate each backend:

    pip install qiskit          # enables QumulatorBackend
    pip install cirq            # enables QumulatorSimulator
"""

# Lazy imports so missing qiskit/cirq doesn't break the core SDK.

def _qiskit_backend():
    from qumulator.backends.qiskit_backend import QumulatorBackend
    return QumulatorBackend


def _cirq_simulator():
    from qumulator.backends.cirq_simulator import QumulatorSimulator
    return QumulatorSimulator


# Direct exports — only available when the optional dependency is installed.
try:
    from qumulator.backends.qiskit_backend import QumulatorBackend
except ImportError:
    pass

try:
    from qumulator.backends.cirq_simulator import QumulatorSimulator
except ImportError:
    pass


__all__ = ["QumulatorBackend", "QumulatorSimulator"]
