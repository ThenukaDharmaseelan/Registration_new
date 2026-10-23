from . import metrics, visualization
from . import pipeline as _pipeline

register = _pipeline.register
RegistrationResult = _pipeline.RegistrationResult

__version__ = "0.1.0"

__all__ = [
    "register",
    "RegistrationResult",
    "metrics",
    "visualization",
    "__version__",
]