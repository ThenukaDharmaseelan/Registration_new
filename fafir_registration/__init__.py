"""Legacy compatibility package for the old ``fafir_registration`` name.

This shim preserves the historical import surface for the validated legacy
pipeline in :mod:`retina_multimodal_reg`, while still allowing the project to
keep the newer experimental implementation in a separate package.
"""

from __future__ import annotations

import importlib
import sys

_MODULES = [
    "io",
    "matching",
    "metrics",
    "pipeline",
    "preprocessing",
    "registration",
    "utils",
    "visualization",
]

_impl = importlib.import_module("retina_multimodal_reg_new")
_pipeline = importlib.import_module("retina_multimodal_reg.pipeline")

register = getattr(_impl, "register", _pipeline.register)
RegistrationResult = getattr(_impl, "RegistrationResult", _pipeline.RegistrationResult)
metrics = _impl.metrics
visualization = _impl.visualization
__version__ = _impl.__version__

__all__ = list(getattr(_impl, "__all__", ()))

for module_name in _MODULES:
    module = importlib.import_module(f"retina_multimodal_reg_new.{module_name}")
    sys.modules[f"{__name__}.{module_name}"] = module
    globals()[module_name] = module
