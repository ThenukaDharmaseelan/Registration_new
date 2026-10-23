"""Compatibility entry point for ``python -m fafir_registration.cli``."""

from __future__ import annotations

import importlib
import runpy

_legacy_cli = importlib.import_module("retina_multimodal_reg_new.cli")
for _name in dir(_legacy_cli):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_legacy_cli, _name)

for _name in [
    "_iter_pairs",
    "_session_tag",
    "_good_image_only_gate",
    "_pick_better_result",
    "_is_low_keypoint_quality_issue",
    "_vessel_mask_missing",
]:
    if hasattr(_legacy_cli, _name):
        globals()[_name] = getattr(_legacy_cli, _name)

__all__ = [
    name
    for name in globals()
    if not name.startswith("_")
    or name in {
        "_iter_pairs",
        "_session_tag",
        "_good_image_only_gate",
        "_pick_better_result",
        "_is_low_keypoint_quality_issue",
        "_vessel_mask_missing",
    }
]

if __name__ == "__main__":
    runpy.run_module("retina_multimodal_reg_new.cli", run_name="__main__")
