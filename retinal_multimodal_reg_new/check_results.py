#!/usr/bin/env python3
from __future__ import annotations

# Compatibility constants used by the batch CLI and report helpers.
DICE_OK = 0.30
ROT_FLAG = 5.0
ANISO_FLAG = 1.10
FOV_RATIO = 55.0 / 30.0

__all__ = ["DICE_OK", "ROT_FLAG", "ANISO_FLAG", "FOV_RATIO"]

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# def load(path):
#     df = pd.read_csv(path)
#     df = df[df.get("status", "ok") == "ok"].copy()
#     if "name" in df.columns:
#         df["pair"] = df["name"].astype(str).str[:4]
#     for c in ("transform_sx", "transform_sy"):
#         if c not in df.columns:
#             df[c] = np.nan
#     df["aniso"] = (df[["transform_sx", "transform_sy"]].max(1)
#                    / df[["transform_sx", "transform_sy"]].min(1))
#     df["rot"] = df.get("transform_angle_deg", pd.Series(np.nan)).abs()
#     df["scale_mean"] = (df.transform_sx + df.transform_sy) / 2
#     df["good"] = df.dice_after >= DICE_OK
#     df["hd95_worse"] = df.hd95_after > df.hd95_before
#     return df


# def summary(df, label=""):
#     n = len(df)
#     print(f"\n=== {label or 'run'}  (n={n}) ===")
#     print(f"  success (dice >= {DICE_OK}) : {int(df.good.sum())}/{n}"
#           f"  ({100*df.good.mean():.0f}%)")
#     for m in ("dice_after", "hd95_after", "asd_after", "ncc_after"):
#         if m in df.columns:
#             print(f"  {m:<12} mean {df[m].mean():8.4f}   median {df[m].median():8.4f}")
#     print(f"  hd95 WORSE than before   : {int(df.hd95_worse.sum())}/{n}"
#           f"   <- registration actively harmful")


# def why(r):
#     t = []
#     if r.rot > ROT_FLAG:
#         t.append(f"rot {r.rot:.1f}deg")
#     if r.aniso > ANISO_FLAG:
#         t.append(f"stretch {r.aniso:.2f}")
#     if abs(r.scale_mean - FOV_RATIO) > 0.25:
#         t.append(f"scale {r.scale_mean:.2f} (want {FOV_RATIO:.2f})")
#     if "-img" in str(r.get("label", "")):
#         t.append("image-fallback")
#     if r.hd95_worse:
#         t.append("hd95 worse")
#     return ", ".join(t) or "low-quality inputs"


# def failures(df):
#     bad = df[~df.good].sort_values("dice_after")
#     if bad.empty:
#         print("\n  no failures")
#         return
#     print(f"\n=== {len(bad)} FAILING PAIRS ===")
#     print(f"{'pair':>5}{'dice':>8}{'hd95':>8}{'rot':>7}{'scale':>7}  why")
#     for _, r in bad.iterrows():
#         print(f"{r.pair:>5}{r.dice_after:>8.3f}{r.hd95_after:>8.1f}"
#               f"{r.rot:>7.1f}{r.scale_mean:>7.2f}  {why(r)}")


# def show(df, pairs):
#     print(f"\n=== requested pairs ===")
#     for p in pairs:
#         key = f"{int(p):04d}"
#         sub = df[df.pair == key]
#         if sub.empty:
#             print(f"  {key}: not found")
#             continue
#         r = sub.iloc[0]
#         print(f"\n  pair {key}   label={r.get('label','?')}")
#         print(f"    dice  {r.dice_before:.4f} -> {r.dice_after:.4f}")
#         print(f"    hd95  {r.hd95_before:8.2f} -> {r.hd95_after:8.2f}"
#               f"   {'(WORSE)' if r.hd95_worse else ''}")
#         print(f"    asd   {r.asd_before:8.2f} -> {r.asd_after:8.2f}")
#         print(f"    rot {r.get('transform_angle_deg', float('nan')):+7.2f}deg   "
#               f"sx {r.transform_sx:.3f}  sy {r.transform_sy:.3f}  "
#               f"aniso {r.aniso:.3f}")
#         print(f"    verdict: {'OK' if r.good else 'FAIL — ' + why(r)}")


# def compare(a, b, name_a, name_b):
#     m = a[["pair", "dice_after", "hd95_after"]].merge(
#         b[["pair", "dice_after", "hd95_after"]], on="pair",
#         suffixes=("_a", "_b"))
#     m["d_dice"] = m.dice_after_b - m.dice_after_a
#     print(f"\n=== {name_b} vs {name_a}  (n={len(m)}) ===")
#     print(f"  success  {int((m.dice_after_a>=DICE_OK).sum())} -> "
#           f"{int((m.dice_after_b>=DICE_OK).sum())}")
#     print(f"  mean dice {m.dice_after_a.mean():.4f} -> {m.dice_after_b.mean():.4f}"
#           f"   ({m.d_dice.mean():+.4f})")
#     print(f"  mean hd95 {m.hd95_after_a.mean():.2f} -> {m.hd95_after_b.mean():.2f}")
#     imp = m[m.d_dice > 0.05].sort_values("d_dice", ascending=False)
#     reg = m[m.d_dice < -0.05].sort_values("d_dice")
#     if not imp.empty:
#         print(f"\n  IMPROVED ({len(imp)}):")
#         for _, r in imp.head(15).iterrows():
#             print(f"    {r.pair}  {r.dice_after_a:.3f} -> {r.dice_after_b:.3f}"
#                   f"  ({r.d_dice:+.3f})")
#     if not reg.empty:
#         print(f"\n  REGRESSED ({len(reg)}):")
#         for _, r in reg.head(15).iterrows():
#             print(f"    {r.pair}  {r.dice_after_a:.3f} -> {r.dice_after_b:.3f}"
#                   f"  ({r.d_dice:+.3f})")


# def main(argv):
#     if len(argv) < 2:
#         print(__doc__)
#         return 2
#     path = argv[1]
#     df = load(path)

#     if "--compare" in argv:
#         other = argv[argv.index("--compare") + 1]
#         base = load(other)
#         summary(base, Path(other).parent.name)
#         summary(df, Path(path).parent.name)
#         compare(base, df, Path(other).parent.name, Path(path).parent.name)
#         return 0

#     pairs = [a for a in argv[2:] if a.isdigit()]
#     summary(df, Path(path).parent.name)
#     if pairs:
#         show(df, pairs)
#     else:
#         failures(df)
#     return 0


# if __name__ == "__main__":
#     raise SystemExit(main(sys.argv))