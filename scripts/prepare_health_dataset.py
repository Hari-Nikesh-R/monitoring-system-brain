"""
Prepare a YOLO classification dataset from the repo's example images.

Input (existing):
  - trained/cow/                 (treated as "healthy")
  - trained/unhealthy_cow/       (treated as "unhealthy")

Output:
  datasets/cow_health/
    train/{healthy,unhealthy}/
    val/{healthy,unhealthy}/

This script creates *symlinks* by default to avoid copying large images.

Usage:
  python3 scripts/prepare_health_dataset.py
  python3 scripts/prepare_health_dataset.py --val-ratio 0.2 --copy
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def list_images(d: Path) -> list[Path]:
    if not d.exists():
        return []
    return [p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS and p.is_file()]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    if dst.exists():
        return
    if copy:
        dst.write_bytes(src.read_bytes())
        return
    # symlink relative when possible
    try:
        rel = os.path.relpath(src, start=dst.parent)
        dst.symlink_to(rel)
    except Exception:
        dst.symlink_to(src)


def split(items: list[Path], val_ratio: float, seed: int) -> tuple[list[Path], list[Path]]:
    rnd = random.Random(seed)
    items = items[:]
    rnd.shuffle(items)
    n_val = max(1, int(len(items) * val_ratio)) if len(items) >= 5 else max(1, int(len(items) * val_ratio))
    val = items[:n_val]
    train = items[n_val:]
    return train, val


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--healthy-dir",
        default="trained/cow",
        help="Directory of healthy cow images (or comma-separated list)",
    )
    ap.add_argument("--unhealthy-dir", default="trained/unhealthy_cow", help="Directory of unhealthy cow images")
    ap.add_argument("--out", default="datasets/cow_health", help="Output dataset directory")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy", action="store_true", help="Copy files instead of symlink")
    args = ap.parse_args()

    healthy_dirs = [s.strip() for s in str(args.healthy_dir).split(",") if s.strip()]
    healthy: list[Path] = []
    for d in healthy_dirs:
        healthy.extend(list_images(Path(d)))
    unhealthy = list_images(Path(args.unhealthy_dir))

    if not healthy or not unhealthy:
        raise SystemExit(
            f"Need images in both '{args.healthy_dir}' and '{args.unhealthy_dir}'. "
            f"Found healthy={len(healthy)} unhealthy={len(unhealthy)}."
        )

    out = Path(args.out)
    for split_name in ("train", "val"):
        for cls in ("healthy", "unhealthy"):
            ensure_dir(out / split_name / cls)

    h_tr, h_val = split(healthy, args.val_ratio, args.seed)
    u_tr, u_val = split(unhealthy, args.val_ratio, args.seed + 1)

    for src in h_tr:
        link_or_copy(src, out / "train" / "healthy" / src.name, args.copy)
    for src in h_val:
        link_or_copy(src, out / "val" / "healthy" / src.name, args.copy)
    for src in u_tr:
        link_or_copy(src, out / "train" / "unhealthy" / src.name, args.copy)
    for src in u_val:
        link_or_copy(src, out / "val" / "unhealthy" / src.name, args.copy)

    print("Dataset prepared at:", out)
    print("Train healthy:", len(h_tr), "Train unhealthy:", len(u_tr))
    print("Val healthy:", len(h_val), "Val unhealthy:", len(u_val))


if __name__ == "__main__":
    main()

