#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a 1:1 balanced KSDD2 subset directly from KSDD2_raw:
- Detect defects by non-empty masks found anywhere under RAW (train/test)
- Use ALL defect images
- Randomly sample an equal number of good images
- Output under OUT/val/{good,defect} as symlinks (use --copy to copy files)
- Save split lists for reproducibility
"""

import os, re, glob, json, random, shutil, hashlib
from pathlib import Path
from PIL import Image
import numpy as np
import argparse

IMG_EXT = {'.png','.jpg','.jpeg','.bmp','.tif','.tiff','.webp',
           '.PNG','.JPG','.JPEG','.BMP','.TIF','.TIFF','.WEBP'}
MSK_EXT = {'.png','.bmp','.tif','.tiff','.PNG','.BMP','.TIF','.TIFF'}

HINT    = re.compile(r'(mask|labels?|gt|ground[_-]?truth)', re.I)
SUFFIX  = re.compile(r'(?i)(?:[_-]?(?:mask|labels?|gt|ground[_-]?truth))$')

def base_stem_from_mask(stem: str) -> str:
    # strip trailing "_mask"/"_gt"/"_label" etc. to map masks -> image stems
    return SUFFIX.sub('', stem)

def nonzero_mask(p: Path) -> bool:
    try:
        im = Image.open(p)
        if im.mode not in ("L","I","F"):
            im = im.convert("L")
        a = np.array(im)
        return bool(a.any())
    except Exception:
        return False

def ensure_clean_dir(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

def safe_link_or_copy(dst: Path, src: Path, copy: bool):
    # stable name with short hash to avoid collisions
    h = hashlib.md5(str(src.resolve()).encode()).hexdigest()[:6]
    out = dst / f"{src.stem}_{h}{src.suffix.lower()}"
    if out.exists() or out.is_symlink():
        try: out.unlink()
        except: pass
    if copy:
        shutil.copy2(src, out)
    else:
        out.symlink_to(src.resolve())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/home/vipuser/data/KSDD2_raw", help="KSDD2_raw root directory")
    ap.add_argument("--out", default="/home/vipuser/data/ksdd2_1to1_allDef", help="output directory")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--copy", action="store_true", help="copy files instead of symlink")
    args = ap.parse_args()

    RAW = Path(args.raw)
    OUT = Path(args.out)
    if not RAW.exists():
        raise SystemExit(f"RAW not found: {RAW}")

    rng = random.Random(args.seed)

    # 1) collect positive mask stems (non-empty masks)
    pos_stems = set()
    for mp in RAW.rglob("*"):
        if mp.is_file() and (mp.suffix in MSK_EXT) and (HINT.search(mp.name) or HINT.search(str(mp.parent))):
            if nonzero_mask(mp):
                pos_stems.add(base_stem_from_mask(mp.stem).lower())

    # 2) gather all candidate images from train/ and test/, skipping mask/label dirs/files
    imgs = []
    for split in ("train", "test"):
        base = RAW / split
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if not (p.is_file() and p.suffix in IMG_EXT):
                continue
            if HINT.search(p.name) or HINT.search(str(p.parent)):
                continue  # skip annotation images themselves
            imgs.append(p)

    # 3) separate defects vs good by stem membership
    defects, goods = [], []
    for p in imgs:
        if p.stem.lower() in pos_stems:
            defects.append(p)
        else:
            goods.append(p)

    n_def, n_good = len(defects), len(goods)
    if n_def == 0:
        raise SystemExit("No defect images detected (non-empty masks not found). Check RAW structure.")

    # 4) choose all defects + equal number of random goods
    k = n_def
    if k > n_good:
        print(f"[warn] good({n_good}) < defect({n_def}), truncating defects to {n_good}")
        defects = defects[:n_good]
        k = n_good
    sel_goods = rng.sample(goods, k)

    # 5) write outputs
    val_good = OUT / "val" / "good"
    val_def  = OUT / "val" / "defect"
    ensure_clean_dir(val_good.parent)  # clears OUT/val entirely
    val_good.mkdir(parents=True, exist_ok=True)
    val_def.mkdir(parents=True, exist_ok=True)

    for p in sel_goods:
        safe_link_or_copy(val_good, p, copy=args.copy)
    for p in defects:
        safe_link_or_copy(val_def, p, copy=args.copy)

    # 6) save split lists + manifest
    splits_dir = Path("/home/vipuser/runs_eval/splits")
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir/"ksdd2_1to1_good.txt").write_text("\n".join(str(x.resolve()) for x in sel_goods))
    (splits_dir/"ksdd2_1to1_defect.txt").write_text("\n".join(str(x.resolve()) for x in defects))

    manifest = {
        "raw": str(RAW),
        "out": str(OUT),
        "seed": args.seed,
        "counts": {"good_pool": n_good, "defect_pool": n_def, "good_selected": len(sel_goods), "defect_selected": len(defects)},
        "copy_mode": bool(args.copy),
    }
    (OUT/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[1to1 READY] good={len(sel_goods)}  defect={len(defects)}  -> {OUT}")
    print(f"lists: {splits_dir/'ksdd2_1to1_good.txt'} | {splits_dir/'ksdd2_1to1_defect.txt'}")

if __name__ == "__main__":
    main()
