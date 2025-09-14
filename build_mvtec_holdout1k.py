#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 MVTec AD 构造评测用 holdout：
out/val/{good,defect}，复制时用 类别/子类 前缀改名，避免同名覆盖。
"""
import os, glob, shutil, argparse, json, random
from pathlib import Path

EXTS = (".png",".jpg",".jpeg",".bmp",".tif",".tiff")

def list_imgs(d):
    fs=[]
    for e in EXTS: fs += glob.glob(os.path.join(d, f"*{e}"))
    return sorted(fs)

def safecopy(src_path: str, dst_dir: Path, prefix: str) -> str:
    dst_dir.mkdir(parents=True, exist_ok=True)
    bn = os.path.basename(src_path)
    dst = dst_dir / f"{prefix}__{bn}"
    i = 1
    while dst.exists():
        dst = dst_dir / f"{prefix}__{i}__{bn}"
        i += 1
    shutil.copy2(src_path, dst)
    return str(dst)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="mvtec_anomaly_detection 根目录")
    ap.add_argument("--out", required=True, help="输出根目录（将创建 out/val/good, out/val/defect）")
    ap.add_argument("--n-good", type=int, default=500)
    ap.add_argument("--n-defect", type=int, default=500)
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    good_dst   = out/"val"/"good"
    defect_dst = out/"val"/"defect"
    if out.exists():
        shutil.rmtree(out)
    good_dst.mkdir(parents=True, exist_ok=True)
    defect_dst.mkdir(parents=True, exist_ok=True)

    cats=[d for d in sorted(os.listdir(args.src))
          if all(os.path.isdir(os.path.join(args.src,d,x)) for x in ("train","test","ground_truth"))]


    good_pool = []      # [(path, cat)]
    defect_pool = []    # [(path, cat, sub)]
    for c in cats:
        cdir = os.path.join(args.src, c)
        for p in list_imgs(os.path.join(cdir,"test","good")):
            good_pool.append((p, c))
        test_root = os.path.join(cdir,"test")
        for sub in os.listdir(test_root):
            if sub == "good": continue
            for p in list_imgs(os.path.join(test_root, sub)):
                defect_pool.append((p, c, sub))

    rng.shuffle(good_pool)
    rng.shuffle(defect_pool)


    pick_good   = good_pool[:min(args.n_good,   len(good_pool))]
    pick_defect = defect_pool[:min(args.n_defect, len(defect_pool))]


    copied_good=[]; copied_def=[]
    for p, c in pick_good:
        copied_good.append(safecopy(p, good_dst, f"{c}"))
    for p, c, sub in pick_defect:
        copied_def.append(safecopy(p, defect_dst, f"{c}__{sub}"))

    manifest = {
        "src": args.src,
        "out": str(out),
        "seed": args.seed,
        "requested": {"good": args.n_good, "defect": args.n_defect},
        "actual": {"good": len(copied_good), "defect": len(copied_def), "total": len(copied_good)+len(copied_def)},
        "copied_good": copied_good[:10] + (["..."] if len(copied_good)>10 else []),
        "copied_defect": copied_def[:10] + (["..."] if len(copied_def)>10 else []),
    }
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n[HOLDOUT READY]")
    print(f"good={len(copied_good)}  defect={len(copied_def)}  total={len(copied_good)+len(copied_def)}")
    print(f"root: {out}  (结构：val/good, val/defect)")
    print(f"manifest: {out/'manifest.json'}\n")

if __name__ == "__main__":
    main()


