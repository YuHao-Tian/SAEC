1) 数据准备脚本（train/val JSONL）
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, glob, random, argparse
from pathlib import Path

IMG_EXTS = (".png",".jpg",".jpeg",".bmp",".tif",".tiff")

def list_files(d):
    return [f for f in glob.glob(os.path.join(d, "*"))
            if os.path.splitext(f)[1].lower() in IMG_EXTS]

def main(src, out, defect_train_ratio=0.6, seed=42):
    random.seed(seed)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    jtrain, jval = [], []

    cats = [c for c in sorted(os.listdir(src))
            if all(os.path.isdir(os.path.join(src,c,x)) for x in ("train","test","ground_truth"))]
    assert cats, f"No MVTec categories under {src}"

    for c in cats:
        train_good = list_files(os.path.join(src, c, "train", "good"))
        test_root  = os.path.join(src, c, "test")
        test_good  = list_files(os.path.join(test_root, "good"))

        defects = []
        for sub in os.listdir(test_root):
            if sub=="good": continue
            defects += list_files(os.path.join(test_root, sub))
        random.shuffle(defects)

        k = int(len(defects)*defect_train_ratio)
        train_def, val_def = defects[:k], defects[k:]

        # 组装指令数据（messages 将在训练时再拼成 chat 模版）
        for f in train_good:
            jtrain.append({"image": f, "label": 0})
        for f in train_def:
            jtrain.append({"image": f, "label": 1})
        for f in test_good:
            jval.append({"image": f, "label": 0})
        for f in val_def:
            jval.append({"image": f, "label": 1})

    random.shuffle(jtrain); random.shuffle(jval)
    (out/"train.jsonl").write_text("\n".join(json.dumps(x,ensure_ascii=False) for x in jtrain))
    (out/"val.jsonl").write_text("\n".join(json.dumps(x,ensure_ascii=False) for x in jval))
    print(f"[done] train={len(jtrain)}  val={len(jval)} -> {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/home/vipuser/mvtec_anomaly_detection")
    ap.add_argument("--out", default="/home/vipuser/data/mvtec_qwen_sft")
    ap.add_argument("--defect-train-ratio", type=float, default=0.6)
    args = ap.parse_args()
    main(args.src, args.out, args.defect_train_ratio)
