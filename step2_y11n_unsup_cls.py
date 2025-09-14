#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step2: 在 simple 子集上用 YOLO-*-cls（未微调）做“无监督二分类”，并导出逐样本 CSV。

新增功能：
- --smax_thr / --margin_thr / --ent_thr：可显式指定阈值（用于“让 11s 用 11n 的阈值”）
"""

import os, glob, argparse, math
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO


def list_imgs(root):
    items = []
    for cls in ["good", "defect"]:
        d = os.path.join(root, "val", cls)
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
            for p in glob.glob(os.path.join(d, ext)):
                items.append((p, cls))
    return items


def scan_train_good(mvtec_root, model):
    """仅在 train/good 上做阈值统计（无监督、避免泄漏）"""
    goods = []
    for cat in sorted(os.listdir(mvtec_root)):
        gdir = os.path.join(mvtec_root, cat, "train", "good")
        if not os.path.isdir(gdir):
            continue
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            goods += glob.glob(os.path.join(gdir, ext))

    smaxs, margins, ents = [], [], []
    for p in tqdm(goods, ncols=100, desc="calibrate on train/good"):
        r = model.predict(p, imgsz=448, verbose=False)[0]
        prob = r.probs.data.cpu().numpy()
        ps = np.sort(prob)[::-1]
        smax = float(ps[0])
        margin = float(ps[0] - ps[1] if ps.size > 1 else ps[0])
        ent = float(-(prob * np.log(prob + 1e-12)).sum() / math.log(len(prob)))
        smaxs.append(smax)
        margins.append(margin)
        ents.append(ent)

    thr = {
        "smax_p20": float(np.quantile(smaxs, 0.20)),
        "margin_p20": float(np.quantile(margins, 0.20)),
        "ent_p80": float(np.quantile(ents, 0.80)),
    }
    return thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simple_data", default="/home/vipuser/data/mvtec_holdout_simple")
    ap.add_argument("--mvtec_root", default="/home/vipuser/data/mvtec_anomaly_detection")
    ap.add_argument("--weights", default="yolo11n-cls.pt")
    ap.add_argument("--save_csv", default="/home/vipuser/runs_eval/y11n_simple_preds.csv")
    # >>> 新增：可选外部阈值（若提供则覆盖统计得到的阈值）
    ap.add_argument("--smax_thr", type=float, default=None)
    ap.add_argument("--margin_thr", type=float, default=None)
    ap.add_argument("--ent_thr", type=float, default=None)
    args = ap.parse_args()

    model = YOLO(args.weights)

    # 1) 在 train/good 上统计阈值
    THR = scan_train_good(args.mvtec_root, model)
    print("[THR]", THR)

    # 2) 如果命令行传入阈值，则覆盖
    if args.smax_thr is not None:
        THR["smax_p20"] = float(args.smax_thr)
    if args.margin_thr is not None:
        THR["margin_p20"] = float(args.margin_thr)
    if args.ent_thr is not None:
        THR["ent_p80"] = float(args.ent_thr)
    print("[THR effective]", THR)

    # 3) 推理 simple 子集并二分类
    items = list_imgs(args.simple_data)
    rows = []
    tp = tn = fp = fn = 0
    for p, gt in tqdm(items, ncols=100, desc=f"infer simple with {os.path.basename(args.weights).split('.')[0]}"):
        r = model.predict(p, imgsz=448, verbose=False)[0]
        prob = r.probs.data.cpu().numpy()
        ps = np.sort(prob)[::-1]
        smax = float(ps[0])
        margin = float(ps[0] - ps[1] if ps.size > 1 else ps[0])
        ent = float(-(prob * np.log(prob + 1e-12)).sum() / math.log(len(prob)))

        # 规则：正常性高 => good，否则 defect
        pred = "good" if (smax >= THR["smax_p20"] and margin >= THR["margin_p20"] and ent <= THR["ent_p80"]) else "defect"
        rows.append([p, gt, pred, smax, margin, ent])

        if   gt == "good"   and pred == "good":   tp += 1
        elif gt == "good"   and pred == "defect": fn += 1
        elif gt == "defect" and pred == "defect": tn += 1
        else:                                      fp += 1

    os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)
    pd.DataFrame(rows, columns=["path", "gt", "pred", "smax", "margin", "ent"]).to_csv(args.save_csv, index=False)

    # 4) 指标
    N = len(items)
    acc = (tp + tn) / max(N, 1)
    good_acc = tp / max(tp + fn, 1)
    defect_acc = tn / max(tn + fp, 1)
    bal = 0.5 * (good_acc + defect_acc)

    print(f"\n[{os.path.basename(args.weights)} simple] N={N}  Acc={acc*100:.2f}%  "
          f"good={good_acc*100:.2f}%  defect={defect_acc*100:.2f}%  BalAcc={bal*100:.2f}%")
    print(f"Confusion: [[good→good {tp}], [good→defect {fn}], [defect→defect {tn}], [defect→good {fp}]]")
    print(f"[CSV] {args.save_csv}")


if __name__ == "__main__":
    main()
