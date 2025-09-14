#!/usr/bin/env python3
import argparse, json, time, subprocess, psutil, os
from typing import Optional

try:
    import pynvml
    _nvml_ok = True
except Exception:
    _nvml_ok = False

def now_gpu(dev_idx=0):
    if not _nvml_ok:
        return None
    try:
        d = pynvml.nvmlDeviceGetHandleByIndex(dev_idx)
        util = pynvml.nvmlDeviceGetUtilizationRates(d)
        pwr_mw = pynvml.nvmlDeviceGetPowerUsage(d)
        return {"gpu_util": float(util.gpu), "power_w": float(pwr_mw) / 1000.0}
    except Exception:
        return None

def rss_with_children(proc: psutil.Process) -> int:
    total = 0
    try:
        if proc.is_running():
            total += proc.memory_info().rss
            for c in proc.children(recursive=True):
                try:
                    total += c.memory_info().rss
                except Exception:
                    pass
    except Exception:
        pass
    return total

def read_acc_from_summary(summary_path: str) -> Optional[float]:
    try:
        obj = json.load(open(summary_path, "r", encoding="utf-8"))
        m = obj.get("metrics", {}).get("merged", {})
        if "acc" in m and isinstance(m["acc"], (int, float)):
            return float(m["acc"])
        cm = m.get("cm")
        N = m.get("N")
        if isinstance(cm, list) and len(cm) == 2 and len(cm[0]) == 2 and len(cm[1]) == 2 and isinstance(N, (int, float)):
            tp, fn = cm[0]
            fp, tn = cm[1]
            correct = float(tp + tn)
            if N:
                return correct / float(N)
    except Exception:
        pass
    return None

def read_correct_from_summary(summary_path: str) -> Optional[int]:
    try:
        obj = json.load(open(summary_path, "r", encoding="utf-8"))
        m = obj.get("metrics", {}).get("merged", {})
        cm = m.get("cm")
        if isinstance(cm, list) and len(cm) == 2 and len(cm[0]) == 2 and len(cm[1]) == 2:
            tp, fn = cm[0]
            fp, tn = cm[1]
            return int(tp + tn)
        acc = m.get("acc", None)
        N = m.get("N", None)
        if isinstance(acc, (int, float)) and isinstance(N, (int, float)):
            return int(round(float(acc) * float(N)))
    except Exception:
        pass
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--interval", type=float, default=0.2)
    ap.add_argument("--save", required=True)
    ap.add_argument("--summary_json", default=None)
    ap.add_argument("--accuracy", type=float, default=None)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    if args.cmd and args.cmd[0] == "--":
        args.cmd = args.cmd[1:]

    if _nvml_ok:
        try:
            pynvml.nvmlInit()
        except Exception:
            pass

    t0 = time.time()
    p = subprocess.Popen(args.cmd)
    proc = psutil.Process(p.pid)

    gpu_utils, powers = [], []
    rss_samples_mb = []

    while True:
        ret = p.poll()
        g = now_gpu(args.gpu)
        if g:
            gpu_utils.append(g["gpu_util"])
            powers.append(g["power_w"])
        rss = rss_with_children(proc)
        if rss > 0:
            rss_samples_mb.append(rss / (1024 ** 2))
        if ret is not None:
            break
        time.sleep(args.interval)

    wall = time.time() - t0

    acc = None
    correct = None
    if args.summary_json and os.path.isfile(args.summary_json):
        acc = read_acc_from_summary(args.summary_json)
        correct = read_correct_from_summary(args.summary_json)
    if acc is None and args.accuracy is not None:
        acc = float(args.accuracy)

    gpu_util_avg = round(sum(gpu_utils) / len(gpu_utils), 2) if gpu_utils else 0.0
    power_avg_w  = round(sum(powers) / len(powers), 3) if powers else 0.0
    mem_avg_gb = round((sum(rss_samples_mb) / max(len(rss_samples_mb), 1)) / 1024.0, 3) if rss_samples_mb else 0.0

    total_energy_mwh = power_avg_w * wall * (1000.0 / 3600.0)
    if correct is not None and correct > 0:
        energy_per_correct_mwh = round(total_energy_mwh / float(correct), 4)
    elif acc is not None and acc > 0 and args.summary_json:
        try:
            obj = json.load(open(args.summary_json, "r", encoding="utf-8"))
            N = obj.get("metrics", {}).get("merged", {}).get("N", None)
            if isinstance(N, (int, float)) and N > 0:
                energy_per_correct_mwh = round(total_energy_mwh / (acc * float(N)), 4)
            else:
                energy_per_correct_mwh = 0.0
        except Exception:
            energy_per_correct_mwh = 0.0
    else:
        energy_per_correct_mwh = 0.0

    out = {
        "accuracy": None if acc is None else float(acc),
        "time_parallel_s": float(wall),
        "gpu_util_avg_%": float(gpu_util_avg),
        "gpu_power_avg_W": float(power_avg_w),
        "energy_per_correct_mWh": float(energy_per_correct_mwh),
        "cpu_mem_avg_GB": float(mem_avg_gb),
    }

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    with open(args.save, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("\n===== EVAL + PROFILING (minimal) =====")
    print(json.dumps(out, indent=2, ensure_ascii=False))

    if _nvml_ok:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

if __name__ == "__main__":
    main()
