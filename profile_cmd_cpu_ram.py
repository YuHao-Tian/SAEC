#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, time, subprocess, psutil, math, os, sys
try:
    import pynvml
    _nvml_ok = True
except Exception:
    _nvml_ok = False

def now_gpu(n=0):
    if not _nvml_ok: return None
    h = {}
    try:
        d = pynvml.nvmlDeviceGetHandleByIndex(n)
        util = pynvml.nvmlDeviceGetUtilizationRates(d)
        mem  = pynvml.nvmlDeviceGetMemoryInfo(d)
        pwr  = pynvml.nvmlDeviceGetPowerUsage(d)  # mW
        h["gpu_util"] = float(util.gpu)
        h["vram_used_mb"] = float(mem.used) / (1024**2)
        h["power_w"] = float(pwr) / 1000.0
        return h
    except Exception:
        return None

def sum_children_mem(proc):
    rss = 0
    try:
        if proc.is_running():
            rss += proc.memory_info().rss
            for c in proc.children(recursive=True):
                try: rss += c.memory_info().rss
                except: pass
    except: pass
    return rss

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--interval", type=float, default=0.2)
    ap.add_argument("--save", required=True)
    ap.add_argument("--images", type=int, default=0, help="optional: total images for avg s/img")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="command to run after '--'")
    args = ap.parse_args()

    if _nvml_ok:
        pynvml.nvmlInit()

    if args.cmd and args.cmd[0] == "--":  # allow: ... profile.py -- python script.py ...
        args.cmd = args.cmd[1:]

    t0 = time.time()
    p = subprocess.Popen(args.cmd)
    proc = psutil.Process(p.pid)

    gpu_utils, vrams, pwrs = [], [], []
    ram_proc_samples = []
    cpu_util_samples = []

    while True:
        ret = p.poll()
        # GPU
        g = now_gpu(args.gpu)
        if g:
            gpu_utils.append(g["gpu_util"])
            vrams.append(g["vram_used_mb"])
            pwrs.append(g["power_w"])
        # CPU (process+children RSS) & system CPU util
        ram_bytes = sum_children_mem(proc)
        if ram_bytes > 0:
            ram_proc_samples.append(ram_bytes / (1024**2))  # MB
        try:
            cpu_util_samples.append(psutil.cpu_percent(interval=None))
        except: pass

        if ret is not None:
            break
        time.sleep(args.interval)

    wall = time.time() - t0
    N = max(args.images, 0)
    avg_s_per_img = (wall / N) if N > 0 else 0.0

    out = {
        "nvml": _nvml_ok,
        "duration_s": wall,
        "wall_time_s": wall,
        "avg_s_per_img": avg_s_per_img,
        "interval_s": args.interval,
        "return_code": p.returncode,
        # GPU
        "gpu_util_avg_%": round(sum(gpu_utils)/len(gpu_utils), 2) if gpu_utils else 0.0,
        "gpu_util_p95_%": round(sorted(gpu_utils)[int(0.95*len(gpu_utils))-1], 1) if gpu_utils else 0.0,
        "vram_avg_MB": round(sum(vrams)/len(vrams), 1) if vrams else 0.0,
        "vram_peak_MB": round(max(vrams), 1) if vrams else 0.0,
        "power_avg_W": round(sum(pwrs)/len(pwrs), 1) if pwrs else 0.0,
        "power_peak_W": round(max(pwrs), 1) if pwrs else 0.0,
        # CPU (process)
        "ram_proc_avg_MB": round(sum(ram_proc_samples)/len(ram_proc_samples), 1) if ram_proc_samples else 0.0,
        "ram_proc_peak_MB": round(max(ram_proc_samples), 1) if ram_proc_samples else 0.0,
        "cpu_util_avg_%": round(sum(cpu_util_samples)/len(cpu_util_samples), 2) if cpu_util_samples else 0.0,
    }
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    json.dump(out, open(args.save, "w"), indent=2)
    print("\n===== PROFILING SUMMARY (GPU + CPU-RAM) =====")
    print(json.dumps(out, indent=2))

    if _nvml_ok:
        pynvml.nvmlShutdown()

if __name__ == "__main__":
    main()
