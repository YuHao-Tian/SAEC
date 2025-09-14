#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, re, csv, json, time, argparse, subprocess, pathlib, glob
from pathlib import Path
import numpy as np
from PIL import Image

PROMPT_ZERO_SHOT = """[Inspector role]
You are an industrial quality inspector at the final station.

[Task]
Classify the product in the image into exactly one label: good or defect.

[Focus]
Only consider the product itself; ignore background, fixtures, table, lighting glare, and minor illumination variations.

[Defect cues]
• Scratch / crack / hole / missing material
• Stain / foreign object / residue / leakage
• Missing or misaligned printing
• Clear color or texture anomaly
• Misassembly / omission / position shift
• Abnormal weld / seam

[Output]
Respond strictly as: "Label: 0" or "Label: 1".
"""

PATH2GT = {}
EXTS = (".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp",".PNG",".JPG",".JPEG",".BMP",".TIF",".TIFF",".WEBP")

def list_imgs(root):
    items=[]
    for cls in ("good","defect"):
        d=os.path.join(root,"val",cls)
        for ext in EXTS:
            for p in glob.glob(os.path.join(d,"*"+ext)):
                rp=os.path.realpath(p)
                PATH2GT[rp]=cls
                PATH2GT[p]=cls
                items.append((rp,cls))
    return sorted(items)

def gt_from_path(path:str)->str:
    rp=os.path.realpath(path)
    if rp in PATH2GT: return PATH2GT[rp]
    if path in PATH2GT: return PATH2GT[path]
    p=(path if isinstance(path,str) else str(path)).replace("\\","/").lower()
    if "/val/good/" in p or "/images/normal/" in p: return "good"
    if "/val/defect/" in p or "/images/anomaly/" in p: return "defect"
    parts=[x for x in p.split("/") if x]
    if len(parts)>=2 and parts[-2] in ("good","defect"): return parts[-2]
    name=parts[-1]
    if name.startswith(("good_","normal_")): return "good"
    if name.startswith(("defect_","anomaly_")): return "defect"
    return "good"

def metrics_from_rows(rows):
    tp=tn=fp=fn=0
    for r in rows:
        gt=r["gt"]; pr=r["pred"]
        if   gt=="good"   and pr=="good":   tp+=1
        elif gt=="good"   and pr=="defect": fn+=1
        elif gt=="defect" and pr=="defect": tn+=1
        elif gt=="defect" and pr=="good":   fp+=1
    N=tp+tn+fp+fn
    gtot=tp+fn; dtot=tn+fp
    acc=(tp+tn)/max(N,1)
    gacc=tp/max(gtot,1); dacc=tn/max(dtot,1)
    bal=0.5*(gacc+dacc)
    return dict(N=N, acc=acc, gacc=gacc, dacc=dacc, bal=bal, cm=[[tp,fn],[fp,tn]])

def print_metrics(tag,m):
    print(f"\n=== {tag} ===")
    print(f"N={m['N']} | Acc={m['acc']*100:.2f}% | good={m['gacc']*100:.2f}% | defect={m['dacc']*100:.2f}% | BalAcc={m['bal']*100:.2f}%")
    print("Confusion (rows=GT, cols=Pred) [good,defect]:")
    print(f"[[{m['cm'][0][0]:>3} {m['cm'][0][1]:>3}]")
    print(f" [{m['cm'][1][0]:>3} {m['cm'][1][1]:>3}]]")

def ensure_clean_dir(d):
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)
    for p in pathlib.Path(d).rglob("*"):
        try:
            if p.is_symlink() or p.is_file(): p.unlink()
        except: pass

def make_subset_dataset(img_list, dst_root, gt_map=None):
    if gt_map is None: gt_map = {}
    gdir = os.path.join(dst_root, 'val', 'good'); ddir = os.path.join(dst_root, 'val', 'defect')
    ensure_clean_dir(gdir); ensure_clean_dir(ddir)
    import hashlib
    for p in img_list:
        rp  = os.path.realpath(p)
        cls = gt_map.get(rp, gt_from_path(rp))
        h   = hashlib.md5(rp.encode('utf-8')).hexdigest()[:8]
        name= os.path.splitext(os.path.basename(p))[0] + f'__{h}.png'
        dst = os.path.join(gdir if cls=='good' else ddir, name)
        if os.path.lexists(dst): os.unlink(dst)
        try: os.symlink(p, dst)
        except FileExistsError: pass

def _score_one(path, size=192):
    import cv2
    im  = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if im is None: return (path, 0.0)
    im  = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
    g   = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([g],[0],None,[256],[0,256]).ravel()
    p    = hist / max(hist.sum(), 1)
    H    = float( -np.sum(p*(np.log(p+1e-12))) / np.log(256) )
    Lap  = cv2.Laplacian(g, cv2.CV_32F); s2 = float(np.var(Lap))
    gx   = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy   = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    G    = float(np.mean(np.hypot(gx, gy)))
    ed   = cv2.Canny(g, 64, 128); edge = float(ed.mean())
    enc  = cv2.imencode(".jpg", im, [int(cv2.IMWRITE_JPEG_QUALITY), 30])[1]
    im2  = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    res  = cv2.absdiff(im.astype(np.float32), im2.astype(np.float32))
    C  = 0.30*H + 0.25*edge + 0.20*np.log1p(s2)/8.0 + 0.15*(G/16.0) + 0.10*(res.mean()/255.0)
    return (path, float(C))

def complexity_split(data_root, target_q_ratio=0.70, size=192):
    items = list_imgs(data_root)
    paths = [p for p,_ in items]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as ex:
        scores = list(ex.map(lambda p:_score_one(p,size), paths))
    vals   = np.array([s for _,s in scores], dtype=np.float32)
    thr    = float(np.quantile(vals, 1.0 - target_q_ratio))
    q_list = [p for (p,s) in scores if s >= thr]
    s_list = [p for (p,s) in scores if s <  thr]
    return s_list, q_list, thr

# -------- Qwen eval (batched, thr on prob); if pred==1, also return JSON {bboxes, desc} --------
def qwen_eval_subset_with_thr_batched(base, adapter, data_root, thr=0.19, device="cuda:0",
                                      prompt=PROMPT_ZERO_SHOT,
                                      batch_size=16, resize=448, load_4bit=False, attn="auto",
                                      gen_json_if_defect=True, json_max_new_tokens=192):
    import torch, glob
    from PIL import Image as _Image
    from transformers import AutoProcessor
    from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

    torch.backends.cuda.matmul.allow_tf32 = True
    try: torch.set_float32_matmul_precision("high")
    except Exception: pass

    try:
        if attn in ("flash","flash2"):
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)
        elif attn == "sdpa":
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)
        elif attn == "eager":
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(base)

    qkwargs = {}
    if load_4bit:
        try:
            from transformers import BitsAndBytesConfig
            qkwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        except Exception:
            pass

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base, device_map="auto", torch_dtype=torch.bfloat16, **qkwargs
    )
    if adapter and os.path.exists(adapter):
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter)
        except Exception:
            pass
    model.eval()

    def _list_imgs(d):
        files=[]
        for e in (".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp"):
            files += glob.glob(os.path.join(d, f"*{e}"))
        return sorted(files)

    good = [os.path.realpath(p) for p in _list_imgs(os.path.join(data_root,"val","good"))]
    defect = [os.path.realpath(p) for p in _list_imgs(os.path.join(data_root,"val","defect"))]

    tok = processor.tokenizer
    id0 = tok.encode("0", add_special_tokens=False)[0]
    id1 = tok.encode("1", add_special_tokens=False)[0]

    messages = [{"role":"user","content":[{"type":"image"},{"type":"text","text":prompt.strip()}]}]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    base_text = chat + "Label: "

    json_text = (
        prompt.strip()
        + f"\n\nIf the product is defective (Label 1), return a JSON object with keys 'bboxes' and 'desc'. "
          f"'bboxes' is a list of [x1,y1,x2,y2] pixel coordinates (integers) in the resized {resize}x{resize} image space. "
          "Respond with JSON only."
    )
    json_chat = processor.apply_chat_template(
        [{"role":"user","content":[{"type":"image"},{"type":"text","text":json_text}]}],
        tokenize=False, add_generation_prompt=True
    )

    def _iter_batches(paths):
        for i in range(0, len(paths), batch_size):
            yield paths[i:i+batch_size]

    @torch.inference_mode()
    def _run_prob(paths):
        imgs=[]
        for p in paths:
            im=_Image.open(p).convert("RGB")
            if resize and resize>0: im = im.resize((resize,resize), _Image.BILINEAR)
            imgs.append(im)
        texts=[base_text]*len(imgs)
        inputs = processor(text=texts, images=imgs, return_tensors="pt", padding=True)
        inputs = {k:v.to(dev) for k,v in inputs.items()}
        out = model(**inputs)
        logits = out.logits
        last_idx = inputs["attention_mask"].sum(dim=1) - 1
        gather = logits[torch.arange(logits.size(0), device=logits.device), last_idx]
        two = torch.stack([gather[:, id0], gather[:, id1]], dim=1)
        prob1 = torch.softmax(two, dim=1)[:,1]
        return prob1.detach().float().cpu().numpy()

    @torch.inference_mode()
    def _gen_json_one(path):
        im=_Image.open(path).convert("RGB")
        if resize and resize>0: im = im.resize((resize,resize), _Image.BILINEAR)
        inputs = processor(text=[json_chat], images=[im], return_tensors="pt", padding=True)
        inputs = {k:v.to(dev) for k,v in inputs.items()}
        out_ids = model.generate(**inputs, max_new_tokens=json_max_new_tokens, do_sample=False)
        txt = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
        m = re.search(r"\{.*\}", txt, re.S)
        if not m: return {"bboxes": [], "desc": ""}
        try:
            j = json.loads(m.group(0))
            if "bboxes" not in j: j["bboxes"]=[]
            if "desc"   not in j: j["desc"]=""
            return j
        except Exception:
            return {"bboxes": [], "desc": ""}

    tn=tp=fp=fn=0
    rows=[]
    json_map={}  # path -> JSON for defect images

    for batch in _iter_batches(good):
        p1 = _run_prob(batch)
        pred = (p1 >= thr).astype(np.int32)
        tn += int((pred==0).sum()); fp += int((pred==1).sum())
        for p, v in zip(batch, pred.tolist()):
            rows.append({"path": p, "gt":"good", "pred":"defect" if v else "good"})
            if v==1 and gen_json_if_defect:
                json_map[p] = _gen_json_one(p)

    for batch in _iter_batches(defect):
        p1 = _run_prob(batch)
        pred = (p1 >= thr).astype(np.int32)
        tp += int((pred==1).sum()); fn += int((pred==0).sum())
        for p, v in zip(batch, pred.tolist()):
            rows.append({"path": p, "gt":"defect", "pred":"defect" if v else "good"})
            if v==1 and gen_json_if_defect:
                json_map[p] = _gen_json_one(p)

    return rows, dict(tn=tn, fp=fp, fn=fn, tp=tp), json_map

# -------- YOLO detection for Label==1 visualization (draw rectangles only) --------
def yolo_detect_and_draw(paths, weights, out_dir, conf=0.25, iou=0.45, device="cpu"):
    os.makedirs(out_dir, exist_ok=True)
    try:
        from ultralytics import YOLO
        import cv2
        model = YOLO(weights)
        results = model(paths, conf=conf, iou=iou, device=device, verbose=False)
        if not isinstance(results, (list, tuple)): results=[results]
        for p, r in zip(paths, results):
            try:
                img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None: continue
                if getattr(r, "boxes", None) is None:
                    # nothing detected
                    pass
                else:
                    for xyxy in r.boxes.xyxy.detach().cpu().numpy().tolist():
                        x1,y1,x2,y2 = [int(round(v)) for v in xyxy[:4]]
                        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)  # no text
                base = os.path.splitext(os.path.basename(p))[0] + "_yolo.jpg"
                cv2.imencode(".jpg", img)[1].tofile(os.path.join(out_dir, base))
            except Exception:
                continue
    except Exception as e:
        print(f"[warn] YOLO detection skipped ({e})")

# ================== main ==================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--save_dir", required=True)

    ap.add_argument("--yolo_env_py", default="/home/vipuser/yoloenv/bin/python")
    ap.add_argument("--yolo_step2",  default="/home/vipuser/step2_y11n_unsup_cls.py")
    ap.add_argument("--yolo_weights", default="yolo11s-cls.pt")
    ap.add_argument("--yolo_det_weights", default="/home/vipuser/models/yolo11n.pt")
    ap.add_argument("--mvtec_root", default="/home/vipuser/data/mvtec_anomaly_detection")
    ap.add_argument("--yolo_smax_thr", type=float, default=0.83)
    ap.add_argument("--yolo_margin_thr", type=float, default=0.08)
    ap.add_argument("--yolo_ent_thr", type=float, default=0.95)
    ap.add_argument("--yolo_keep_bidir", type=int, default=0)

    ap.add_argument("--qwen_base", required=True)
    ap.add_argument("--qwen_adapter", required=True)
    ap.add_argument("--qwen_thr", type=float, default=0.19)
    ap.add_argument("--qwen_device", default="cuda:0")
    ap.add_argument("--qwen_batch", type=int, default=16)
    ap.add_argument("--qwen_resize", type=int, default=448)
    ap.add_argument("--qwen_prompt", default=PROMPT_ZERO_SHOT)
    ap.add_argument("--qwen_4bit", type=int, default=0)
    ap.add_argument("--qwen_attn", default="auto")

    ap.add_argument("--init_q_ratio", type=float, default=0.70)
    ap.add_argument("--cpx_size", type=int, default=192)

    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # complexity
    t0=time.time()
    simple_list, q_list_init, cpx_thr = complexity_split(args.data, target_q_ratio=args.init_q_ratio, size=args.cpx_size)
    t_complex = time.time()-t0
    open(os.path.join(args.save_dir,"simple_list.txt"),"w").write("\n".join(simple_list))
    open(os.path.join(args.save_dir,"q_list_init.txt"),"w").write("\n".join(q_list_init))

    # subset for YOLO
    simple_data = os.path.join(args.save_dir, 'simple_data')
    gt_map = {p:cls for p,cls in list_imgs(args.data)}
    make_subset_dataset(simple_list, simple_data, gt_map)

    # YOLO on CPU (classification/uncertainty filter)
    yolo_csv = os.path.join(args.save_dir,"yolo_simple_keep.csv")
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = ""
    cmd = [
        args.yolo_env_py, args.yolo_step2,
        "--simple_data", simple_data,
        "--mvtec_root", args.mvtec_root,
        "--weights", args.yolo_weights,
        "--save_csv", yolo_csv,
        "--smax_thr", str(args.yolo_smax_thr),
        "--margin_thr", str(args.yolo_margin_thr),
        "--ent_thr", str(args.yolo_ent_thr),
    ]
    t0=time.time()
    subprocess.run(cmd, check=True, env=env)
    t_yolo = time.time()-t0

    # read YOLO kept
    kept_rows=[]
    if os.path.exists(yolo_csv):
        with open(yolo_csv, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for t in r:
                pth = (t.get("path") or t.get("img") or t.get("file") or "").strip()
                if not pth: continue
                pth = os.path.realpath(pth)
                pr  = (t.get("pred") or t.get("label") or "").strip().lower()
                try: smax = float((t.get("smax")   or t.get("score") or "0").strip())
                except: smax = 0.0
                try: margin = float((t.get("margin") or "0").strip())
                except: margin = 0.0
                try: ent = float((t.get("ent") or "1").strip())
                except: ent = 1.0

                is_good = (pr=="good"   and smax>=args.yolo_smax_thr and margin>=args.yolo_margin_thr and ent<=args.yolo_ent_thr)
                is_def  = (pr=="defect" and smax>=args.yolo_smax_thr and margin>=max(args.yolo_margin_thr,0.08) and ent<=args.yolo_ent_thr)
                if not (is_good or (args.yolo_keep_bidir and is_def)):
                    continue
                kept_rows.append({"path": pth, "gt": gt_from_path(pth), "pred": pr})
    kept_paths=set([r["path"] for r in kept_rows])

    # Q list
    to_q_from_yolo = sorted(list(set(simple_list) - kept_paths))
    q_list_final = sorted(list(set(q_list_init) | set(to_q_from_yolo)))
    q_list_final = [p for p in q_list_final if p not in kept_paths]
    open(os.path.join(args.save_dir,"q_list_final.txt"),"w").write("\n".join(q_list_final))

    # build q subset and eval by Qwen + JSON for defects
    q_data = os.path.join(args.save_dir, 'q_subset_data')
    make_subset_dataset(q_list_final, q_data, gt_map)

    t0=time.time()
    q_rows, q_counts, q_json_map = qwen_eval_subset_with_thr_batched(
        base=args.qwen_base,
        adapter=args.qwen_adapter,
        data_root=q_data,
        thr=args.qwen_thr,
        device=args.qwen_device,
        prompt=args.qwen_prompt,
        batch_size=args.qwen_batch,
        resize=args.qwen_resize,
        load_4bit=bool(args.qwen_4bit),
        attn=args.qwen_attn,
        gen_json_if_defect=True
    )
    t_qwen = time.time()-t0

    # write Qwen JSON for defect images
    q_jsonl = os.path.join(args.save_dir, "qwen_defect_json.jsonl")
    with open(q_jsonl, "w", encoding="utf-8") as f:
        for p, j in q_json_map.items():
            f.write(json.dumps({"path": p, "json": j}, ensure_ascii=False) + "\n")

    # metrics
    y_m = metrics_from_rows([{"gt":r["gt"], "pred":r["pred"]} for r in kept_rows])
    q_m = metrics_from_rows([{"gt":r["gt"], "pred":r["pred"]} for r in q_rows])
    merged = [{"gt":r["gt"], "pred":r["pred"], "path":r["path"]} for r in kept_rows] + \
             [{"gt":r["gt"], "pred":r["pred"], "path":r["path"]} for r in q_rows]
    m_m = metrics_from_rows([{"gt":r["gt"], "pred":r["pred"]} for r in merged])

    # YOLO detection visualization for all images predicted as defect (Label==1)
    defect_paths_final = [r["path"] for r in merged if r["pred"]=="defect"]
    vis_dir = os.path.join(args.save_dir, "vis_yolo_det")
    if defect_paths_final:
        yolo_detect_and_draw(defect_paths_final, args.yolo_det_weights, vis_dir, device="cpu")

    N_all = len(list_imgs(args.data))
    q_share = 100.0 * q_m["N"] / max(N_all,1)
    total_parallel = t_complex + max(t_yolo, t_qwen)
    total_serial   = t_complex + t_yolo + t_qwen

    print("\n================ Scheme B (Complexity + Uncertainty fallback) ================")
    print(f"[TIME] complexity   = {t_complex:.1f}s")
    print(f"[TIME] YOLO (CPU)   = {t_yolo:.1f}s   for simple-list N={len(simple_list)} (keep {y_m['N']}, toQ {len(to_q_from_yolo)})")
    print(f"[TIME] Qwen (GPU)   = {t_qwen:.1f}s   for Q-list N={q_m['N']}")
    print(f"[TIME] total (parallel) = {total_parallel:.1f}s   avg={total_parallel/max(N_all,1):.3f}s/img")
    print(f"[TIME] total (serial)   = {total_serial:.1f}s    avg={total_serial/max(N_all,1):.3f}s/img")
    print(f"[ROUTING] Qwen share = {q_share:.2f}%   (YOLO keep {y_m['N']}, Qwen {q_m['N']}, Total {N_all})")

    print_metrics("YOLO-11s (simple branch, real cls)", y_m)
    print_metrics("Qwen (QLoRA, complex branch, thr-prob, batched)", q_m)
    print_metrics("Final (merged)", m_m)

    summary = {
        "time": {"complex": t_complex, "yolo": t_yolo, "qwen": t_qwen,
                 "parallel": total_parallel, "serial": total_serial},
        "routing": {"q_share_percent": q_share, "yolo_keep": y_m["N"], "qwen": q_m["N"], "total": N_all},
        "metrics": {"yolo": y_m, "qwen": q_m, "merged": m_m},
        "artifacts": {
            "qwen_defect_jsonl": q_jsonl,
            "yolo_det_vis_dir": vis_dir,
        },
        "params": {
            "init_q_ratio": args.init_q_ratio, "cpx_size": args.cpx_size,
            "yolo_smax_thr": args.yolo_smax_thr, "yolo_margin_thr": args.yolo_margin_thr, "yolo_ent_thr": args.yolo_ent_thr,
            "yolo_keep_bidir": args.yolo_keep_bidir,
            "qwen_thr": args.qwen_thr, "qwen_batch": args.qwen_batch, "qwen_resize": args.qwen_resize,
            "qwen_4bit": bool(args.qwen_4bit), "qwen_attn": args.qwen_attn
        }
    }
    json.dump(summary, open(os.path.join(args.save_dir,"summary.json"),"w"), indent=2, ensure_ascii=False)
    print(f"\n[FILES] summary : {os.path.join(args.save_dir,'summary.json')}")
    print(f"[FILES] qwen JSON (defect) : {q_jsonl}")
    print(f"[FILES] YOLO detections drawn to : {vis_dir}")
    print("Done.")

if __name__ == "__main__":
    main()
