## ☁️ Edge–Cloud Collaboration Experiments

Run **YOLOv11s on CPU** as the *edge* branch and **QLoRA–Qwen2.5-VL-7B on GPU** as the *cloud* branch.  
Reported metrics: **Accuracy**, **Parallel wall time**, **GPU util avg (%)**, **GPU power avg (W)**, **Energy per correct (mWh)**, **CPU memory avg (GB)**.

---

### 🔷 SAEC (ours)

#### MVTec AD (1K v1 holdout)

```bash
DATA=/home/vipuser/data/mvtec_cls_holdout1k_v1
RUN=/home/vipuser/runs_eval/saec_mvtecad
PROFILE=/home/vipuser/profile.py
mkdir -p "$RUN"
N_IMG=$(find -L "$DATA/val" -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.bmp" -o -iname "*.tif" -o -iname "*.tiff" -o -iname "*.webp" \) | wc -l)
/home/vipuser/qwenenv/bin/python "$PROFILE"   --gpu 0 --interval 0.2 --images "$N_IMG"   --save "$RUN/prof_schemeB_$(date +%m%d_%H%M%S).json"   -- /home/vipuser/qwenenv/bin/python /home/vipuser/schemeB_1k_minrun_v4.py      --data "$DATA"      --save_dir "$RUN"      --yolo_env_py /home/vipuser/yoloenv/bin/python      --yolo_step2 /home/vipuser/step2_y11n_unsup_cls.py      --yolo_weights /home/vipuser/models/yolo11s-cls.pt      --mvtec_root /home/vipuser/data/mvtec_anomaly_detection      --qwen_base /home/vipuser/models/Qwen2.5-VL-7B-Instruct_weights      --qwen_adapter /home/vipuser/qlora_qwen25_mvtec      --init_q_ratio 0.30      --cpx_size 192      --yolo_smax_thr 0.55      --yolo_margin_thr 0.03      --yolo_ent_thr 0.95      --yolo_keep_bidir 1      --qwen_thr 0.26      --qwen_batch 24      --qwen_resize 512      --qwen_4bit 0      --qwen_attn flash2
```

#### KSDD2 (1:1, “all defects + equal good”)

```bash
DATA=/home/vipuser/data/ksdd2_1to1_allDef
RUN=/home/vipuser/runs_eval/saec_ksdd2
PROFILE=/home/vipuser/profile.py
mkdir -p "$RUN"
N_IMG=$(find -L "$DATA/val" -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.bmp" -o -iname "*.tif" -o -iname "*.tiff" -o -iname "*.webp" \) | wc -l)
/home/vipuser/qwenenv/bin/python "$PROFILE"   --gpu 0 --interval 0.2 --images "$N_IMG"   --save "$RUN/prof_schemeB_$(date +%m%d_%H%M%S).json"   -- /home/vipuser/qwenenv/bin/python /home/vipuser/schemeB_1k_minrun_v4.py      --data "$DATA"      --save_dir "$RUN"      --yolo_env_py /home/vipuser/yoloenv/bin/python      --yolo_step2 /home/vipuser/step2_y11n_unsup_cls.py      --yolo_weights /home/vipuser/models/yolo11s-cls.pt      --mvtec_root /home/vipuser/data/mvtec_anomaly_detection      --qwen_base /home/vipuser/models/Qwen2.5-VL-7B-Instruct_weights      --qwen_adapter /home/vipuser/qlora_qwen25_mvtec      --init_q_ratio 0.30      --cpx_size 192      --yolo_smax_thr 0.55      --yolo_margin_thr 0.03      --yolo_ent_thr 0.95      --yolo_keep_bidir 1      --qwen_thr 0.26      --qwen_batch 24      --qwen_resize 512      --qwen_4bit 0      --qwen_attn flash2
```

---

### 🔹 Qwen (base, zero-shot)

#### MVTec AD (1K v1 holdout)

```bash
PROFILE=/home/vipuser/profile.py
DATA=/home/vipuser/data/mvtec_cls_holdout1k_v1
RUN=/home/vipuser/runs_eval/qwen_mvtecad_$(date +%m%d_%H%M%S)
mkdir -p "$RUN"

N_IMG=$(find -L "$DATA/val" -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.bmp" -o -iname "*.tif" -o -iname "*.tiff" -o -iname "*.webp" \) | wc -l)

/home/vipuser/qwenenv/bin/python "$PROFILE"   --gpu $GPU --interval 0.2 --images "$N_IMG"   --save "$RUN/prof_qwen_base.json"   -- /home/vipuser/qwenenv/bin/python /home/vipuser/schemeB_1k_minrun_v4.py      --data "$DATA"      --save_dir "$RUN"      --yolo_env_py /home/vipuser/yoloenv/bin/python      --yolo_step2 /home/vipuser/step2_y11n_unsup_cls.py      --yolo_weights /home/vipuser/models/yolo11s-cls.pt      --mvtec_root /home/vipuser/data/mvtec_anomaly_detection      --qwen_base /home/vipuser/models/Qwen2.5-VL-7B-Instruct_weights      --qwen_adapter ""      --init_q_ratio 1.00      --cpx_size 192      --yolo_smax_thr 9.99 --yolo_margin_thr 9.99 --yolo_ent_thr 0.00      --yolo_keep_bidir 0      --qwen_thr 0.26      --qwen_batch 16      --qwen_resize 448      --qwen_4bit 0      --qwen_attn flash2
```

#### KSDD2 (1:1, “all defects + equal good”)

```bash
PROFILE=/home/vipuser/profile.py
DATA=/home/vipuser/data/ksdd2_1to1_allDef
RUN=/home/vipuser/runs_eval/qwen_ksdd2_$(date +%m%d_%H%M%S)
mkdir -p "$RUN"

N_IMG=$(find -L "$DATA/val" -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.bmp" -o -iname "*.tif" -o -iname "*.tiff" -o -iname "*.webp" \) | wc -l)

/home/vipuser/qwenenv/bin/python "$PROFILE"   --gpu $GPU --interval 0.2 --images "$N_IMG"   --save "$RUN/prof_qwen_base.json"   -- /home/vipuser/qwenenv/bin/python /home/vipuser/schemeB_1k_minrun_v4.py      --data "$DATA"      --save_dir "$RUN"      --yolo_env_py /home/vipuser/yoloenv/bin/python      --yolo_step2 /home/vipuser/step2_y11n_unsup_cls.py      --yolo_weights /home/vipuser/models/yolo11s-cls.pt      --mvtec_root /home/vipuser/data/mvtec_anomaly_detection      --qwen_base /home/vipuser/models/Qwen2.5-VL-7B-Instruct_weights      --qwen_adapter ""      --init_q_ratio 1.00      --cpx_size 192      --yolo_smax_thr 9.99 --yolo_margin_thr 9.99 --yolo_ent_thr 0.00      --yolo_keep_bidir 0      --qwen_thr 0.26      --qwen_batch 16      --qwen_resize 448      --qwen_4bit 0      --qwen_attn flash2
```

---

### 🔹 LLaVA-1.5-7B (zero-shot)

#### MVTec AD (1K v1 holdout)

```bash
PROFILE=/home/vipuser/profile.py
DATA=/home/vipuser/data/mvtec_cls_holdout1k_v1
RUN=/home/vipuser/runs_eval/llava_mvtecad_$(date +%m%d_%H%M%S)
mkdir -p "$RUN"

N_IMG=$(find -L "$DATA/val" -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.bmp" -o -iname "*.tif" -o -iname "*.tiff" -o -iname "*.webp" \) | wc -l)

/home/vipuser/qwenenv/bin/python "$PROFILE"   --gpu $GPU --interval 0.2 --images "$N_IMG"   --save "$RUN/prof_llava_u448_none.json"   -- /home/vipuser/qwenenv/bin/python /home/vipuser/eval_llava_base_mvtec.py      --base /home/vipuser/models/llava-1.5-7b-hf      --data "$DATA"      --resize 448      --max_new_tokens 2 --num_beams 1      --tta none      --save_csv "$RUN/probs.csv"
```

#### KSDD2 (1:1, “all defects + equal good”)

```bash
PROFILE=/home/vipuser/profile.py
DATA=/home/vipuser/data/ksdd2_1to1_allDef
RUN=/home/vipuser/runs_eval/llava_ksdd2_$(date +%m%d_%H%M%S)
mkdir -p "$RUN"

N_IMG=$(find -L "$DATA/val" -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.bmp" -o -iname "*.tif" -o -iname "*.tiff" -o -iname "*.webp" \) | wc -l)

/home/vipuser/qwenenv/bin/python "$PROFILE"   --gpu $GPU --interval 0.2 --images "$N_IMG"   --save "$RUN/prof_llava_u448_none.json"   -- /home/vipuser/qwenenv/bin/python /home/vipuser/eval_llava_base_mvtec.py      --base /home/vipuser/models/llava-1.5-7b-hf      --data "$DATA"      --resize 448      --max_new_tokens 2 --num_beams 1      --tta none      --save_csv "$RUN/probs.csv"
```
