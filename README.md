# SAEC: Scene-Aware Enhanced Edge-Cloud-Collaborative Industrial Vision Inspection with Multimodal LLM

## Summary
A minimal edge–cloud pipeline for binary industrial inspection (`good=0`, `defect=1`). A lightweight YOLO runs at the edge CPU, and a multimodal LLM (Qwen-2.5-VL-7B) runs in the cloud. This README mirrors your current structure and commands for quick upload.

## Experiment Setup
All experiments were conducted on a single NVIDIA A100 (40 GB) GPU (cloud) and an 8-core Intel Xeon Platinum 8575C CPU (edge).  
The system runs Ubuntu 22.04 with CUDA 12.4.  

Please adjust dependency versions according to your own environment.
## Environment & Quick Start
We recommend separate virtual environments to avoid dependency conflicts. Python 3.10+ is recommended.

```bash
# Qwen (GPU)
python -m venv qwenenv && source qwenenv/bin/activate
pip install -r requirements/requirements_qwen.txt

#Download Qwen-2.5-VL-7B-Instruct weights to a fixed path
mkdir -p /home/vipuser/models
pip install -U huggingface_hub  # in case it's missing
# If the model is gated, accept license on HF and set HF_TOKEN beforehand.
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct \
  --local-dir /home/vipuser/models/Qwen2.5-VL-7B-Instruct_weights \
  --local-dir-use-symlinks False --resume-download
deactivate


# YOLO (CPU)
python -m venv yoloenv && source yoloenv/bin/activate
pip install -r requirements/requirements_yolo.txt

# Download YOLO11s classification weights (pre-download so scripts can point to a fixed file)
python - <<'PY'
from ultralytics import YOLO
# This triggers an automatic download to the Ultralytics cache (~/.cache/ultralytics)
YOLO('yolo11s-cls.pt')
print('downloaded yolo11s-cls.pt to cache')
PY
# Move the cached weight to /home/vipuser/models for consistent referencing
mkdir -p /home/vipuser/models
find ~/.cache -name 'yolo11s-cls.pt' -print -quit | xargs -I{} cp {} /home/vipuser/models/yolo11s-cls.pt
deactivate


# (Optional) LLaVA baseline
python -m venv llavaenv && source llavaenv/bin/activate
pip install -r requirements/requirements_llava.txt

# (Optional) Download LLaVA 1.5 7B weights to a fixed path
mkdir -p /home/vipuser/models
huggingface-cli download liuhaotian/llava-v1.5-7b \
  --local-dir /home/vipuser/models/llava-1.5-7b-hf \
  --local-dir-use-symlinks False --resume-download
deactivate


```



## Dataset downloads and preparation

We use two datasets: **MVTec AD** (https://www.mvtec.com/company/research/datasets/mvtec-ad) and **KolektorSDD2 (KSDD2)** (https://www.vicos.si/resources/kolektorsdd2/).

### MVTec (evaluation holdout)

MVTec is converted into a **binary holdout** with a unified directory structure and an approximately 1:1 class ratio:
  
<OUT>/  
  val/  
    good/  
    defect/  
  manifest.json  
   
The holdout is built **only from the official `test/` split**, ensuring no overlap with the QLoRA SFT training data.

**Build the holdout:**
```bash
/home/vipuser/qwenenv/bin/python /home/vipuser/build_mvtec_holdout1k.py \
  --src /home/vipuser/data/mvtec_anomaly_detection \
  --out /home/vipuser/data/mvtec_cls_holdout1k_v1 \
  --n-good 500 --n-defect 500 --seed 2025  
```
### KSDD2 (direct RAW → balanced 1:1)

Download and unzip **KSDD2** to `/home/vipuser/data/KSDD2_raw/`.
Build a **balanced 1:1 subset** directly from RAW (all defects + equal random goods):
```bash
/home/vipuser/qwenenv/bin/python build_ksdd2_1to1_from_raw.py
```
This creates:  
data/ksdd2_1to1_allDef/val/{good, defect}  
data/ksdd2_1to1_allDef/manifest.json  
runs_eval/splits/ksdd2_1to1_good.txt  
runs_eval/splits/ksdd2_1to1_defect.txt  
  
## 4bit Qlora Fine-tuning on Qwen-2.5L-VL

## Edge-Cloud Collaboration

**1) Data preparation script (train/val JSONL): `prep_mvtec_qwen_sft.py`**

```bash
/home/vipuser/qwenenv/bin/python /home/vipuser/prep_mvtec_sft.py   --src /home/vipuser/data/mvtec_anomaly_detection   --out /home/vipuser/data/mvtec_qwen_sft   --defect-train-ratio 0.5
# The output prints train/val counts, e.g., [done] train=xxxx val=yyyy
```

**2) QLoRA training script (attention + vision projector only): `train_qwen25_qlora.py`**  
4-bit QLoRA training

```bash
/home/vipuser/qwenenv/bin/python /home/vipuser/train_qwen25_qlora.py   --model /home/vipuser/models/Qwen2.5-VL-7B-Instruct_weights   --data  /home/vipuser/data/mvtec_qwen_sft   --out   /home/vipuser/qlora_qwen25_mvtec   --epochs 3 --bs 2 --ga 8 --lr 2e-4 --use-4bit
```

**3) Evaluation script: ranking + multi-crop TTA: `eval_qwen25_rank_tta.py`**  
Evaluation (load QLoRA adapter):

```bash
python /home/vipuser/eval_qwen25_rank_tta.py   --model /home/vipuser/models/Qwen-2.5-VL-7B-Instruct_weights   --adapter /home/vipuser/qlora_qwen25_mvtec   --data /home/vipuser/data/mvtec_qwen_sft --imgsz 448
```
