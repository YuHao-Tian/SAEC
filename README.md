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
# Install dependencies
pip install -r requirements/requirements_qwen.txt


# YOLO (CPU)
python -m venv yoloenv && source yoloenv/bin/activate
# Install dependencies
pip install -r requirements/requirements_yolo.txt


# (Optional) LLaVA baseline
python -m venv llavaenv && source llavaenv/bin/activate
# Install dependencies
pip install -r requirements/requirements_llava.txt

```

## Dataset downloads and preparation
The experiments used two datasets: MVTec AD dataset(Download link:https://www.mvtec.com/company/research/datasets/mvtec-ad) and ksdd2 dataset(Download link:https://www.vicos.si/resources/kolektorsdd2/) 
我们把实验的数据集进行改动，把Both are converted to a binary format val/{good, defect} with a unified directory struc- ture and an approximately 1:1 class ratio.的同时，并不影响qlora的训练的数据集，不让数据被预先看见或污染（这块怎么说，你来，这句需要说嘛？）


/home/vipuser/qwenenv/bin/python /home/vipuser/build_mvtec_holdout1k.py \
  --src /home/vipuser/data/mvtec_anomaly_detection \
  --out /home/vipuser/data/mvtec_cls_holdout1k_v1 \
  --n-good 500 --n-defect 500 --seed 2025

[HOLDOUT1K READY]
good=467  defect=500  total=967
root: <Your ROOT>/data/mvtec_cls_holdout1k_v1
manifest: <Your ROOT>/data/mvtec_cls_holdout1k_v1/manifest.json

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
