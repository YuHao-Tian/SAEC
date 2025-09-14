# SAEC:Scene-Aware Enhanced Edge-Cloud-Collaborative Industrial Vision Inspection with Multimodal LLM
## Summary

## Experiment Setup
All experiments are conducted on a single NVIDIA A100
(40 GB) GPU (cloud) and an 8-core Intel Xeon Platinum
8575C CPU (edge).
We recommand Qlora, Llava and Yolo 11s are stored in separate virtual enviroments to avoid conflicts.
## 4bit Qlora Fine-tuning on Qwen-2.5L-VL

## Edge-Cloud Collaboration
1) 数据准备脚本（train/val JSONL）
/home/vipuser/qwenenv/bin/python /home/vipuser/prep_mvtec_sft.py \
  --src /home/vipuser/data/mvtec_anomaly_detection \
  --out /home/vipuser/data/mvtec_qwen_sft \
  --defect-train-ratio 0.5
输出会显示 train/val 条数，例如 [done] train=xxxx val=yyyy
2) QLoRA 训练脚本（只训注意力 + 视觉投影）
4bit QLoRA 训练
/home/vipuser/qwenenv/bin/python /home/vipuser/train_qwen25_qlora.py \
  --model /home/vipuser/models/Qwen2.5-VL-7B-Instruct_weights \
  --data  /home/vipuser/data/mvtec_qwen_sft \
  --out   /home/vipuser/qlora_qwen25_mvtec \
  --epochs 3 --bs 2 --ga 8 --lr 2e-4 --use-4bit
3) 评测脚本：排名式 + 多裁剪 TTA
评测（加载 QLoRA 适配器）：
python /home/vipuser/eval_qwen25_rank_tta.py \
  --model /home/vipuser/models/Qwen2.5-VL-7B-Instruct_weights \
  --adapter /home/vipuser/qlora_qwen25_mvtec \
  --data /home/vipuser/data/mvtec_qwen_sft --imgsz 448
