#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4-bit QLoRA fine-tuning for Qwen2.5-VL-7B-Instruct on MVTec SFT JSONL.
- Transformers 4.55.3 / PEFT 0.13.x / Accelerate 0.34.x / bitsandbytes 0.43.x
- 数据格式：每行 {"messages":[...]}，其中 user 段含 {"type":"image","image":"file:///abs/path.png"}
"""

import os, re, json, argparse
from pathlib import Path
from typing import List, Dict, Any

import torch
from PIL import Image
from tqdm import tqdm

from transformers import (
    AutoProcessor,
    BitsAndBytesConfig, 
    TrainingArguments,
    Trainer,
)
from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# ---------- utils ----------

def _load_images_from_messages(messages) -> List[Image.Image]:
    imgs = []
    for turn in messages:
        for piece in turn.get("content", []):
            if isinstance(piece, dict) and piece.get("type") == "image":
                p = piece.get("image", "")
                if p.startswith("file://"):
                    p = p[7:]
                img = Image.open(p).convert("RGB")
                imgs.append(img)
    return imgs


# ---------- dataset ----------

class JsonlSFT(torch.utils.data.Dataset):
    """
    读取 SFT 的 messages（包含 assistant 标签），并构造一份 prompt_messages（去掉最后一条 assistant）。
    """
    def __init__(self, jsonl_path: str):
        self.samples: List[Dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                assert "messages" in obj, f"Bad row: {obj}"
                msgs = obj["messages"]
                assert msgs[-1]["role"] == "assistant", "last message must be assistant"
                self.samples.append({
                    "messages": msgs,
                    "prompt_messages": msgs[:-1],  # no assistant
                })

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


class QwenVLDataCollator:
    """
    - 用 processor 把 (texts_full, images) 打成 batch
    - 再用 (texts_prompt, images) 计算每条样本的 prompt 长度，mask 掉 prompt 段 labels=-100
    """
    def __init__(self, processor: AutoProcessor):
        self.processor = processor

    def __call__(self, feats: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts_full, texts_prompt, images_batch = [], [], []
        for ex in feats:
            msgs_full   = ex["messages"]
            msgs_prompt = ex["prompt_messages"]
            texts_full.append(self.processor.apply_chat_template(msgs_full, tokenize=False))
            texts_prompt.append(self.processor.apply_chat_template(msgs_prompt, tokenize=False, add_generation_prompt=True))
            images_batch.append(_load_images_from_messages(msgs_full))

        # tokenization (full)
        full_batch = self.processor(text=texts_full, images=images_batch, padding=True, return_tensors="pt")

        # tokenization (prompt only) 用于计算每条样本的有效长度
        prompt_batch = self.processor(text=texts_prompt, images=images_batch, padding=True, return_tensors="pt")
        prompt_len = prompt_batch["attention_mask"].sum(dim=1)  # [B]

        labels = full_batch["input_ids"].clone()
        for i, L in enumerate(prompt_len.tolist()):
            labels[i, :L] = -100  # 只训练 assistant 段

        full_batch["labels"] = labels
        return full_batch


# ---------- eval ----------

@torch.no_grad()
def evaluate_on_jsonl(model, processor, jsonl_path: str, device):
    ds = JsonlSFT(jsonl_path)
    model.eval()

    total = correct = 0
    n_good = n_def = cg = cd = 0

    for ex in tqdm(ds, desc="eval", ncols=100):
        messages = ex["prompt_messages"]  # 只用到 user 提示（含图）
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images = [_load_images_from_messages(messages)]
        batch = processor(text=[text], images=images, padding=True, return_tensors="pt").to(device)

        out = model.generate(**batch, max_new_tokens=8, do_sample=False, temperature=0.0)
        new_tokens = out[:, batch.input_ids.shape[-1]:]
        resp = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

        m = re.search(r"Label\s*[:：]\s*([01])", resp) or re.search(r"\b([01])\b", resp)
        pred = int(m.group(1)) if m else 1  # 保守判 defect

        # 真值来自 messages 的最后一条 assistant
        gt_text = ex["messages"][-1]["content"][0]["text"]
        gt = 0 if "Label: 0" in gt_text else 1

        total += 1
        ok = int(pred == gt)
        correct += ok
        if gt == 0:
            n_good += 1; cg += ok
        else:
            n_def += 1;  cd += ok

    acc = correct/total if total else 0.0
    acc_good = cg/n_good if n_good else 0.0
    acc_def  = cd/n_def if n_def else 0.0

    print("\n================= EVAL ACCURACY =================")
    print("Label mapping: good=0 (no defect), defect=1 (defect)")
    print(f"Total images: {total}  |  good={n_good}  defect={n_def}")
    print(f"Overall Accuracy: {acc*100:.2f}%  (Correct / Total = {correct} / {total})")
    print(f"Per-class Accuracy -> good: {acc_good*100:.2f}%   defect: {acc_def*100:.2f}%")
    print("=================================================\n")


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="e.g. /home/vipuser/models/Qwen2.5-VL-7B-Instruct_weights")
    ap.add_argument("--data",  required=True, help="dir with train.jsonl / val.jsonl")
    ap.add_argument("--out",   required=True, help="output dir")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs",     type=int, default=2)
    ap.add_argument("--ga",     type=int, default=8)
    ap.add_argument("--lr",     type=float, default=2e-4)
    ap.add_argument("--use-4bit", action="store_true")
    args = ap.parse_args()

    print(f"[load] {args.model} | 4bit={args.use_4bit}")

    quant_cfg = None
    if args.use_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quant_cfg,
    )
    processor = AutoProcessor.from_pretrained(args.model)

    # QLoRA
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_path = os.path.join(args.data, "train.jsonl")
    val_path   = os.path.join(args.data, "val.jsonl")
    assert os.path.isfile(train_path) and os.path.isfile(val_path), "train.jsonl/val.jsonl 不存在"

    train_ds = JsonlSFT(train_path)
    val_ds   = JsonlSFT(val_path)
    collator = QwenVLDataCollator(processor)

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=max(1, args.bs),
        gradient_accumulation_steps=args.ga,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=2,
        report_to=[],                
        remove_unused_columns=False,  
        tf32=True,                  
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,     
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(args.out) 

    # 手动 eval（生成式）
    evaluate_on_jsonl(trainer.model, processor, val_path, device=trainer.model.device)


if __name__ == "__main__":
    main()
