#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, re, glob, argparse, math
from pathlib import Path
from PIL import Image

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info

def list_val_images(root):
    files=[]
    for sub in ("good","defect"):
        d = os.path.join(root,"val",sub)
        for ext in ("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff"):
            files += glob.glob(os.path.join(d,ext))
    return sorted(files)

def build_prompt(img):
    sys="You are an industrial visual inspector. Output strictly as required."
    task=("Task: Determine if the image has a defect.\n"
          "If NO defect, output exactly: 'Label: 0'\n"
          "If there is ANY defect, output exactly: 'Label: 1'\n"
          "No extra words.")
    messages=[
        {"role":"system","content":[{"type":"text","text":sys}]},
        {"role":"user","content":[{"type":"image","image":img},{"type":"text","text":task}]}
    ]
    return messages

def five_crops(img: Image.Image, size=448):
    w,h = img.size
    s = min(w,h)
    if s<size: img = img.resize((max(size,w), max(size,h)), Image.BICUBIC)
    w,h = img.size
    crops = []
    # tl,tr,bl,br,center
    boxes = [(0,0,size,size),
             (w-size,0,w,h if size>h else size),
             (0,h-size,size,h),
             (w-size,h-size,w,h),
             ((w-size)//2,(h-size)//2,(w+size)//2,(h+size)//2)]
    for x0,y0,x1,y1 in boxes:
        x0=max(0,x0);y0=max(0,y0);x1=min(w,x1);y1=min(h,y1)
        c=img.crop((x0,y0,x1,y1)).resize((size,size), Image.BICUBIC)
        crops.append(c)
    return crops

@torch.no_grad()
def score_option(model, processor, img, option_text:str):
    # 准备 prompt-only & full（用于定位答案 token）
    msgs_prompt = build_prompt(img)
    prompt_text = processor.apply_chat_template(msgs_prompt, tokenize=False, add_generation_prompt=True)

    msgs_full = msgs_prompt + [{"role":"assistant","content":[{"type":"text","text":option_text}]}]
    full_text  = processor.apply_chat_template(msgs_full, tokenize=False, add_generation_prompt=False)

    inputs_prompt = processor(text=[prompt_text], images=[img], return_tensors="pt").to(model.device)
    inputs_full   = processor(text=[full_text],   images=[img], return_tensors="pt").to(model.device)

    Lp = inputs_prompt.input_ids.shape[1]
    out = model(**inputs_full)
    logits = out.logits[:, Lp-1:-1, :]             # 预测答案各 token 的 logits
    target = inputs_full.input_ids[:, Lp:]         # 答案 tokens
    logprobs = torch.log_softmax(logits, dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return logprobs.sum().item()

def eval_dir(model_dir, adapter_dir, data_root, imgsz=448):
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    if adapter_dir and Path(adapter_dir).exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
        model.eval()
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)

    files = list_val_images(data_root)
    assert files, f"No images under {data_root}/val/{{good,defect}}"

    correct=total=0; cg=cd=ng=nd=0
    for f in files:
        img0 = Image.open(f).convert("RGB")
        crops = five_crops(img0, size=imgsz)

        s0 = sum(score_option(model, processor, c, "Label: 0") for c in crops) / len(crops)
        s1 = sum(score_option(model, processor, c, "Label: 1") for c in crops) / len(crops)
        pred = 1 if s1>s0 else 0
        gt   = 0 if Path(f).parent.name.lower()=="good" else 1

        total += 1
        ok = int(pred==gt); correct += ok
        if gt==0: ng+=1; cg+=ok
        else: nd+=1; cd+=ok

    acc = correct/total
    print("\n================  ZERO-SHOT / QLoRA (ranking+TTA)  ================")
    print("Label mapping: good=0 (no defect), defect=1 (defect)")
    print(f"Total images: {total}  |  good={ng}  defect={nd}")
    print(f"Overall Accuracy: {acc*100:.2f}%  (Correct / Total = {correct} / {total})")
    if ng: print(f"Per-class Accuracy -> good: {cg/ng*100:.2f}%  defect: {cd/nd*100:.2f}%")
    print("===================================================================\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   default="/home/vipuser/models/Qwen2.5-VL-7B-Instruct_weights")
    ap.add_argument("--adapter", default="")  # 留空=零样本；给 LoRA 目录=微调后
    ap.add_argument("--data",    default="/home/vipuser/data/mvtec_qwen_sft")  # 用第1步生成的 val
    ap.add_argument("--imgsz",   type=int, default=448)
    args = ap.parse_args()
    eval_dir(args.model, args.adapter, args.data, imgsz=args.imgsz)
