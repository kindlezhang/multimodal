import os
import math
import random
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from clip import clip  # openai/CLIP

# ---------- Config ----------
CSV_PATH = "./CLIP-main/data/train.csv"   # image_path, report
IMAGE_ROOT = "/Users/kindle/Desktop/file/project/LLM-with-mixed-type-data/data/raw/mimic-cxr-jpg/"       # if your csv has relative paths, set root
MODEL_NAME = "ViT-B/32"       # or "ViT-B/16"
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 32               # increase if you have more memory
EPOCHS = 5
LR = 2e-5
WEIGHT_DECAY = 1e-6
LOGIT_SCALE_INIT = 100.0      # common initial scale
SAVE_DIR = "./CLIP-main/checkpoints"
NUM_WORKERS = 0
FREEZE_CLIP = False      # True: freeze backbone, only tune small heads. False: tune full CLIP
# ----------------------------

os.makedirs(SAVE_DIR, exist_ok=True)
torch.manual_seed(42)
random.seed(42)

# ---------- Dataset ----------
class ImageTextDataset(Dataset):
    def __init__(self, csv_path, preprocess, image_root=""):
        df = pd.read_csv(
            csv_path,
            usecols=[0, 1], 
            header=None,
            names=["image", "text"],
            dtype={"image": str, "text": str},  # 强制为字符串
            keep_default_na=False,  # 防止空值变 NaN
        )

        df = df[~df["image"].isna()]              # 去掉空
        df = df[~df["image"].str.endswith('.0')]

        # 拼接完整路径
        if image_root:
            df["full_path"] = df["image"].apply(lambda x: os.path.join(image_root, x) if not os.path.isabs(x) else x)
        else:
            df["full_path"] = df["image"]

        # 过滤掉不存在的文件
        mask_exists = df["full_path"].apply(os.path.exists)
        skipped = df.loc[~mask_exists, "full_path"].tolist()
        if skipped:
            print(f"Skipped {len(skipped)} missing images, e.g.: {skipped[:5]}")
        df = df[mask_exists].reset_index(drop=True)

        self.df = df
        self.preprocess = preprocess

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['full_path']
        image = Image.open(image_path).convert("RGB")
        image = self.preprocess(image)
        image = image.float()
        text = str(row['text'])
        return image, text

# ---------- Collate (tokenize texts in batch) ----------
def collate_fn(batch):
    images, texts = zip(*batch)
    images = torch.stack(images, dim=0)
    # tokenize later on device
    return images, list(texts)

# ---------- Load model ----------
model, preprocess = clip.load(MODEL_NAME, device=DEVICE, jit=False)  # non-jit for training
model = model.to(DEVICE).float()

# Optionally re-init logit_scale to a value (CLIP uses learnable logit_scale)
with torch.no_grad():
    try:
        model.logit_scale.data.fill_(math.log(LOGIT_SCALE_INIT))
    except Exception:
        pass

# Freeze if requested
if FREEZE_CLIP:
    for p in model.parameters():
        p.requires_grad = False
    # still allow logit_scale learning
    try:
        model.logit_scale.requires_grad = True
    except Exception:
        pass

# If you want to fine-tune only text projection or add small adapter, modify here.

# ---------- DataLoader ----------
dataset = ImageTextDataset(CSV_PATH, preprocess, IMAGE_ROOT)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                        collate_fn=collate_fn, pin_memory=True)

# ---------- Optimizer ----------
# decide which params to train
trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)

# ---------- Training loop ----------
def train_one_epoch(epoch):
    model.train()
    total_loss = 0.0
    steps = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for images, texts in pbar:
        # move images
        images = images.to(DEVICE)
        # tokenize texts on device
        text_tokens = clip.tokenize(texts).to(DEVICE)  # shape [B, 77]
        # forward
        image_features = model.encode_image(images).float()   # [B, D]
        text_features = model.encode_text(text_tokens).float() # [B, D]
        
        print("Image features dtype:", image_features.dtype)
        print("Text features dtype:", text_features.dtype)

        # normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # logits (image x text^T) scaled by logit_scale
        logit_scale = model.logit_scale.exp().float()
        print("Logit scale dtype:", logit_scale.dtype)

        logits_per_image = logit_scale * image_features @ text_features.t().float() # [B, B]
        print("Logits dtype:", logits_per_image.dtype)

        logits_per_text = logits_per_image.t()

        labels = torch.arange(len(images), device=DEVICE).long()
        print("Labels dtype:", labels.dtype)

        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2
        print("Loss dtype:", loss.dtype)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1
        pbar.set_postfix(loss=total_loss / steps)

    return total_loss / steps if steps else 0.0

# ---------- Simple eval: Recall@1 on in-batch (proxy) ----------
@torch.no_grad()
def compute_inbatch_recall():
    model.eval()
    all_r1 = []
    for images, texts in dataloader:
        images = images.to(DEVICE)
        text_tokens = clip.tokenize(texts).to(DEVICE)
        image_features = model.encode_image(images)
        text_features = model.encode_text(text_tokens)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = (model.logit_scale.exp() * image_features @ text_features.t())
        # retrieval: for each image, top-1 text index
        top1 = logits.argmax(dim=1)
        labels = torch.arange(len(images), device=DEVICE)
        r1 = (top1 == labels).float().mean().item()
        all_r1.append(r1)
    return float(sum(all_r1) / len(all_r1)) if all_r1 else 0.0

# ---------- Main ----------

if __name__ == "__main__":

    best_loss = 1e9
    for epoch in range(EPOCHS):
        avg_loss = train_one_epoch(epoch)
        r1 = compute_inbatch_recall()
        print(f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}, inbatch_R@1={r1:.4f}")
        # save checkpoint
        ckpt = {
            "epoch": epoch+1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict()
        }
        torch.save(ckpt, os.path.join(SAVE_DIR, f"clip_med_epoch{epoch+1}.pt"))

    