# -*- coding: utf-8 -*-
"""U-Net(resnet34) 건물 세그멘테이션 학습. 로컬 GPU/CPU·Colab 공용.

사용:
  python -m src.train --tiles data/tiles --epochs 15 --batch 8 --out runs/unet_r34
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import TileDataset, split_files
from .metrics import evaluate_pair


def get_transforms(train: bool):
    import albumentations as A

    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
        ])
    return None


def build_model(arch="unet", encoder="resnet34"):
    import segmentation_models_pytorch as smp

    if arch == "unet":
        return smp.Unet(encoder_name=encoder, encoder_weights="imagenet", in_channels=3, classes=1)
    if arch == "deeplabv3plus":
        return smp.DeepLabV3Plus(encoder_name=encoder, encoder_weights="imagenet", in_channels=3, classes=1)
    raise ValueError(arch)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    scores = []
    for img, msk in loader:
        img = img.to(device)
        pred = torch.sigmoid(model(img)).cpu().numpy()
        scores.append(evaluate_pair(pred, msk.numpy()))
    keys = scores[0].keys()
    return {k: float(np.mean([s[k] for s in scores])) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", required=True)
    ap.add_argument("--out", default="runs/unet_r34")
    ap.add_argument("--arch", default="unet", choices=["unet", "deeplabv3plus"])
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    import segmentation_models_pytorch as smp

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    train_files, val_files = split_files(a.tiles, a.val_ratio)
    print(f"train {len(train_files)} / val {len(val_files)} tiles, device={device}")

    train_ds = TileDataset(a.tiles, train_files, transform=get_transforms(True))
    val_ds = TileDataset(a.tiles, val_files)
    train_ld = DataLoader(train_ds, batch_size=a.batch, shuffle=True, num_workers=a.workers, pin_memory=True)
    val_ld = DataLoader(val_ds, batch_size=a.batch, num_workers=a.workers, pin_memory=True)

    model = build_model(a.arch, a.encoder).to(device)
    dice = smp.losses.DiceLoss(mode="binary")
    bce = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    best_iou = 0.0
    log_path = out / "log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_iou", "val_f1", "val_precision", "val_recall", "sec"])

    for epoch in range(1, a.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for img, msk in train_ld:
            img, msk = img.to(device), msk.to(device)
            opt.zero_grad()
            logit = model(img)
            loss = dice(logit, msk) + bce(logit, msk)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()

        m = validate(model, val_ld, device)
        sec = time.time() - t0
        print(f"[{epoch:02d}/{a.epochs}] loss {np.mean(losses):.4f} | IoU {m['iou']:.4f} F1 {m['f1']:.4f} | {sec:.0f}s")
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{np.mean(losses):.4f}", f"{m['iou']:.4f}", f"{m['f1']:.4f}",
                                    f"{m['precision']:.4f}", f"{m['recall']:.4f}", f"{sec:.0f}"])
        if m["iou"] > best_iou:
            best_iou = m["iou"]
            torch.save({"model": model.state_dict(), "arch": a.arch, "encoder": a.encoder,
                        "val_iou": best_iou, "epoch": epoch}, out / "best.pt")
            print(f"  ↳ best 갱신 (IoU {best_iou:.4f}) → {out/'best.pt'}")

    print(f"완료. best IoU {best_iou:.4f}, 로그 {log_path}")


if __name__ == "__main__":
    main()
