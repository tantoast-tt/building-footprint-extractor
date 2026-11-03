# -*- coding: utf-8 -*-
"""학습된 모델을 타일 폴더(images/, masks/) 전체에 대해 평가.

Inria 검증셋이든 AI Hub 국내 위성 타일이든 같은 명령으로 IoU/F1을 잰다.

사용:
  python -m src.eval_tiles --model runs/unet_r34/best.pt --tiles data/aihub_val --batch 8
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import TileDataset
from .metrics import evaluate_pair


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tiles", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--thr", type=float, default=0.5)
    a = ap.parse_args()

    from .infer import load_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(a.model, device)
    ds = TileDataset(a.tiles)
    ld = DataLoader(ds, batch_size=a.batch, num_workers=2)
    print(f"타일 {len(ds)}개, device={device}")

    scores = []
    for i, (img, msk) in enumerate(ld):
        pred = torch.sigmoid(model(img.to(device))).cpu().numpy()
        scores.append(evaluate_pair(pred, msk.numpy(), thr=a.thr))
        if (i + 1) % 20 == 0:
            print(f"  {min((i+1)*a.batch, len(ds))}/{len(ds)}")

    agg = {k: float(np.mean([s[k] for s in scores])) for k in scores[0]}
    print(f"IoU {agg['iou']:.4f} | F1 {agg['f1']:.4f} | "
          f"Precision {agg['precision']:.4f} | Recall {agg['recall']:.4f}")


if __name__ == "__main__":
    main()
