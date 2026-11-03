# -*- coding: utf-8 -*-
"""후처리 단계별 효과를 GT와 비교해 수치로 낸다.

단계: ① 원시 마스크 → ② 벡터화(단순화 없음) → ③ +Douglas-Peucker 단순화 → ④ +직사각형 보정
각 단계의 건물 수·꼭짓점 수·GT 대비 IoU/F1을 표와 CSV로 출력.

사용:
  python -m src.evaluate --pred out/mask.png --gt vienna1_gt.tif --out out/postprocess_eval.csv
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from .metrics import evaluate_pair
from .postprocess import mask_to_polygons, polygons_to_mask, refine_polygons
from .tiling import read_image


def n_vertices(polys):
    return int(sum(len(p.exterior.coords) - 1 for p in polys))


def stage_rows(pred: np.ndarray, gt: np.ndarray, min_area=50.0, simplify_tol=1.5, rect_iou=0.75):
    gt_bin = (gt > 127).astype(np.uint8)
    pred_bin = (pred > 127).astype(np.uint8)
    rows = []

    m = evaluate_pair(pred_bin, gt_bin)
    rows.append(["① 원시 마스크", "-", "-", f"{m['iou']:.4f}", f"{m['f1']:.4f}"])

    raw = mask_to_polygons(pred, min_area=min_area, simplify_tol=0.0)
    m = evaluate_pair(polygons_to_mask(raw, pred.shape), gt_bin)
    rows.append(["② 벡터화", len(raw), n_vertices(raw), f"{m['iou']:.4f}", f"{m['f1']:.4f}"])

    simp = mask_to_polygons(pred, min_area=min_area, simplify_tol=simplify_tol)
    m = evaluate_pair(polygons_to_mask(simp, pred.shape), gt_bin)
    rows.append([f"③ +단순화(tol={simplify_tol})", len(simp), n_vertices(simp), f"{m['iou']:.4f}", f"{m['f1']:.4f}"])

    rect = refine_polygons(simp, iou_thr=rect_iou)
    m = evaluate_pair(polygons_to_mask(rect, pred.shape), gt_bin)
    rows.append([f"④ +직사각형 보정(IoU≥{rect_iou})", len(rect), n_vertices(rect), f"{m['iou']:.4f}", f"{m['f1']:.4f}"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="모델 출력 마스크 (png)")
    ap.add_argument("--gt", required=True, help="정답 마스크 (tif/png)")
    ap.add_argument("--out", default="out/postprocess_eval.csv")
    ap.add_argument("--min-area", type=float, default=50.0)
    ap.add_argument("--simplify", type=float, default=1.5)
    ap.add_argument("--rect-iou", type=float, default=0.75)
    a = ap.parse_args()

    pred = read_image(Path(a.pred))
    gt = read_image(Path(a.gt))
    if pred.ndim == 3:
        pred = pred[:, :, 0]
    if gt.ndim == 3:
        gt = gt[:, :, 0]
    assert pred.shape == gt.shape, f"크기 불일치: pred {pred.shape} vs gt {gt.shape}"

    header = ["단계", "건물 수", "꼭짓점 수", "IoU(GT)", "F1(GT)"]
    rows = stage_rows(pred, gt, a.min_area, a.simplify, a.rect_iou)

    widths = [26, 8, 10, 9, 9]
    print("  ".join(str(h).ljust(w) for h, w in zip(header, widths)))
    for r in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
