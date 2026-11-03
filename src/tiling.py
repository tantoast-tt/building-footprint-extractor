# -*- coding: utf-8 -*-
"""대형 항공영상(tif/png)을 학습용 타일로 분할.

Inria Aerial Image Labeling 기준:
  images/  austin1.tif ... (5000x5000, RGB)
  gt/      austin1.tif ... (5000x5000, 0/255 건물 마스크)

사용:
  python -m src.tiling --images data/raw/images --masks data/raw/gt \
      --out data/tiles --tile 512 --stride 512 --min-fg 0.01
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # 5000x5000 tif 경고 해제


def read_image(path: Path) -> np.ndarray:
    """tif/png/jpg를 ndarray로. rasterio가 있으면 GeoTIFF도 지원."""
    try:
        import rasterio

        with rasterio.open(path) as src:
            arr = src.read()  # (C, H, W)
            arr = np.transpose(arr, (1, 2, 0))
            if arr.shape[2] == 1:
                arr = arr[:, :, 0]
            return arr
    except Exception:
        return np.array(Image.open(path))


def tile_pair(
    image: np.ndarray,
    mask: np.ndarray,
    tile: int = 512,
    stride: int = 512,
    min_fg: float = 0.0,
):
    """(이미지, 마스크) 쌍을 타일로 자른다. min_fg: 건물 픽셀 비율 하한(미만이면 버림)."""
    h, w = image.shape[:2]
    out = []
    for y in range(0, h - tile + 1, stride):
        for x in range(0, w - tile + 1, stride):
            img_t = image[y : y + tile, x : x + tile]
            msk_t = mask[y : y + tile, x : x + tile]
            fg = (msk_t > 127).mean()
            if fg < min_fg:
                continue
            out.append((x, y, img_t, msk_t))
    return out


def run(images_dir, masks_dir, out_dir, tile=512, stride=512, min_fg=0.0, limit=None):
    images_dir, masks_dir, out_dir = Path(images_dir), Path(masks_dir), Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in {".tif", ".tiff", ".png", ".jpg"})
    if limit:
        paths = paths[:limit]
    n = 0
    for p in paths:
        mp = masks_dir / p.name
        if not mp.exists():
            print(f"[skip] 마스크 없음: {p.name}")
            continue
        img = read_image(p)
        msk = read_image(mp)
        if msk.ndim == 3:
            msk = msk[:, :, 0]
        for x, y, img_t, msk_t in tile_pair(img, msk, tile, stride, min_fg):
            name = f"{p.stem}_{y}_{x}.png"
            Image.fromarray(img_t).save(out_dir / "images" / name)
            Image.fromarray((msk_t > 127).astype(np.uint8) * 255).save(out_dir / "masks" / name)
            n += 1
        print(f"[done] {p.name} → 누적 {n} 타일")
    print(f"총 {n} 타일 생성 → {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--min-fg", type=float, default=0.0, help="건물 픽셀 비율 하한 (예: 0.01)")
    ap.add_argument("--limit", type=int, default=None, help="처리할 원본 이미지 수 제한")
    a = ap.parse_args()
    run(a.images, a.masks, a.out, a.tile, a.stride, a.min_fg, a.limit)
