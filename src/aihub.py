# -*- coding: utf-8 -*-
"""AI Hub '위성영상 객체판독' 건물 데이터 → 학습/평가용 타일 변환.

원본 구성(zip 그대로 지원):
  *_images.zip : BLDxxxxx_PS3_...png (1024x1024 RGB) + PS4 tif + kml
  *_labels.zip : BLDxxxxx_PS3_...json (GeoJSON, properties.building_imcoords 에
                 "x1, y1, x2, y2, ..." 픽셀좌표 폴리곤)

사용:
  python -m src.aihub \
      --images-zip <validate_buildings_images.zip> \
      --labels-zip <validate_buildings_labels.zip> \
      --out data/aihub_val --tile 512 --limit 50
"""
import argparse
import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .tiling import tile_pair


def parse_imcoords(s: str):
    """'x1, y1, x2, y2, ...' → (N,2) float ndarray. EMPTY/비정상은 None."""
    if not s or s.strip().upper() == "EMPTY":
        return None
    try:
        vals = [float(v) for v in s.replace(";", ",").split(",") if v.strip()]
    except ValueError:
        return None
    if len(vals) < 6 or len(vals) % 2 != 0:
        return None
    return np.array(vals, dtype=np.float64).reshape(-1, 2)


def label_to_mask(label: dict, shape=(1024, 1024)) -> np.ndarray:
    """라벨 JSON(dict)의 building_imcoords들을 0/255 마스크로 래스터화."""
    mask = np.zeros(shape, dtype=np.uint8)
    for feat in label.get("features", []):
        pts = parse_imcoords(feat.get("properties", {}).get("building_imcoords", ""))
        if pts is None:
            continue
        cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 255)
    return mask


def iter_pairs(images_zip: str, labels_zip: str):
    """(이름, 이미지 ndarray, 마스크 ndarray)를 차례로 낸다. zip을 풀지 않고 읽음."""
    with zipfile.ZipFile(labels_zip) as lz, zipfile.ZipFile(images_zip) as iz:
        image_names = {Path(n).name: n for n in iz.namelist() if n.lower().endswith(".png")}
        for ln in sorted(lz.namelist()):
            if not ln.lower().endswith(".json"):
                continue
            label = json.loads(lz.read(ln).decode("utf-8"))
            img_name = None
            for feat in label.get("features", []):
                img_name = feat.get("properties", {}).get("image_id")
                if img_name:
                    break
            if img_name is None:  # 라벨에 image_id가 없으면 파일명으로 추정
                img_name = Path(ln).with_suffix(".png").name
            if img_name not in image_names:
                print(f"[skip] 이미지 없음: {img_name}")
                continue
            img = np.array(Image.open(io.BytesIO(iz.read(image_names[img_name]))).convert("RGB"))
            mask = label_to_mask(label, shape=img.shape[:2])
            yield Path(ln).stem, img, mask


def run(images_zip, labels_zip, out_dir, tile=512, stride=512, min_fg=0.0, limit=None):
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    n_img, n_tile = 0, 0
    for name, img, mask in iter_pairs(images_zip, labels_zip):
        for x, y, img_t, msk_t in tile_pair(img, mask, tile, stride, min_fg):
            tname = f"{name}_{y}_{x}.png"
            Image.fromarray(img_t).save(out / "images" / tname)
            Image.fromarray(msk_t).save(out / "masks" / tname)
            n_tile += 1
        n_img += 1
        if n_img % 20 == 0:
            print(f"[{n_img}] 누적 {n_tile} 타일")
        if limit and n_img >= limit:
            break
    print(f"원본 {n_img}장 → 타일 {n_tile}개 → {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-zip", required=True)
    ap.add_argument("--labels-zip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--min-fg", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=None, help="변환할 원본 이미지 수 제한")
    a = ap.parse_args()
    run(a.images_zip, a.labels_zip, a.out, a.tile, a.stride, a.min_fg, a.limit)
