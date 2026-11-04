# -*- coding: utf-8 -*-
"""학습된 모델로 이미지 1장 추론 → 마스크 + GeoJSON.

사용:
  python -m src.infer --model runs/unet_r34/best.pt --image sample.tif --out out/
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .postprocess import extract, save_geojson
from .tiling import read_image

Image.MAX_IMAGE_PIXELS = None


def load_model(ckpt_path, device="cpu"):
    from .train import build_model

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ckpt.get("arch", "unet"), ckpt.get("encoder", "resnet34"))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model


@torch.no_grad()
def predict_image(model, image: np.ndarray, device="cpu", tile=512, thr=0.5, progress=None) -> np.ndarray:
    """큰 이미지를 타일 단위로 추론해 이어 붙인 0/255 마스크 반환.

    progress: 콜백 f(done, total). 타일 하나 처리할 때마다 호출된다.
    """
    h, w = image.shape[:2]
    pad_h = (tile - h % tile) % tile
    pad_w = (tile - w % tile) % tile
    img = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    prob = np.zeros(img.shape[:2], dtype=np.float32)

    total = (img.shape[0] // tile) * (img.shape[1] // tile)
    done = 0
    for y in range(0, img.shape[0], tile):
        for x in range(0, img.shape[1], tile):
            patch = img[y : y + tile, x : x + tile]
            t = torch.from_numpy(patch.transpose(2, 0, 1)).float()[None] / 255.0
            p = torch.sigmoid(model(t.to(device)))[0, 0].cpu().numpy()
            prob[y : y + tile, x : x + tile] = p
            done += 1
            if progress is not None:
                progress(done, total)

    mask = (prob[:h, :w] > thr).astype(np.uint8) * 255
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--min-area", type=float, default=50.0)
    ap.add_argument("--simplify", type=float, default=1.5)
    ap.add_argument("--rect-iou", type=float, default=0.75)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    model = load_model(a.model, device)
    image = read_image(Path(a.image))
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    image = image[:, :, :3]

    mask = predict_image(model, image, device, thr=a.thr)
    Image.fromarray(mask).save(out / "mask.png")

    # GeoTIFF면 transform을 읽어 지리좌표 GeoJSON으로
    transform = None
    try:
        import rasterio

        with rasterio.open(a.image) as src:
            if src.transform and src.transform.a != 1.0:
                transform = src.transform
    except Exception:
        pass

    polys, _ = extract(mask, min_area=a.min_area, simplify_tol=a.simplify, rect_iou=a.rect_iou)
    save_geojson(polys, out / "buildings.geojson", transform)
    print(f"건물 {len(polys)}동 → {out/'mask.png'}, {out/'buildings.geojson'}")


if __name__ == "__main__":
    main()
