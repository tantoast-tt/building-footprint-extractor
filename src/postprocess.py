# -*- coding: utf-8 -*-
"""세그멘테이션 마스크 → 건물 폴리곤 → GeoJSON.

파이프라인:
  1) 모폴로지로 노이즈 제거
  2) 윤곽 추출(cv2) → shapely Polygon
  3) 최소 면적 필터
  4) Douglas-Peucker 단순화
  5) 직사각형 근사: 최소회전사각형과의 IoU가 높으면 사각형으로 교체(형상 보정)
  6) GeoJSON 출력 (GeoTIFF transform이 있으면 지리좌표, 없으면 픽셀좌표)
"""
import json

import cv2
import numpy as np
from shapely.geometry import Polygon, mapping


def clean_mask(mask: np.ndarray, open_ksize: int = 3) -> np.ndarray:
    """0/1 또는 0/255 마스크를 받아 노이즈 제거한 0/255 마스크 반환."""
    m = (np.asarray(mask) > 127).astype(np.uint8) if mask.max() > 1 else np.asarray(mask).astype(np.uint8)
    if open_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (open_ksize, open_ksize))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return m * 255


def mask_to_polygons(mask: np.ndarray, min_area: float = 50.0, simplify_tol: float = 1.5):
    """마스크에서 외곽 폴리곤 목록 추출(shapely Polygon, 픽셀좌표)."""
    m = clean_mask(mask)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if len(c) < 3:
            continue
        pts = c.reshape(-1, 2).astype(float)
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area < min_area:
            continue
        if simplify_tol > 0:
            poly = poly.simplify(simplify_tol, preserve_topology=True)
        if poly.is_empty or poly.area < min_area:
            continue
        if poly.geom_type == "MultiPolygon":
            polys.extend([g for g in poly.geoms if g.area >= min_area])
        else:
            polys.append(poly)
    return polys


def rectangularize(poly: Polygon, iou_thr: float = 0.75) -> Polygon:
    """최소회전사각형과의 IoU가 iou_thr 이상이면 사각형으로 교체(직각 형상 보정)."""
    pts = np.array(poly.exterior.coords, dtype=np.float32)
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)
    rect_poly = Polygon(box)
    if not rect_poly.is_valid or rect_poly.is_empty:
        return poly
    inter = poly.intersection(rect_poly).area
    union = poly.union(rect_poly).area
    if union <= 0:
        return poly
    return rect_poly if inter / union >= iou_thr else poly


def refine_polygons(polys, iou_thr: float = 0.75):
    return [rectangularize(p, iou_thr) for p in polys]


def polygons_to_mask(polys, shape) -> np.ndarray:
    """폴리곤 목록을 다시 마스크로(후처리 전후 IoU 비교용). 반환 0/1."""
    m = np.zeros(shape[:2], dtype=np.uint8)
    for p in polys:
        pts = np.array(p.exterior.coords, dtype=np.int32)
        cv2.fillPoly(m, [pts], 1)
    return m


def pixel_to_geo(polys, transform):
    """rasterio Affine transform으로 픽셀좌표 → 지리좌표 변환."""
    out = []
    for p in polys:
        coords = [transform * (x, y) for x, y in p.exterior.coords]
        out.append(Polygon(coords))
    return out


def to_geojson(polys, transform=None, properties=None) -> dict:
    """FeatureCollection dict 생성. transform이 있으면 지리좌표로 변환."""
    if transform is not None:
        polys = pixel_to_geo(polys, transform)
    feats = []
    for i, p in enumerate(polys):
        props = {"id": i, "area": round(p.area, 2)}
        if properties:
            props.update(properties)
        feats.append({"type": "Feature", "geometry": mapping(p), "properties": props})
    return {"type": "FeatureCollection", "features": feats}


def save_geojson(polys, path, transform=None):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_geojson(polys, transform), f, ensure_ascii=False)


def extract(mask: np.ndarray, min_area=50.0, simplify_tol=1.5, rect_iou=0.75, transform=None):
    """마스크 → (후처리 폴리곤, GeoJSON dict). 한 번에 실행하는 편의 함수."""
    polys = mask_to_polygons(mask, min_area=min_area, simplify_tol=simplify_tol)
    polys = refine_polygons(polys, iou_thr=rect_iou)
    return polys, to_geojson(polys, transform)
