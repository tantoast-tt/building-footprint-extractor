# -*- coding: utf-8 -*-
import numpy as np
import pytest

from src.metrics import evaluate_pair, iou_score
from src.postprocess import extract, mask_to_polygons, polygons_to_mask, refine_polygons, to_geojson


def synthetic_mask():
    """건물 2동(사각형) + 노이즈 점 1개가 있는 512x512 마스크."""
    m = np.zeros((512, 512), dtype=np.uint8)
    m[50:150, 60:200] = 255      # 건물 1: 100x140
    m[300:420, 250:330] = 255    # 건물 2: 120x80
    m[480:483, 480:483] = 255    # 노이즈: 3x3 = 9px
    return m


def test_polygon_count_and_min_area_filter():
    polys = mask_to_polygons(synthetic_mask(), min_area=50)
    assert len(polys) == 2  # 노이즈(9px)는 걸러짐


def test_noise_kept_when_min_area_zero():
    # clean_mask의 open 연산이 3x3 노이즈를 지울 수 있으므로 4x4 이상으로 확인
    m = synthetic_mask()
    m[470:478, 470:478] = 255
    polys = mask_to_polygons(m, min_area=0)
    assert len(polys) >= 3


def test_rectangularize_recovers_rectangle():
    polys = mask_to_polygons(synthetic_mask(), min_area=50, simplify_tol=2.0)
    refined = refine_polygons(polys, iou_thr=0.75)
    # 직사각형 입력이므로 보정 후 꼭짓점 5개(닫힌 사각형)여야 함
    for p in refined:
        assert len(p.exterior.coords) == 5


def test_roundtrip_iou():
    m = synthetic_mask()
    polys, _ = extract(m, min_area=50, simplify_tol=1.5, rect_iou=0.75)
    recon = polygons_to_mask(polys, m.shape)
    assert iou_score(recon, m // 255) > 0.93


def test_geojson_structure():
    polys, gj = extract(synthetic_mask(), min_area=50)
    assert gj["type"] == "FeatureCollection"
    assert len(gj["features"]) == len(polys) == 2
    f = gj["features"][0]
    assert f["geometry"]["type"] == "Polygon"
    assert "area" in f["properties"]


def test_metrics_sanity():
    a = np.zeros((10, 10)); a[:5] = 1
    m = evaluate_pair(a, a)
    assert m["iou"] == pytest.approx(1.0, abs=1e-5)
    assert m["f1"] == pytest.approx(1.0, abs=1e-5)
    b = np.zeros((10, 10))
    assert iou_score(a, b) < 0.01
