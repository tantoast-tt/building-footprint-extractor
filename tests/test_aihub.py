# -*- coding: utf-8 -*-
import numpy as np

from src.aihub import label_to_mask, parse_imcoords


def test_parse_imcoords():
    pts = parse_imcoords("10, 10, 100, 10, 100, 80, 10, 80")
    assert pts.shape == (4, 2)
    assert parse_imcoords("EMPTY") is None
    assert parse_imcoords("") is None
    assert parse_imcoords("1, 2, 3") is None  # 홀수 개


def test_label_to_mask():
    label = {"features": [
        {"properties": {"building_imcoords": "10, 10, 100, 10, 100, 80, 10, 80", "image_id": "a.png"}},
        {"properties": {"building_imcoords": "EMPTY", "image_id": "a.png"}},
    ]}
    m = label_to_mask(label, shape=(128, 128))
    assert m.dtype == np.uint8
    assert m[45, 50] == 255   # 사각형 내부
    assert m[5, 5] == 0       # 외부
    area = (m > 0).sum()
    assert abs(area - 90 * 70) < 400  # 90x70 사각형 근사
