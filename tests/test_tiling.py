# -*- coding: utf-8 -*-
import numpy as np

from src.tiling import tile_pair


def test_tile_count_and_shape():
    img = np.zeros((1024, 1024, 3), dtype=np.uint8)
    msk = np.zeros((1024, 1024), dtype=np.uint8)
    msk[0:600, 0:600] = 255
    tiles = tile_pair(img, msk, tile=512, stride=512, min_fg=0.0)
    assert len(tiles) == 4
    x, y, img_t, msk_t = tiles[0]
    assert img_t.shape == (512, 512, 3)
    assert msk_t.shape == (512, 512)


def test_min_fg_filter():
    img = np.zeros((1024, 1024, 3), dtype=np.uint8)
    msk = np.zeros((1024, 1024), dtype=np.uint8)
    msk[0:512, 0:512] = 255  # 좌상단 타일만 건물 100%
    tiles = tile_pair(img, msk, tile=512, stride=512, min_fg=0.5)
    assert len(tiles) == 1
    assert tiles[0][:2] == (0, 0)
