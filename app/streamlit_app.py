# -*- coding: utf-8 -*-
"""건물 추출 데모: 항공영상 업로드 → 마스크 → 폴리곤 → GeoJSON 다운로드.

실행:  streamlit run app/streamlit_app.py
모델:  runs/unet_r34/best.pt (없으면 안내만 표시)
"""
import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.postprocess import extract, mask_to_polygons, polygons_to_mask  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
MODEL_PATH = Path("runs/unet_r34/best.pt")

st.set_page_config(page_title="Building Footprint Extractor", layout="wide")
st.title("항공영상 건물 추출")
st.caption("세그멘테이션 마스크를 폴리곤으로 벡터화하고 형상을 보정해 GeoJSON으로 내보냅니다.")


@st.cache_resource
def get_model():
    if not MODEL_PATH.exists():
        return None
    import torch  # noqa: F401
    from src.infer import load_model

    return load_model(MODEL_PATH)


with st.sidebar:
    st.header("후처리 설정")
    min_area = st.slider("최소 면적(px²)", 10, 500, 50, 10)
    simplify = st.slider("단순화 허용오차(px)", 0.0, 5.0, 1.5, 0.5)
    rect_iou = st.slider("직사각형 보정 IoU 임계값", 0.5, 0.95, 0.75, 0.05)
    thr = st.slider("마스크 임계값", 0.1, 0.9, 0.5, 0.05)

tab1, tab2 = st.tabs(["항공영상 추론", "마스크 후처리만"])

with tab1:
    up = st.file_uploader("항공영상 (png/jpg/tif)", type=["png", "jpg", "jpeg", "tif", "tiff"], key="img")
    model = get_model()
    if up is None:
        st.info("이미지를 올리면 모델 추론 → 벡터화까지 실행합니다.")
    elif model is None:
        st.warning("모델 가중치(runs/unet_r34/best.pt)가 없습니다. 학습 후 다시 실행하거나 '마스크 후처리만' 탭을 사용하세요.")
    else:
        from src.infer import predict_image

        image = np.array(Image.open(up).convert("RGB"))
        with st.spinner("추론 중..."):
            mask = predict_image(model, image, thr=thr)
        polys, gj = extract(mask, min_area=min_area, simplify_tol=simplify, rect_iou=rect_iou)

        c1, c2 = st.columns(2)
        c1.image(image, caption="입력 영상", use_container_width=True)
        overlay = image.copy()
        pm = polygons_to_mask(polys, mask.shape).astype(bool)
        overlay[pm] = (overlay[pm] * 0.4 + np.array([255, 80, 80]) * 0.6).astype(np.uint8)
        c2.image(overlay, caption=f"추출 결과 · 건물 {len(polys)}동", use_container_width=True)

        st.download_button("GeoJSON 다운로드", json.dumps(gj, ensure_ascii=False),
                           file_name="buildings.geojson", mime="application/geo+json")

with tab2:
    st.caption("모델 없이 후처리 단계만 시험: 0/255 이진 마스크 이미지를 올리세요.")
    upm = st.file_uploader("마스크 (png)", type=["png"], key="mask")
    if upm is not None:
        mask = np.array(Image.open(upm).convert("L"))
        raw_polys = mask_to_polygons(mask, min_area=min_area, simplify_tol=0.0)
        polys, gj = extract(mask, min_area=min_area, simplify_tol=simplify, rect_iou=rect_iou)

        c1, c2 = st.columns(2)
        c1.image(mask, caption=f"입력 마스크 · 원시 폴리곤 {len(raw_polys)}개", use_container_width=True)
        out = polygons_to_mask(polys, mask.shape) * 255
        total_pts_raw = sum(len(p.exterior.coords) for p in raw_polys)
        total_pts = sum(len(p.exterior.coords) for p in polys)
        c2.image(out, caption=f"후처리 결과 · 건물 {len(polys)}동 · 꼭짓점 {total_pts_raw}→{total_pts}",
                 use_container_width=True)

        st.download_button("GeoJSON 다운로드", json.dumps(gj, ensure_ascii=False),
                           file_name="buildings.geojson", mime="application/geo+json")
