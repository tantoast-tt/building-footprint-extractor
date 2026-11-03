# Building Footprint Extractor

항공영상에서 건물을 추출하는 파이프라인. 세그멘테이션 모델(U-Net)로 건물 마스크를 만들고,
후처리로 폴리곤 벡터화·형상 보정을 거쳐 GeoJSON으로 내보낸다.

```
항공영상 → 타일 분할 → U-Net 추론 → 마스크 정제 → 폴리곤화 → 단순화·직각 보정 → GeoJSON
```

## 구성

| 경로 | 내용 |
|---|---|
| `src/tiling.py` | 대형 영상(5000×5000 tif)을 512px 학습 타일로 분할, 건물 비율 필터 |
| `src/dataset.py` | 타일 Dataset, 원본 영상 단위 train/val 분리(타일 누수 방지) |
| `src/train.py` | U-Net(resnet34) 학습 — Dice+BCE, IoU/F1 로그, best 체크포인트 |
| `src/infer.py` | 큰 영상을 타일 추론 후 이어붙여 마스크·GeoJSON 생성 |
| `src/postprocess.py` | 마스크 → 폴리곤 → Douglas-Peucker 단순화 → 직사각형 형상 보정 → GeoJSON |
| `src/metrics.py` | IoU / F1 / Precision / Recall |
| `app/streamlit_app.py` | 데모: 영상 업로드 → 추출 결과 오버레이 → GeoJSON 다운로드 |
| `tests/` | 후처리·타일링 pytest |

## 빠른 시작

```bash
pip install -r requirements.txt
pytest                      # 후처리·타일링 테스트

# 1) 데이터 (Inria Aerial Image Labeling, ~20GB)
bash scripts/download_inria.sh

# 2) 타일 생성 (처음엔 --limit 10 으로 일부만)
python -m src.tiling --images data/raw/AerialImageDataset/train/images \
    --masks data/raw/AerialImageDataset/train/gt \
    --out data/tiles --tile 512 --stride 512 --min-fg 0.01 --limit 10

# 3) 학습 (GPU 권장 — 없으면 notebooks/train_colab.ipynb 사용)
python -m src.train --tiles data/tiles --epochs 15 --batch 8 --out runs/unet_r34

# 4) 추론 + GeoJSON
python -m src.infer --model runs/unet_r34/best.pt --image <영상.tif> --out out/

# 5) 데모
streamlit run app/streamlit_app.py
```

## 후처리 설계

모델 출력(래스터 마스크)은 그대로 쓰기 어렵다. 경계가 울퉁불퉁하고, 작은 오탐이 섞이고,
GIS 도구는 벡터를 요구하기 때문이다. 후처리는 네 단계로 나눈다.

1. **정제** — 모폴로지 open/close로 점 노이즈 제거
2. **벡터화** — 윤곽 추출로 픽셀 마스크를 폴리곤으로 변환, 최소 면적 필터
3. **형상 보정** — Douglas-Peucker 단순화 후, 최소회전사각형과 IoU가 임계값 이상이면
   사각형으로 교체(건물 대부분이 직각 형상이라는 사전 지식 반영)
4. **좌표 변환** — GeoTIFF의 affine transform이 있으면 픽셀좌표를 지리좌표로 변환해 GeoJSON 출력

각 단계의 효과는 후처리 전후 폴리곤을 다시 마스크로 되돌려 원본과의 IoU로 수치 확인한다.

## 데이터

- [Inria Aerial Image Labeling](https://project.inria.fr/aerialimagelabeling/) — 0.3m 항공 정사영상,
  미국·오스트리아 5개 도시 180장(5000×5000), 건물 이진 마스크 제공


## 결과 (1차 베이스라인)

- 데이터: Inria train 20장(austin) → 512px 타일, 건물 비율 1% 이상 필터
- 모델: U-Net(resnet34, ImageNet 사전학습), Dice+BCE, 15 epoch, Colab T4
- 검증 성능: **IoU 0.739 / F1 0.849** (원본 영상 단위 분리 검증, `runs` 로그 참조)
- 일반화 확인: 학습에 없는 도시(vienna1, 5000×5000)에서 건물 1,971동 추출 → GeoJSON 출력