#!/usr/bin/env bash
# Inria Aerial Image Labeling Dataset 다운로드 (약 20GB, 7z 분할 5개)
# 공식 안내: https://project.inria.fr/aerialimagelabeling/
# 실행 후: data/raw/AerialImageDataset/train/{images,gt}/ 가 생긴다.
set -e
mkdir -p data/raw && cd data/raw

for i in 001 002 003 004 005; do
  [ -f aerialimagelabeling.7z.$i ] || curl -L -O "https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.$i"
done

7z x aerialimagelabeling.7z.001   # NEW2-AerialImageDataset.zip 생성
unzip -o NEW2-AerialImageDataset.zip
echo "완료: data/raw/AerialImageDataset/train/{images,gt}"
