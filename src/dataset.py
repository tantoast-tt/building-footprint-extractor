# -*- coding: utf-8 -*-
"""타일 폴더(images/, masks/)를 읽는 PyTorch Dataset."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class TileDataset(Dataset):
    def __init__(self, root, file_list=None, transform=None):
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.masks_dir = self.root / "masks"
        if file_list is None:
            self.files = sorted(p.name for p in self.images_dir.iterdir() if p.suffix == ".png")
        else:
            self.files = list(file_list)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        name = self.files[i]
        img = np.array(Image.open(self.images_dir / name).convert("RGB"))
        msk = np.array(Image.open(self.masks_dir / name))
        if msk.ndim == 3:
            msk = msk[:, :, 0]
        msk = (msk > 127).astype(np.float32)

        if self.transform is not None:
            t = self.transform(image=img, mask=msk)
            img, msk = t["image"], t["mask"]

        img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        msk = torch.from_numpy(msk)[None].float()
        return img, msk


def split_files(root, val_ratio=0.15, seed=42):
    """타일 파일명을 원본 이미지 단위로 train/val 분리(타일 누수 방지)."""
    root = Path(root)
    files = sorted(p.name for p in (root / "images").iterdir() if p.suffix == ".png")
    stems = sorted({f.rsplit("_", 2)[0] for f in files})
    rng = np.random.default_rng(seed)
    rng.shuffle(stems)
    n_val = max(1, int(len(stems) * val_ratio))
    val_stems = set(stems[:n_val])
    train = [f for f in files if f.rsplit("_", 2)[0] not in val_stems]
    val = [f for f in files if f.rsplit("_", 2)[0] in val_stems]
    return train, val
