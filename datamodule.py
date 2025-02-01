import numpy as np
import cv2
from typing import List, Optional, Tuple
from torchvision.transforms import transforms, Compose
import pytorch_lightning as pl
from PIL import Image
from torch import FloatTensor, LongTensor
from torch.utils.data.dataloader import DataLoader

import math

from dataclasses import dataclass
import torch

import os
import random
from tqdm import tqdm

from vocab import vocab, vocab_full

# DataType: fname, image, output
Data = List[Tuple[str, Image.Image, List[str]]]

# Scale images down for easy processing
H_LO = 16
H_HI = 640
W_LO = 16
W_HI = 640


class ScaleToImageSize:
    def __init__(
        self,
        patch_size: int = 16,
        w_lo: int = W_LO,
        w_hi: int = W_HI,
        h_lo: int = H_LO,
        h_hi: int = H_HI,
    ) -> None:
        assert w_lo <= w_hi and h_lo <= h_hi
        self.w_lo = w_lo
        self.w_hi = w_hi
        self.h_lo = h_lo
        self.h_hi = h_hi
        self.patch_size = patch_size

    def __call__(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        r = h / w
        lo_r = self.h_lo / self.w_hi
        hi_r = self.h_hi / self.w_lo
        assert lo_r <= h / w <= hi_r, f"img ratio h:w {r} not in range [{lo_r}, {hi_r}]"

        scale_r = min(self.h_hi / h, self.w_hi / w)
        if scale_r < 1.0:
            # one of h or w highr that hi, so scale down
            img = cv2.resize(
                img, None, fx=scale_r, fy=scale_r, interpolation=cv2.INTER_CUBIC
            )

        scale_r = max(self.h_lo / h, self.w_lo / w)
        if scale_r > 1.0:
            # one of h or w lower that lo, so scale up
            img = cv2.resize(
                img, None, fx=scale_r, fy=scale_r, interpolation=cv2.INTER_CUBIC
            )

        h, w = img.shape[:2]

        # in the rectangle, do not scale
        assert self.h_lo <= h <= self.h_hi and self.w_lo <= w <= self.w_hi

        new_h, new_w = img.shape[:2]

        new_h = self.patch_size * math.ceil(new_h / self.patch_size)
        new_w = self.patch_size * math.ceil(new_w / self.patch_size)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return img


MAX_SIZE = 32e4  # change here according to your GPU memory
MIN_SIZE = 3600


# load data
def data_iterator(
    data: Data,
    batch_size: int,
    batch_Imagesize: int = MAX_SIZE * 5,
    maxlen: int = 300,
    minImagesize: int = MIN_SIZE,
    mode="train",
):
    fname_batch = []
    feature_batch = []
    label_batch = []
    feature_total = []
    label_total = []
    fname_total = []
    biggest_image_size = 0

    data.sort(key=lambda x: x[1].size)

    transform = Compose([ScaleToImageSize(), transforms.ToTensor()])

    i = 0
    for fname, fea, lab in data:
        size = fea.size
        fea = transform(fea)
        if size > biggest_image_size:
            biggest_image_size = size
        batch_image_size = biggest_image_size * (i + 1)
        if len(lab) > maxlen:
            print("sentence", i, "length bigger than", maxlen, "ignore")
        elif size < minImagesize:
            print(f"image: {fname} size:{size} less than {minImagesize}, ignore")
        else:
            if mode == "train":
                if (
                    batch_image_size > batch_Imagesize or i == batch_size
                ):  # a batch is full
                    fname_total.append(fname_batch)
                    feature_total.append(feature_batch)
                    label_total.append(label_batch)
                    i = 0
                    biggest_image_size = size
                    fname_batch = []
                    feature_batch = []
                    label_batch = []
                    fname_batch.append(fname)
                    feature_batch.append(fea)
                    label_batch.append(lab)
                    i += 1
                else:
                    fname_batch.append(fname)
                    feature_batch.append(fea)
                    label_batch.append(lab)
                    i += 1
            else:
                fname_batch.append(fname)
                feature_batch.append(fea)
                label_batch.append(lab)

                fname_total.append(fname_batch)
                feature_total.append(feature_batch)
                label_total.append(label_batch)

                fname_batch = []
                feature_batch = []
                label_batch = []

    if mode == "train":
        # last batch
        fname_total.append(fname_batch)
        feature_total.append(feature_batch)
        label_total.append(label_batch)
    print("total ", len(feature_total), "batch data loaded")
    return list(zip(fname_total, feature_total, label_total))


# n: number of files to use
def read_data(path: str, n: int = -1) -> Data:
    data = []
    fns = os.listdir(path)
    jpg_fns = ["train_38563.jpg", "train_10674.jpg", "train_30738.jpg"]

    label_dict = {
        line.strip().split("\t")[0]: line.strip().split("\t")[1]
        for line in open("train_ssml_sd.txt").readlines()
    }

    # randomise
    if n > 0:
        jpg_fns = random.sample(jpg_fns, n)
    i = 0
    # load into data
    for fn in tqdm(jpg_fns):
        if fn.endswith(".jpg"):
            print(fn, i)
            i += 1
            img = cv2.imread(os.path.join(path, fn), cv2.IMREAD_GRAYSCALE)
            data.append(
                (fn, img, label_dict[os.path.splitext(fn)[0] + ".json"].split())
            )

    return data


def build_dataset(folder: str, batch_size: int, n: int = -1, mode="train"):
    data = read_data(folder, n)
    return data_iterator(data, batch_size, mode=mode)


@dataclass
class Batch:
    img_bases: List[str]  # [b,]
    imgs: FloatTensor  # [b, 1, H, W]
    mask: LongTensor  # [b, H, W]
    indices: List[List[int]]  # [b, l]

    def __len__(self) -> int:
        return len(self.img_bases)

    def to(self, device) -> "Batch":
        return Batch(
            img_bases=self.img_bases,
            imgs=self.imgs.to(device),
            mask=self.mask.to(device),
            indices=self.indices,
        )


def collate_fn(batch):
    assert len(batch) == 1
    # ????
    batch = batch[0]

    # file name
    fnames = batch[0]
    # img Tensor
    images_x = batch[1]
    # expected output
    seqs_y = [vocab.words2indices(x) for x in batch[2]]

    heights_x = [s.size(1) for s in images_x]
    widths_x = [s.size(2) for s in images_x]

    n_samples = len(heights_x)
    max_height_x = max(heights_x)
    max_width_x = max(widths_x)

    x = torch.zeros(n_samples, 1, max_height_x, max_width_x)
    x_mask = torch.ones(
        n_samples,
        math.floor(max_height_x / 16) * 16,
        math.floor(max_width_x / 16) * 16,
        dtype=torch.float,
    )
    for idx, s_x in enumerate(images_x):
        x[idx, :, : heights_x[idx], : widths_x[idx]] = s_x
        x_mask[idx, : heights_x[idx], : widths_x[idx]] = 0

    # return fnames, x, x_mask, seqs_y
    return Batch(fnames, x, x_mask, seqs_y)


class CROCSDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, num_workers: int):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage):
        if stage == "fit" or stage is None:
            self.train_dataset = build_dataset("train", self.batch_size, 40000)
            self.val_dataset = build_dataset("train", self.batch_size, 5000)
        if stage == "test" or stage is None:
            self.test_dataset = build_dataset("train", self.batch_size, 3, mode="test")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )


if __name__ == "__main__":
    # test
    build_dataset("train", 16, 100)
