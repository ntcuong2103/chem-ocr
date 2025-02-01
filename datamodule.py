import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np
from tqdm import tqdm

import pytorch_lightning as pl
import torch
from PIL import Image
from torch import FloatTensor, LongTensor
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms, Compose

import random
import json
from vocab import Vocab

vocab = Vocab(dict_path='vocab_syms_full.txt')
vocab_size = len(vocab)

Data = List[Tuple[str, Image.Image, List[str]]]

H_LO = 16
H_HI = 640
W_LO = 16
W_HI = 640

class ScaleToLimitRange:
    def __init__(self, w_lo: int = W_LO, w_hi: int = W_HI, h_lo: int = H_LO, h_hi: int = H_HI) -> None:
        assert w_lo <= w_hi and h_lo <= h_hi
        self.w_lo = w_lo
        self.w_hi = w_hi
        self.h_lo = h_lo
        self.h_hi = h_hi
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        r = h / w
        lo_r = self.h_lo / self.w_hi
        hi_r = self.h_hi / self.w_lo
        assert lo_r <= h / w <= hi_r, f"img ratio h:w {r} not in range [{lo_r}, {hi_r}]"

        scale_r = min(self.h_hi / h, self.w_hi / w)
        if scale_r < 1.0:
            # one of h or w highr that hi, so scale down
            img = cv2.resize(img, None, fx=scale_r, fy=scale_r, interpolation=cv2.INTER_CUBIC)
            return img

        scale_r = max(self.h_lo / h, self.w_lo / w)
        if scale_r > 1.0:
            # one of h or w lower that lo, so scale up
            img = cv2.resize(img, None, fx=scale_r, fy=scale_r, interpolation=cv2.INTER_CUBIC)
            return img
        
        # in the rectangle, do not scale
        assert self.h_lo <= h <= self.h_hi and self.w_lo <= w <= self.w_hi
        return img


MAX_SIZE = 32e4  # change here according to your GPU memory
MIN_SIZE = 3600
# load data
def data_iterator(
    data: Data,
    batch_size: int,
    batch_Imagesize: int = MAX_SIZE*5,
    maxlen: int = 300,
    minImagesize: int = MIN_SIZE,
):
    fname_batch = []
    feature_batch = []
    label_batch = []
    feature_total = []
    label_total = []
    fname_total = []
    biggest_image_size = 0

    data.sort(key=lambda x: x[1].size)

    transform = Compose([ScaleToLimitRange(), transforms.ToTensor()])

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
            print(
                f"image: {fname} size:{size} less than {minImagesize}, ignore"
            )
        else:
            if batch_image_size > batch_Imagesize or i == batch_size:  # a batch is full
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

    # last batch
    fname_total.append(fname_batch)
    feature_total.append(feature_batch)
    label_total.append(label_batch)
    print("total ", len(feature_total), "batch data loaded")
    return list(zip(fname_total, feature_total, label_total))

def read_data(path: str, n: int = -1) -> Data:
    data = []
    fns = os.listdir(path)
    label_dict = { line.strip().split('\t')[0]:line.strip().split('\t')[1]for line in open ('train_ssml_sd.txt').readlines()}
    if n > 0:
        fns = random.sample(fns, n)
    for fn in tqdm(fns):
        if fn.endswith(".jpg"):
            # img = Image.open(os.path.join(path, fn))
            img = cv2.imread(os.path.join(path, fn), cv2.IMREAD_GRAYSCALE)
            # lbl_fn = fn.replace(".jpg", ".json")
            # with open(os.path.join(path, lbl_fn), "r") as f:
            #     lbl = json.load(f)
            # data.append((fn, img, lbl["ssml_sd"].split()))
            data.append((fn, img, label_dict[os.path.splitext(fn)[0]].split()))
        
    return data


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
    batch = batch[0]
    fnames = batch[0]
    images_x = batch[1]
    seqs_y = [vocab.words2indices(x) for x in batch[2]]

    heights_x = [s.size(1) for s in images_x]
    widths_x = [s.size(2) for s in images_x]

    n_samples = len(heights_x)
    max_height_x = max(heights_x)
    max_width_x = max(widths_x)

    x = torch.zeros(n_samples, 1, max_height_x, max_width_x)
    x_mask = torch.ones(n_samples, max_height_x, max_width_x, dtype=torch.float)
    for idx, s_x in enumerate(images_x):
        x[idx, :, : heights_x[idx], : widths_x[idx]] = s_x
        x_mask[idx, : heights_x[idx], : widths_x[idx]] = 0

    # return fnames, x, x_mask, seqs_y
    return Batch(fnames, x, x_mask, seqs_y)


def build_dataset(folder: str, batch_size: int, n: int = -1):
    data = read_data(folder, n)
    return data_iterator(data, batch_size)


class CROCSDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 8,
        num_workers: int = 5,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = build_dataset("train", self.batch_size, 10000)
            self.val_dataset = build_dataset("train", self.batch_size, 5000)
        if stage == "test" or stage is None:
            self.test_dataset = build_dataset("train", self.batch_size, 1000)

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
    build_dataset("train", 16, 100)
    # from argparse import ArgumentParser

    # batch_size = 2

    # parser = ArgumentParser()
    # parser = CROHMEDatamodule.add_argparse_args(parser)

    # args = parser.parse_args(["--batch_size", f"{batch_size}"])

    # dm = CROHMEDatamodule(**vars(args))
    # dm.setup()

    # train_loader = dm.train_dataloader()
    # for img, mask, tgt, output in train_loader:
    #     break

