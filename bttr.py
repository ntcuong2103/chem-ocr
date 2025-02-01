from typing import List

import pytorch_lightning as pl
import torch
from torch import FloatTensor, LongTensor

from utils import Hypothesis

from decoder import Decoder
from encoder_vit import Encoder


class BTTR(pl.LightningModule):
    def __init__(
        self,
        d_model: int,
        image_patch_size: int,
        num_encoder_layers: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        vocab_size: int,
    ):
        super().__init__()

        self.encoder = Encoder(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            image_patch_size=image_patch_size,
        )
        
        self.decoder = Decoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            vocab_size=vocab_size,
        )

    def forward(
        self, img: FloatTensor, img_mask: LongTensor, tgt: LongTensor
    ) -> FloatTensor:
        """run img and bi-tgt

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h, w]
        img_mask: LongTensor
            [b, h, w]
        tgt : LongTensor
            [2b, l]

        Returns
        -------
        FloatTensor
            [2b, l, vocab_size]
        """
        feature, mask = self.encoder(img, img_mask)  # [b, t, d]
        feature = torch.cat((feature, feature), dim=0)  # [2b, t, d]
        mask = torch.cat((mask, mask), dim=0)

        out = self.decoder(feature, mask, tgt)

        return out

    def beam_search(
        self, img: FloatTensor, img_mask: LongTensor, beam_size: int, max_len: int
    ) -> List[Hypothesis]:
        """run bi-direction beam search for given img

        Parameters
        ----------
        img : FloatTensor
            [1, 1, h', w']
        img_mask: LongTensor
            [1, h', w']
        beam_size : int
        max_len : int

        Returns
        -------
        List[Hypothesis]
        """
        feature, mask = self.encoder(img, img_mask)  # [1, t, d]
        return self.decoder.beam_search(feature, mask, beam_size, max_len)
