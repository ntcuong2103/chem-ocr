import torch
import torch.nn as nn
import pytorch_lightning as pl
from einops import rearrange
from pos_enc import ImgPosEnc
from datamodule import vocab_size
from torch.nn.modules.transformer import TransformerEncoder


def _build_transformer_encoder(
    d_model: int,
    nhead: int,
    num_encoder_layers: int,
    dim_feedforward: int,
    dropout: float,
) -> nn.TransformerEncoder:
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        batch_first=True,
    )

    return TransformerEncoder(encoder_layer, num_encoder_layers)


class Encoder(pl.LightningModule):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        image_patch_size: int = 16,
    ):
        super().__init__()

        self.conv2d = nn.Conv2d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=image_patch_size,
            stride=image_patch_size,
            padding=0,
        )
        self.bn = nn.BatchNorm2d(d_model)

        self.image_pos_enc = ImgPosEnc(d_model, normalize=True)

        self.word_embed = nn.Sequential(
            nn.Embedding(vocab_size, d_model), nn.LayerNorm(d_model)
        )
        # encoder
        self.model = _build_transformer_encoder(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    def forward(self, src, src_mask):
        # conv2d into patch
        src = self.conv2d(src)
        src = self.bn(src)
        src_mask = src_mask[:, 0::16, 0::16]

        # rearrange, d = image_patch_size **2 to img_pos_enc
        src = rearrange(src, "b p h w -> b h w p")

        # image positional encoding
        src = self.image_pos_enc(src, src_mask)

        # flatten to 1d
        src = rearrange(src, "b h w d -> b (h w) d")

        # encoder output
        out_enc = self.model(src=src)

        out_enc_mask = rearrange(src_mask, "b h w -> b (h w)")

        return out_enc, out_enc_mask


if __name__ == "__main__":
    # test encoder
    encoder = Encoder(
        d_model=512,
        nhead=8,
        num_encoder_layers=16,
        dim_feedforward=1024,
        dropout=0.3,
        image_patch_size=16,
    )

    img = torch.randn(2, 1, 256, 256)
    img_mask = torch.ones(2, 256, 256).float()

    out_enc, out_enc_mask = encoder(img, img_mask)
    print(out_enc.shape, out_enc_mask.shape)
    # torch.Size([2, 256, 512]) torch.Size([2, 256])
