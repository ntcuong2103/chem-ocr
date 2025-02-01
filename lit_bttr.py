import zipfile

import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch import FloatTensor, LongTensor

from datamodule import Batch, vocab
from bttr import VisionModel
from utils import ExpRateRecorder, ce_loss, we_rate, to_bi_tgt_out
from vocab import vocab


class CrocsBiVision(pl.LightningModule):
    def __init__(
        self,
        d_model: int,
        # encoder
        num_encoder_layers: int,
        # decoder
        num_heads: int,
        num_decoder_layers: int,
        d_ff: int,
        dropout: float,
        # training
        learning_rate: float,
        patience: int,
        vocab_size: int = 359,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = VisionModel(
            d_model=d_model,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=d_ff,
            dropout=dropout,
            vocab_size=vocab_size,
        )

        self.exprate_recorder = ExpRateRecorder()

    def forward(
        self, img: FloatTensor, img_mask: LongTensor, tgt: LongTensor
    ) -> FloatTensor:
        return self.model(img, img_mask, tgt)

    def training_step(self, batch: Batch, _):
        tgt, out = to_bi_tgt_out(batch.indices, self.device)
        out_hat = self(batch.imgs, batch.mask, tgt)

        loss = ce_loss(out_hat, out)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=len(batch),
        )

        return loss

    def validation_step(self, batch: Batch, _):
        tgt, out = to_bi_tgt_out(batch.indices, self.device)
        out_hat = self(batch.imgs, batch.mask, tgt)

        loss = ce_loss(out_hat, out)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=len(batch),
        )

        wer = we_rate(out_hat.argmax(dim=2), batch.indices, self.device)
        self.log(
            "val_wer",
            wer,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=len(batch),
        )

    def test_step(self, batch: Batch, _):
        tgt, out = to_bi_tgt_out(batch.indices, self.device)
        out_hat = self(batch.imgs, batch.mask, tgt)

        loss = ce_loss(out_hat, out)
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=len(batch),
        )

        wer = we_rate(out_hat.argmax(dim=2), batch.indices, self.device)

        print(out_hat.argmax(dim=2))
        print(batch.indices)
        print(wer)

        self.log(
            "val_wer",
            wer,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=len(batch),
        )

    def test_epoch_end(self, test_outputs) -> None:
        pass
        # exprate = self.exprate_recorder.compute()
        # print(f"ExpRate: {exprate}")

        # print(f"length of total file: {len(test_outputs)}")
        # with zipfile.ZipFile("result.zip", "w") as zip_f:
        #     for img_base, pred in test_outputs:
        #         content = f"%{img_base}\n${pred}$".encode()
        #         with zip_f.open(f"{img_base}.txt", "w") as f:
        #             f.write(content)

    def configure_optimizers(self):
        optimizer = optim.Adadelta(
            self.parameters(),
            lr=self.hparams.learning_rate,
            eps=1e-6,
            weight_decay=1e-4,
        )

        reduce_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.1,
            patience=self.hparams.patience // self.trainer.check_val_every_n_epoch,
        )
        scheduler = {
            "scheduler": reduce_scheduler,
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": self.trainer.check_val_every_n_epoch,
            "strict": True,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
