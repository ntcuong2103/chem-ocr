from datamodule import CROCSDataModule
from model import CrocsBiVision
from pytorch_lightning import Trainer, seed_everything

if __name__ == "__main__":
    # model initialization
    model = CrocsBiVision.load_from_checkpoint(
        "bivision-logs-new2/epoch=49-step=685450-val_loss=0.6076-val_wer=0.2679-359.ckpt"
    )
    model.hparams.learning_rate = 0.01

    # data module
    dm = CROCSDataModule(batch_size=32, num_workers=15)

    # for reproducibility
    seed_everything(80, workers=True)

    # trainer
    trainer = Trainer(
        enable_checkpointing=False,
        fast_dev_run=False,
        deterministic=False,
        max_epochs=10,
        accelerator="gpu",
        devices=1,
    )

    trainer.test(model, dm)