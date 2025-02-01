from datamodule import CROCSDatamodule
from lit_bttr import LitBTTR
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

if __name__ == "__main__":
    model = LitBTTR.load_from_checkpoint('lightning_logs/version_14/checkpoints/epoch=46-step=80464-val_loss=0.3708.ckpt')
    model.hparams.learning_rate = 0.001
    dm = CROCSDatamodule(batch_size=32, num_workers=15)
    trainer = Trainer(
        enable_checkpointing=False,
        fast_dev_run=False,

        deterministic=False, 
        max_epochs=50, 

        accelerator='gpu',
        devices=1,
    )

    trainer.test(model, dm)