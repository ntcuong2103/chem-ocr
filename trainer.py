from datamodule import CROCSDatamodule
from lit_bttr import LitBTTR
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

if __name__ == "__main__":
    # model_load = LitBTTR.load_from_checkpoint('lightning_logs/version_2/checkpoints/epoch=48-step=837214-val_loss=0.3860.ckpt', strict=False)
    # model = LitBTTR(d_model=256, g
    # rowth_rate=24, num_layers=16, nhead=8, num_decoder_layers=3, dim_feedforward=1024, vocab_size=575, dropout=0.3, beam_size=1, max_len=200, alpha=1.0, learning_rate=1, patience=20)
    # model.bttr.encoder = model_load.bttr.encoder
    # model.bttr.decoder.model = model_load.bttr.decoder.model
    model = LitBTTR.load_from_checkpoint('lightning_logs/version_0/checkpoints/epoch=0-step=1307-val_loss=1.6266.ckpt')
    model.hparams.learning_rate = 0.01

    dm = CROCSDatamodule(batch_size=32, num_workers=15)
    trainer = Trainer(
        callbacks = [
            LearningRateMonitor(logging_interval='epoch'),
            ModelCheckpoint(dirpath="crocs-logs",filename='{epoch}-{step}-{val_loss:.4f}', save_top_k=10, monitor='val_loss', mode='min'),
            
        ], 
        check_val_every_n_epoch=1,
        fast_dev_run=False,

        deterministic=False, 
        max_epochs=50, 

        accelerator='gpu',
        devices=1,
    )

    if False:
        trainer.test(model, dm)
    else:
        trainer.fit(model, dm)