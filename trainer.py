from datamodule import CROCSDataModule
from lit_bttr import CrocsBiVision
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

from vocab import vocab_full_size, vocab_size

if __name__ == "__main__":
  # model initialization
  # model = CrocsBiVision(d_model=256, num_encoder_layers=3, num_heads=8, num_decoder_layers=3, d_ff=1024, vocab_size=vocab_size, dropout=0.3, learning_rate=0.01, patience=20)
  model = CrocsBiVision.load_from_checkpoint('bivision-logs-new2/epoch=49-step=685450-val_loss=0.6705-val_wer=0.2955-359.ckpt')
  
  model.hparams.learning_rate = 0.05

  # data module
  dm = CROCSDataModule(batch_size=32, num_workers=15)
  
  # for reproducibility
  seed_everything(80, workers=True)
  
  # trainer
  trainer = Trainer(
    callbacks = [
      LearningRateMonitor(logging_interval='epoch'),
      ModelCheckpoint(dirpath='bivision-logs-new2',filename='{epoch}-{step}-{val_loss:.4f}-{val_wer:.4f}-359', save_top_k=10, monitor='val_loss', mode='min'),
      EarlyStopping(monitor='val_wer', patience=20, mode='min')
    ],
    
    # val check every epoch
    check_val_every_n_epoch=1,
    
    # for debugging
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