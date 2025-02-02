import pytorch_lightning as pl
from transformers import ViTModel, ViTImageProcessor
from einops import rearrange
import torch.nn as nn

# https://github.com/NielsRogge/Transformers-Tutorials/blob/master/VisionTransformer/Fine_tuning_the_Vision_Transformer_on_CIFAR_10_with_PyTorch_Lightning.ipynb

class Encoder(pl.LightningModule):
    def __init__(self, d_model: int):
        super(Encoder, self).__init__()
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k', do_resize=False, do_rescale=False, do_normalize=True)
        self.model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k', add_pooling_layer=False)
        self.linear = nn.Linear(self.model.config.hidden_size, d_model)

    def forward(self, src, src_mask):
        src_mask = src_mask[:, 0::16, 0::16]

        # encoder output
        # make src depth 1 -> 3
        src = src.repeat(1, 3, 1, 1)
        input = self.processor(images=src, return_tensors="pt").to(src.device)
        out_enc = self.model(**input, interpolate_pos_encoding=True, return_dict=False)[0][:, 1:, :] # remove cls token
        out_enc = self.linear(out_enc)

        out_enc_mask = rearrange(src_mask, "b h w -> b (h w)")

        return out_enc, out_enc_mask


if __name__ == "__main__":
    encoder = Encoder(d_model=512)
    # test encoder
    import torch

    img = torch.randn(2, 1, 256, 256)
    img_mask = torch.ones(2, 256, 256).float()

    out_enc, out_enc_mask = encoder(img, img_mask)
    print(out_enc.shape, out_enc_mask.shape)
