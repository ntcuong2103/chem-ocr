import json
from lit_bttr import LitBTTR
from PIL import Image
from torchvision.transforms import ToTensor
from PIL import Image
from tqdm import tqdm

ckpt = 'lightning_logs/version_22/checkpoints/epoch=9-step=17260-val_loss=0.3059.ckpt'
img_path = 'train/train_38498.jpg'
device = 'cuda:2'
img_zip_path = 'crocs-train-zips/test-B/testB.zip'

model = LitBTTR.load_from_checkpoint(ckpt)
model.eval()
model.to(device)

H_LO = 60
H_HI = 1000
W_LO = 60
W_HI = 2000
N_BEAMS = 5

def decoding(img, model, device):
    from datamodule import ScaleToLimitRange
    from torchvision.transforms import Compose
    from torchvision import transforms

    transform = Compose([ScaleToLimitRange(W_LO, W_HI, H_LO, H_HI), transforms.ToTensor()])
    img_trans = transform(img)

    img_trans = img_trans.to(device)

    hyp = model.beam_search(img_trans, beam_size=N_BEAMS, max_len=300)

    return hyp

def run_decode_archive(img_zip_path):
    import zipfile
    import cv2
    import os
    out_fn = f"result_{os.path.splitext(os.path.basename(img_zip_path))[0]}_{N_BEAMS}.txt"

    # list all files in the zip file
    out_list = []
    with zipfile.ZipFile(img_zip_path, "r") as f:
        for fn in tqdm(f.namelist()[:]):
            if fn.endswith(".jpg"):
                with f.open(fn, "r") as img_f:
                    img = Image.open(img_f).copy()
                    img = np.array(img.convert("L"))
                    out_text = decoding(img, model, device)
                    out_list.append((fn, out_text))
    
    with open(out_fn, "w") as f:
        f.writelines([f"{fn}\t{out_text}\n" for fn, out_text in out_list])


def run_decode_folder(path, fns=None):
    import os
    out_fn = f"result_{os.path.basename(path)}_{N_BEAMS}.txt"

    out_list = []
    if fns is None:
        fns = os.listdir(path)

    for fn in tqdm(fns):
        if fn.endswith(".jpg"):
            img = Image.open(os.path.join(path, fn)).copy()
            img = np.array(img.convert("L"))
            out_text = decoding(img, model, device)
            out_list.append((fn, out_text))
    
    with open(out_fn, "w") as f:
        f.writelines([f"{fn}\t{out_text}\n" for fn, out_text in out_list])


if __name__ == "__main__":
    # run_decode_archive(img_zip_path)
    fns = [line.strip().split('\t')[0] for line in open('mini_validation.txt').readlines()]
    # run_decode_folder('train', fns)
    lbls = [json.load(open(f"train/{fn.replace('jpg', 'json')}"))["ssml_sd"] for fn in tqdm(fns)]
    with open('train_gt.txt', 'w') as f:
        f.writelines([f"{fn}\t{lbl}\n" for fn, lbl in zip(fns, lbls)])