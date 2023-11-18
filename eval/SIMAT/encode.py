# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

import clip
import torch
import torchvision.datasets as datasets
from functools import partial
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import argparse 

def encode_simat(set_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DATA_PATH = 'simat_db/images/'
    CLIP_MODEL = 'ViT-B/32'
    weights_path = f'/home/tsuneda/data_dl/CLIP_finetuning/output/{set_name}/models/finetuned_clip_model_5_128.pth'

    model, prep = clip.load(CLIP_MODEL, device=device)
    finetuning_weights = torch.load(weights_path)
    model.load_state_dict(finetuning_weights)
    print('Model loaded')

    ds = datasets.ImageFolder(DATA_PATH, transform=prep)

    dl = torch.utils.data.DataLoader(ds, batch_size=32, num_workers=10, shuffle=False)

    img_enc = torch.cat([model.encode_image(b.to(device)).cpu().detach() for b, i in tqdm(dl)]).float()

    fnames = [x[0].name for x in datasets.ImageFolder(DATA_PATH, loader=Path)]
    region_ids = [int(x[:-4]) for x in fnames]

    img_enc_mapping = dict(zip(region_ids, img_enc))
    torch.save(img_enc_mapping, f'/home/tsuneda/data_dl/CLIP_finetuning/eval/SIMAT/data/simat_img_FTclip_{set_name}.pt')
    print('Image encodings saved')

    # encode words
    transfos = pd.read_csv('simat_db/transfos.csv', index_col=0)
    words = list(set(transfos.target) | set(transfos.value))
    tokens = clip.tokenize(words)

    word_encs = torch.cat([model.encode_text(b.to(device)).cpu().detach() for b in tqdm(tokens.split(32))])

    w2we = dict(zip(words, word_encs))
    torch.save(w2we, f'/home/tsuneda/data_dl/CLIP_finetuning/eval/SIMAT/data/simat_words_FTclip_{set_name}.ptd')
    print('Word encodings saved')

def main():
    parser = argparse.ArgumentParser(description='SIMAT Database Encoding with CLIP')
    parser.add_argument('--set', type=str, required=True, help='Set name for the output files')
    args = parser.parse_args()

    encode_simat(args.set)

if __name__ == "__main__":
    main()
