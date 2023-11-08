# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

import clip
import torch.nn as nn
from torchvision import datasets
import argparse
import torch
import pandas as pd
import numpy as np

torch.Tensor.normalize = lambda x: x/x.norm(dim=-1, keepdim=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def simat_eval1(args):
    emb_key = 'clip'
    output = {}
    # transfosとtripletsの読み込み
    transfos = pd.read_csv('simat_db/transfos.csv', index_col=0)
    triplets = pd.read_csv('simat_db/triplets.csv', index_col=0)
    # dataset_idとregion_idの対応を辞書に格納
    did2rid = dict(zip(triplets.dataset_id, triplets.index))
    rid2did = dict(zip(triplets.index, triplets.dataset_id))
    # transfosがtestかdevかを判定
    # testの場合はis_testがTrueのもののみを抽出
    # devの場合はis_testがFalseのもののみを抽出
    transfos = transfos[transfos.is_test == (args.domain == 'test')]
    # transfosのregion_idをdataset_idに変換
    transfos_did = [rid2did[rid] for rid in transfos.region_id]

    #clip_simatを読み込み
    clip_simat = torch.load('data/simat_img_clip.pt')
    # clip_simatをstackして正規化
    # stack: https://deepage.net/features/numpy-stack.html
    img_embs_stacked = torch.stack([clip_simat[did2rid[i]] for i in range(len(clip_simat))]).float().normalize()
    # value_embsを作成
    # value_embs: transfosのregion_idに対応するclip_simatの値をstackしたもの
    value_embs = torch.stack([img_embs_stacked[did] for did in transfos_did])

    # word_embsを読み込み
    # word_embs: word2vecの辞書
    word_embs = dict(torch.load(f'data/simat_words_{emb_key}.ptd'))
    # word_embsを正規化
    w2v = {k: v.float().normalize() for k, v in word_embs.items()}
    # w2v = {k:model.encode_text(v).normalize() for k, v in word_embs.items()}
    # delta_vectorsを作成
    delta_vectors = torch.stack([w2v[x.target] - w2v[x.value] for i, x in transfos.iterrows()])

    # oscar_scoresを読み込み
    oscar_scores = torch.load('simat_db/oscar_similarity_matrix.pt')
    # weightsを作成
    # weights: 1/√norm2
    # norm2: transfosのnorm2
    weights = 1/np.array(transfos.norm2)**.5
    weights = weights/sum(weights)

    # lambdaを変えながら評価
    for lbd in args.lbds:
        target_embs = value_embs + lbd*delta_vectors

        nnb = (target_embs @ img_embs_stacked.T).topk(5).indices
        nnb_notself = [r[0] if r[0].item() != t else r[1] for r, t in zip(nnb, transfos_did)]

        scores = np.array([oscar_scores[ri, tc] for ri, tc in zip(nnb_notself, transfos.target_ids)]) > .5


        output[lbd] = 100*np.average(scores, weights=weights)
    return output

def simat_eval2(args):
    emb_key = 'clip'
    output = {}

    transfos = pd.read_csv('simat_db/transfos.csv', index_col=0)
    triplets = pd.read_csv('simat_db/triplets.csv', index_col=0)
    did2rid = dict(zip(triplets.dataset_id, triplets.index))
    rid2did = dict(zip(triplets.index, triplets.dataset_id))

    transfos = transfos[transfos.is_test == (args.domain == 'test')]

    transfos_did = [rid2did[rid] for rid in transfos.region_id]

    #new method
    clip_simat = torch.load('data/simat_img_ftclip_1.pt')
    img_embs_stacked = torch.stack([clip_simat[did2rid[i]] for i in range(len(clip_simat))]).float().normalize()
    value_embs = torch.stack([img_embs_stacked[did] for did in transfos_did])

    word_embs = dict(torch.load(f'data/simat_words_ftclip_1.ptd'))
    w2v = {k: v.float().normalize() for k, v in word_embs.items()}
    # w2v = {k:model.encode_text(v).normalize() for k, v in word_embs.items()}
    delta_vectors = torch.stack([w2v[x.target] - w2v[x.value] for i, x in transfos.iterrows()])

    oscar_scores = torch.load('simat_db/oscar_similarity_matrix.pt')
    weights = 1/np.array(transfos.norm2)**.5
    weights = weights/sum(weights)

    for lbd in args.lbds:
        target_embs = value_embs + lbd*delta_vectors

        nnb = (target_embs @ img_embs_stacked.T).topk(5).indices
        nnb_notself = [r[0] if r[0].item() != t else r[1] for r, t in zip(nnb, transfos_did)]

        scores = np.array([oscar_scores[ri, tc] for ri, tc in zip(nnb_notself, transfos.target_ids)]) > .5


        output[lbd] = 100*np.average(scores, weights=weights)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run eval')
    parser.add_argument('--domain', type=str, default='dev', help='domain, test or dev')
    parser.add_argument('--backbone', type=str, default='clip', help='backbone method. Only clip is supported.')
    parser.add_argument('--tau', type=float, default=0.1, help='pretraining temperature tau')
    parser.add_argument('--lbds', nargs='+', default=[1], help='list of values for lambda')
    args = parser.parse_args()
    args.lbds = [float(l) for l in args.lbds]

    output1 = simat_eval1(args)
    output2 = simat_eval2(args)
    print('SIMAT Scores (Original CLIP):')
    for lbd, v in output1.items():
        print(f'{lbd=}: {v:.2f}')

    print('SIMAT Scores (Finetuned CLIP):')
    for lbd, v in output2.items():
        print(f'{lbd=}: {v:.2f}')
