# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

import clip
import torch.nn as nn
from torchvision import datasets
import argparse
import torch
import pandas as pd
import numpy as np
import os
import shutil

torch.Tensor.normalize = lambda x: x/x.norm(dim=-1, keepdim=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def simat_eval(args):
    output = {}
    if args.backbone == 'clip':
        img_emb_path = 'data/simat_img_clip.pt'
        word_emb_path = 'data/simat_words_clip.ptd'
    else:  # ファインチューニングされたモデルの場合
        img_emb_path = f'data/simat_img_FTclip_{args.backbone}.pt'
        word_emb_path = f'data/simat_words_FTclip_{args.backbone}.ptd'

    transfos = pd.read_csv('simat_db/transfos.csv', index_col=0)
    triplets = pd.read_csv('simat_db/triplets.csv', index_col=0)
    did2rid = dict(zip(triplets.dataset_id, triplets.index))
    rid2did = dict(zip(triplets.index, triplets.dataset_id))
    
    transfos = transfos[transfos.is_test == (args.domain == 'test')]
    
    transfos_did = [rid2did[rid] for rid in transfos.region_id]

    # img_embs_stacked と word_embs のロード
    clip_simat = torch.load(img_emb_path)
    img_embs_stacked = torch.stack([clip_simat[did2rid[i]] for i in range(len(clip_simat))]).float().normalize()
    value_embs = torch.stack([img_embs_stacked[did] for did in transfos_did])
    
    word_embs = dict(torch.load(word_emb_path))
    w2v = {k: v.float().normalize() for k, v in word_embs.items()}
    delta_vectors = torch.stack([w2v[x.target] - w2v[x.value] for i, x in transfos.iterrows()])

    oscar_scores = torch.load('simat_db/oscar_similarity_matrix.pt')
    weights = 1/np.array(transfos.norm2)**.5
    weights = weights/sum(weights)

    for lbd in args.lbds:
        target_embs = value_embs + lbd*delta_vectors

        nnb = (target_embs @ img_embs_stacked.T).topk(5).indices
        nnb_notself = [r[0] if r[0].item() != t else r[1] for r, t in zip(nnb, transfos_did)]
        
        scores = np.array([oscar_scores[ri, tc] for ri, tc in zip(nnb_notself, transfos.target_ids)]) > .5
        false_indices = [index for index, (ri, tc) in enumerate(zip(nnb_notself, transfos.target_ids)) if oscar_scores[ri, tc] <= .5]

        output[lbd] = 100*np.average(scores, weights=weights)

    return output

def simat_eval2(args, n=5):
    output = {}
    if args.backbone == 'clip':
        img_emb_path = 'data/simat_img_clip.pt'
        word_emb_path = 'data/simat_words_clip.ptd'
    else:  # ファインチューニングされたモデルの場合
        img_emb_path = f'data/simat_img_FTclip_{args.backbone}.pt'
        word_emb_path = f'data/simat_words_FTclip_{args.backbone}.ptd'

    transfos = pd.read_csv('simat_db/transfos.csv', index_col=0)
    triplets = pd.read_csv('simat_db/triplets.csv', index_col=0)
    did2rid = dict(zip(triplets.dataset_id, triplets.index))
    rid2did = dict(zip(triplets.index, triplets.dataset_id))
    
    transfos = transfos[transfos.is_test == (args.domain == 'test')]
    
    transfos_did = [rid2did[rid] for rid in transfos.region_id]

    # img_embs_stacked と word_embs のロード
    clip_simat = torch.load(img_emb_path)
    img_embs_stacked = torch.stack([clip_simat[did2rid[i]] for i in range(len(clip_simat))]).float().normalize()
    value_embs = torch.stack([img_embs_stacked[did] for did in transfos_did])
    
    word_embs = dict(torch.load(word_emb_path))
    w2v = {k: v.float().normalize() for k, v in word_embs.items()}
    delta_vectors = torch.stack([w2v[x.target] - w2v[x.value] for i, x in transfos.iterrows()])

    oscar_scores = torch.load('simat_db/oscar_similarity_matrix.pt')
    weights = 1/np.array(transfos.norm2)**.5
    weights = weights/sum(weights)

    for lbd in args.lbds:
        target_embs = value_embs + lbd*delta_vectors

        # 最も近い n 個の候補を選択
        nnb = (target_embs @ img_embs_stacked.T).topk(n).indices
        
        # 各ターゲットに対して、選択された n 個の中に正解が含まれているかを確認
        correct_results = []
        for idx, (candidates, target_id) in enumerate(zip(nnb, transfos.target_ids)):
            correct = any(oscar_scores[candidate, target_id] > .5 for candidate in candidates)
            correct_results.append(correct)

        scores = np.array(correct_results)
        output[lbd] = 100 * np.average(scores, weights=weights)

    return output

def save_scores_to_csv(scores, model_name, csv_file='../evaluate/simat_scores.csv'):
    # スコアをDataFrameに変換
    df_scores = pd.DataFrame({model_name: scores})

    # CSVファイルが存在しない場合は、新しいファイルを作成
    if not os.path.exists(csv_file):
        df_scores.to_csv(csv_file)
    else:
        # 既存のデータを読み込む
        df_existing = pd.read_csv(csv_file, index_col=0)
        # 新しいスコアを追加
        df_combined = pd.concat([df_existing, df_scores], axis=1)
        # CSVファイルを更新
        df_combined.to_csv(csv_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run eval')
    parser.add_argument('--domain', type=str, default='dev', help='domain, test or dev')
    parser.add_argument('--backbone', type=str, default='clip', help='backbone method. Use "clip" for original or "set#" for finetuned models.')
    parser.add_argument('--tau', type=float, default=0.1, help='pretraining temperature tau')
    parser.add_argument('--lbds', nargs='+', default=[1, 2, 3, 4, 5], help='list of values for lambda')
    parser.add_argument('--save', action='store_true', help='save scores to csv')
    parser.add_argument('--save2', action='store_true', help='save scores to csv')
    args = parser.parse_args()
    args.lbds = [float(l) for l in args.lbds]
    
    # SIMATスコアを計算
    output = simat_eval(args)
    output2 = simat_eval2(args, n=5)
    # スコアの表示
    print(f'SIMAT Scores for {args.backbone}:')
    for lbd, v in output.items():
        print(f'Lambda {lbd}: {v:.2f}%')

    print(f'SIMAT Scores for {args.backbone} (n=5):')
    for lbd, v in output2.items():
        print(f'Lambda {lbd}: {v:.2f}%')

    # スコアをCSVに保存
    if args.save:
        save_scores_to_csv(output, args.backbone)

    if args.save2:
        save_scores_to_csv(output2, args.backbone + '(n=5)', csv_file='../evaluate/simat_scores2.csv')
