# 実験設定とか

## ワークステーションとの接続
### ログイン
ssh -p "8824" dl_user@165.242.103.60
### データの転送
rsync -e "ssh -p 8824" -avz ./sample_program/ dl_user@165.242.103.60:/home/dl_user/data/tsuneda/CLIP_finetuning
### データのダウンロード
rsync -e "ssh -p 8824" -avz  dl_user@165.242.103.60:/home/dl_user/data/tsuneda/CLIP_finetuning ./sample_program/

---

## 実験結果のまとめ
### set1
- 最初の実験
- 
