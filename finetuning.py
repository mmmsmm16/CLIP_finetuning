import os
import torch
import torch.nn as nn
import torch.optim as optim
import clip
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
# from torchinfo import summary


num_epoch = 5
batch_size = 128

# 学習済みモデルのロード
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# テキストエンコーダの重みを固定
for param in model.transformer.parameters():
    param.requires_grad = False


class CustomDataset(Dataset):
    def __init__(self, tsv_file, image_dir):
        self.data = pd.read_csv(tsv_file, sep="\t", names=["caption", "url"], engine='python')
        self.image_dir = image_dir


    def __len__(self):
        return len(self.data)

    def _load_image(self, idx):
        filename = os.path.join(self.image_dir, f"{idx}.jpg")
        
        # 画像が存在するかを確認し、存在する場合は読み込む
        if os.path.exists(filename):
            try:
                # 画像を開く
                return Image.open(filename)
            except Exception as e:
                # 画像を開く際にエラーが発生した場合はNoneを返す
                return None
        else:
            return None

    def __getitem__(self, idx):
        caption = self.data.iloc[idx, 0]
        image = self._load_image(idx)
        
        # 画像が読み込めなかった場合、次のインデックスを試す
        while image is None and idx < len(self.data) - 1:
            idx += 1
            caption = self.data.iloc[idx, 0]
            image = self._load_image(idx)
        
        # すべての画像の読み込みが失敗した場合
        if image is None:
            return None, None

        try:
            # 前処理を実施
            image = preprocess(image).to(device)
            caption = clip.tokenize([caption]).squeeze(0).to(device)
        except Exception as e:
            return None, None  # エラーハンドリング
        
        return image, caption


def custom_collate_fn(batch):
    # Noneデータをフィルタリング
    batch = list(filter(lambda x: x[0] is not None and x[1] is not None, batch))
    
    # デフォルトのcollate_fnを使用してバッチを作成
    return torch.utils.data.dataloader.default_collate(batch)

# データローダーの定義
train_dataset = CustomDataset("Train_GCC-training.tsv", "train_images")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

val_dataset = CustomDataset("Validation_GCC-1.1.0-Validation.tsv", "val_images")
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

# # データのサンプルを取得
# dataset = CustomDataset("Train_GCC-training.tsv", "train_images")
# image, caption = dataset[1]
# print(image.shape)
# print(caption.shape)

# 損失関数クラスの定義
class proposed_clip_loss(nn.Module):
    def __init__(self):
        super(proposed_clip_loss, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    
    def forward(self, T, I):
        # コサイン類似度の計算
        T_dot_T = self.cos(T, T)
        I_dot_T = self.cos(I, T)
        T_dot_I = self.cos(T, I)

        # L_{I,T} と L_{T,I} の計算
        L_I_T = torch.mean(torch.abs(T_dot_T - I_dot_T))
        L_T_I = torch.mean(torch.abs(T_dot_T - T_dot_I))
        
        loss = (L_I_T + L_T_I) / 2.0
        return loss

# 最適化アルゴリズムの定義
optimizer = optim.SGD(model.parameters(), lr=1e-3)
criterion = proposed_clip_loss()

# 学習ループ
loss_list = []
val_loss_list = []

# 学習の進行状況や損失の履歴を保存するログファイルを開く
with open(f"logs/training_log_{num_epoch}_{batch_size}.txt", "w") as log_file:
    for epoch in range(num_epoch):
        print(f"Starting epoch {epoch + 1}/{num_epoch}")
        log_file.write(f"Starting epoch {epoch + 1}/{num_epoch}\n")
        print("Training...")

        # 一時的な学習損失のリストを初期化
        temp_loss_list = []

        for images, captions in tqdm(train_loader):
            # 画像とキャプションのペアをモデルに入力
            with torch.cuda.amp.autocast():
                image_features = model.encode_image(images)
                text_features = model.encode_text(captions)
                loss = criterion(text_features, image_features)
                temp_loss_list.append(loss.item())

            # 勾配を0に初期化
            optimizer.zero_grad()
            # 勾配を計算
            loss.backward()
            
            # パラメータの更新
            optimizer.step()

        avg_train_loss = sum(temp_loss_list) / len(temp_loss_list)
        loss_list.append(avg_train_loss)

        print("Validating...")
        # 検証データでの損失を計算
        with torch.no_grad():
            val_loss = 0
            for images, captions in tqdm(val_loader):
                # 画像とキャプションのペアをモデルに入力
                with torch.cuda.amp.autocast():
                    image_features = model.encode_image(images)
                    text_features = model.encode_text(captions)
                    loss = criterion(text_features, image_features)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            val_loss_list.append(val_loss)

        # ログを出力
        log_msg = f"epoch: {epoch+1}, loss: {loss.item():.4f}, val_loss: {val_loss:.4f}\n"
        print(log_msg)
        log_file.write(log_msg)

        # 各エポックの終了時にモデルと最適化アルゴリズムの状態を保存する
        os.makedirs("models/set2", exist_ok=True)
        os.makedirs(f"models/set2/progress_{num_epoch}_{batch_size}", exist_ok=True)
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_list': loss_list,
            'val_loss_list': val_loss_list
        }
        torch.save(checkpoint, f"models/set2/progress_{num_epoch}_{batch_size}/checkpoint_epoch_{epoch+1}.pth")
        
# 学習曲線を保存
os.makedirs("logs/set2", exist_ok=True)
pd.DataFrame({"loss": loss_list, "val_loss": val_loss_list}).plot()
plt.savefig(f"logs/set2/loss_{num_epoch}_{batch_size}.png")

# 学習済みモデルの保存
torch.save(model.state_dict(), f"models/set2/finetuned_clip_model_{num_epoch}_{batch_size}.pth")
