import os
import argparse
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

# コマンドライン引数のパース
def parse_args():
    parser = argparse.ArgumentParser(description='Finetune CLIP with custom settings')
    parser.add_argument('--num_epoch', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='set', help='Output directory')
    parser.add_argument('--freeze_text_params', action='store_true', help='Freeze text parameters')
    parser.add_argument('--freeze_image_params', action='store_true', help='Freeze image parameters')
    parser.add_argument('--loss_function', type=str, default='CrossToTextSimilarityLoss', choices=['CrossToTextSimilarityLoss', 'InfoNCELoss'], help='Loss function to use. Chose from [CrossToTextSimilarityLoss, InfoNCELoss]')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD'], help='Optimizer to use. Chose from [Adam, SGD]')

    args = parser.parse_args()
    return args

# データセットクラスの定義
class CustomDataset(Dataset):
    def __init__(self, tsv_file, image_dir, preprocess, device):
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
            image = self.preprocess(image).to(self.device)
            caption = clip.tokenize([caption]).squeeze(0).to(self.device)
        except Exception as e:
            return None, None  # エラーハンドリング

        return image, caption

# 損失関数クラスの定義
class CrossToTextSimilarityLoss(nn.Module):
    # テキスト間のコサイン類似度にテキスト-画像間のコサイン類似度を近づける損失関数
    def __init__(self):
        super(CrossToTextSimilarityLoss, self).__init__()
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

def custom_collate_fn(batch):
    # Noneデータをフィルタリング
    batch = list(filter(lambda x: x[0] is not None and x[1] is not None, batch))

    # デフォルトのcollate_fnを使用してバッチを作成
    return torch.utils.data.dataloader.default_collate(batch)

# 学習ループの定義
def training_loop(num_epoch, batch_size, train_loader, val_loader, model, optimizer, criterion, log_file_path):
    loss_list = []
    val_loss_list = []

    with open(log_file_path, "w") as log_file:
        for epoch in range(num_epoch):
            print(f"Starting epoch {epoch + 1}/{num_epoch}")
            log_file.write(f"Starting epoch {epoch + 1}/{num_epoch}\n")
            print("Training...")
            temp_loss_list = []

            for images, captions in tqdm(train_loader):
                with torch.cuda.amp.autocast():
                    image_features = model.encode_image(images)
                    text_features = model.encode_text(captions)
                    loss = criterion(text_features, image_features)
                    temp_loss_list.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_train_loss = sum(temp_loss_list) / len(temp_loss_list)
            loss_list.append(avg_train_loss)

            print("Validating...")
            with torch.no_grad():
                val_loss = 0
                for images, captions in tqdm(val_loader):
                    with torch.cuda.amp.autocast():
                        image_features = model.encode_image(images)
                        text_features = model.encode_text(captions)
                        loss = criterion(text_features, image_features)
                        val_loss += loss.item()
                val_loss /= len(val_loader)
                val_loss_list.append(val_loss)

            log_msg = f"epoch: {epoch+1}, loss: {loss.item():.4f}, val_loss: {val_loss:.4f}\n"
            print(log_msg)
            log_file.write(log_msg)
            save_checkpoint(model, optimizer, loss_list, val_loss_list, epoch, batch_size)

    return loss_list, val_loss_list

# チェックポイントの保存
def save_checkpoint(model, optimizer, loss_list, val_loss_list, epoch, save_dir):
    checkpoint_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_list': loss_list,
        'val_loss_list': val_loss_list
    }
    torch.save(checkpoint, f"{checkpoint_dir}/epoch_{epoch+1}.pth")

# outputディレクトリの作成とパスの取得
def define_output_dir(name):
    output_dir = os.path.join("./output", name)
    os.makedirs(output_dir, exist_ok=True)
    model_save_path = os.path.join(output_dir, "models")
    os.makedirs(model_save_path, exist_ok=True)
    log_file_path = os.path.join(output_dir, "logs")
    os.makedirs(log_file_path, exist_ok=True)
    return model_save_path, log_file_path

# 学習曲線の保存
def save_training_curve(loss_list, val_loss_list, save_dir):
    save_dir = os.path.join(save_dir, "logs")
    os.makedirs(save_dir, exist_ok=True)
    pd.DataFrame({"loss": loss_list, "val_loss": val_loss_list}).plot()
    plt.savefig(f"{save_dir}/training_curve.png")

# 学習済みモデルの保存
def save_model(model, save_dir):
    save_dir = os.path.join(save_dir, "models")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{save_dir}/finel_model.pth")

# フォルダ名の取得
def get_next_folder_name(base_name="set"):
    idx = 1
    while os.path.exists(f"./output/{base_name}{idx}"):
        idx += 1
    return f"{base_name}{idx}"

# メイン関数
def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # パラメータ固定
    if args.freeze_text_params:
        # テキストエンコーダの重みを固定
        for param in model.transformer.parameters():
            param.requires_grad = False
        # token_embedding層を固定
        for param in model.token_embedding.parameters():
            param.requires_grad = False
        # ln_final層を固定
        for param in model.ln_final.parameters():
            param.requires_grad = False
        print("Text parameters are frozen")
    if args.freeze_image_params:
        for param in model.visual.parameters():
            param.requires_grad = False
        print("Image parameters are frozen")
        
    # データセットの読み込み
    train_dataset = CustomDataset("Train_GCC-training.tsv", "train_images", preprocess, device)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)

    val_dataset = CustomDataset("Validation_GCC-1.1.0-Validation.tsv", "val_images", preprocess, device)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)

    # 損失関数の定義
    if args.loss_function == "CrossToTextSimilarityLoss":
        criterion = CrossToTextSimilarityLoss()
    elif args.loss_function == "InfoNCELoss":
        criterion = clip.loss.InfoNCELoss()
    else:
        raise ValueError(f"Invalid loss function: {args.loss_function}")

    # 最適化アルゴリズムの定義
    if args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    else:
        raise ValueError(f"Invalid optimizer: {args.optimizer}")

    # 出力ディレクトリの作成
    set_name = get_next_folder_name(args.output_dir)
    model_save_path, log_file_path = define_output_dir(set_name)
    print("Output directory:", set_name)

    # 学習の実行
    loss_list, val_loss_list = training_loop(args.num_epoch, args.batch_size, train_loader, val_loader, model, optimizer, criterion, f"{log_file_path}/training_log.txt")

    # 学習曲線の保存
    save_training_curve(loss_list, val_loss_list, model_save_path)

# メイン関数の実行
if __name__ == "__main__":
    main()
