import os 
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import clip
from torch.utils.data import DataLoader, Dataset, Subset
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import torch.nn.functional as F
import random
# from torchinfo import summary

def parse_args():
    parser = argparse.ArgumentParser(description='Finetune CLIP with custom settings')
    parser.add_argument('--set', type=str, required=True, help='name of the Experiment setting')
    parser.add_argument('--test', action='store_true', help='test mode')
    
    args = parser.parse_args()
    return args

# yamlファイルの読み込みする関数
def yaml_load(set):
    with open(f"config/{set}.yml") as file:
        return yaml.safe_load(file)

# フォルダ名の確認
# 既に同じ名前のフォルダが存在する場合は、エラーを返す
def get_next_folder_name(folder_name):
    folder_list = os.listdir("./output")
    if folder_name in folder_list:
        raise ValueError(f"Folder name {folder_name} already exists")
    else:
        return folder_name

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

# 損失関数クラスの定義
class CrossToTextSimilarityLoss(nn.Module):
    # テキスト間のコサイン類似度にテキスト-画像間のコサイン類似度を近づける損失関数
    def __init__(self):
        super(CrossToTextSimilarityLoss, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, txtf, imgf):
        # コサイン類似度の計算
        T_dot_T = self.cos(txtf, txtf)
        I_dot_T = self.cos(imgf, txtf)

        # L_{I,T} と L_{T,I} の計算
        loss = torch.mean(torch.abs(T_dot_T - I_dot_T))
        return loss

# ClipLoss損失関数クラスの定義  
class ClipLoss(nn.Module):
    def __init__(self):
        super(ClipLoss, self).__init__()

    def forward(self, txtf, imgf):
        txtf = txtf / txtf.norm(2, dim=-1, keepdim=True)
        imgf = imgf / imgf.norm(2, dim=-1, keepdim=True)

        mma = (txtf @ imgf.T)/0.01

        labels = torch.arange(mma.shape[0], device=mma.device)
        loss1 = F.cross_entropy(mma, labels)
        loss2 = F.cross_entropy(mma.T, labels)
        loss = (loss1 + loss2)/2.0
        return loss
    
# Delta損失関数クラスの定義
class DeltaLoss(nn.Module):
    def __init__(self):
        super(DeltaLoss, self).__init__()

    def forward(self, txtf, imgf):
        # モダリティ内の差分ベクトルの計算
        device = txtf.device
        n, _ = txtf.size()
        txt_delta_vectors = []
        img_delta_vectors = []

        # デルタベクトルの計算
        txt_delta_vectors = [(txtf[i] - txtf[j]).div((txtf[i] - txtf[j]).norm(2, dim=-1, keepdim=True) + 1e-8) 
                            for i in range(n) for j in range(n) if i != j]
        img_delta_vectors = [(imgf[i] - imgf[j]).div((imgf[i] - imgf[j]).norm(2, dim=-1, keepdim=True) + 1e-8) 
                            for i in range(n) for j in range(n) if i != j]

        # NaNを含む要素のインデックスを特定
        nan_indices = set()
        for idx, (txt_vec, img_vec) in enumerate(zip(txt_delta_vectors, img_delta_vectors)):
            if torch.isnan(txt_vec).any() or torch.isnan(img_vec).any():
                nan_indices.add(idx)

        # NaNを含む要素を削除
        txt_delta_vectors = [vec for idx, vec in enumerate(txt_delta_vectors) if idx not in nan_indices]
        img_delta_vectors = [vec for idx, vec in enumerate(img_delta_vectors) if idx not in nan_indices]

        # テンソルのリストをスタック
        if txt_delta_vectors and img_delta_vectors:
            txt_delta_vectors = torch.stack(txt_delta_vectors)
            img_delta_vectors = torch.stack(img_delta_vectors)
        else:
            # 空の場合の代替処理
            return torch.tensor(0.0, device=device, requires_grad=True)


        mma = (txt_delta_vectors @ img_delta_vectors.T)/0.01
        labels = torch.arange(mma.shape[0], device=mma.device)
        loss1 = F.cross_entropy(mma, labels)
        loss2 = F.cross_entropy(mma.T, labels)
        loss = (loss1 + loss2)/2.0
        
        return loss

# 組み合わせ損失関数クラスの定義
class CombineLoss(nn.Module):
    # CrossToTextSimilarityLossとClipLossを組み合わせた損失関数
    def __init__(self, alpha=0.5):
        super(CombineLoss, self).__init__()
        self.alpha = alpha
        self.cross_to_text_similarity_loss = CrossToTextSimilarityLoss()
        self.clip_loss = ClipLoss()

    def forward(self, txtf, imgf):
        loss1 = self.cross_to_text_similarity_loss(txtf, imgf)
        loss2 = self.clip_loss(txtf, imgf)
        loss = self.alpha * loss1 + (1 - self.alpha) * loss2
        return loss

# DeltaLossとClipLossを組み合わせた損失関数クラスの定義
class DeltaClipLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(DeltaClipLoss, self).__init__()
        self.alpha = alpha
        self.delta_loss = DeltaLoss()
        self.clip_loss = ClipLoss()

    def forward(self, txtf, imgf):
        loss1 = self.delta_loss(txtf, imgf)
        loss2 = self.clip_loss(txtf, imgf)
        loss = self.alpha * loss1 + (1 - self.alpha) * loss2
        return loss

# データセットクラスの定義
class CustomDataset(Dataset):
    def __init__(self, tsv_file, image_dir, preprocess, device):
        self.data = pd.read_csv(tsv_file, sep="\t", names=["caption", "url"], engine='python')
        self.image_dir = image_dir
        self.preprocess = preprocess
        self.device = device

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
        while idx < len(self.data):
            caption = self.data.iloc[idx, 0]
            image = self._load_image(idx)

            if image is not None:
                try:
                    # 前処理を実施
                    image = self.preprocess(image).to(self.device)
                    caption = clip.tokenize([caption]).squeeze(0).to(self.device)
                    return image, caption
                except Exception as e:
                    pass

            idx += 1  # 次のインデックスに進む

        # すべての画像の読み込みが失敗した場合
        return None, None

def custom_collate_fn(batch):
    # Noneデータをフィルタリング
    batch = list(filter(lambda x: x[0] is not None and x[1] is not None, batch))
    
    # デフォルトのcollate_fnを使用してバッチを作成
    return torch.utils.data.dataloader.default_collate(batch)

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

# 学習ループの定義
def training_loop(num_epoch, batch_size, train_loader, val_loader, model, optimizer, criterion, log_file_path, model_save_path):
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
            # avg_train_lossがnanの場合知らせる
            if avg_train_loss != avg_train_loss:
                raise ValueError("Loss becomes nan")
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
                # val_lossがnanの場合知らせる
                if val_loss != val_loss:
                    raise ValueError("Loss becomes nan")
                
                val_loss_list.append(val_loss)

            log_msg = f"epoch: {epoch+1}, loss: {loss.item():.4f}, val_loss: {val_loss:.4f}\n"
            print(log_msg)
            log_file.write(log_msg)
            save_checkpoint(model, optimizer, loss_list, val_loss_list, epoch, model_save_path)

    return loss_list, val_loss_list

# テキストエンコーダの重みを固定する関数
def freeze_text_params(model):
    for param in model.transformer.parameters():
        param.requires_grad = False
    # token_embedding層を固定
    for param in model.token_embedding.parameters():
        param.requires_grad = False
    # ln_final層を固定
    for param in model.ln_final.parameters():
        param.requires_grad = False
    # positional_embedding層を固定
    model.positional_embedding.requires_grad = False
    # text_projection層を固定
    model.text_projection.requires_grad = False
    print("Text parameters are frozen")

# 学習済みモデルの保存
def save_model(model, save_dir):
    save_dir = os.path.join(save_dir, "models")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{save_dir}/final_model.pth")

def main():
    # コマンドライン引数の取得
    args = parse_args()

    # 設定ファイルの読み込み
    config = yaml_load(args.set)
    print(f"{config['set']}の実験です")
    num_epoch = config["num_epoch"]
    batch_size = config["batch_size"]
    if args.test:
        print('testです')
        output_dir_name = get_next_folder_name(config["set"] + "_test")
        model_save_path, log_save_path = define_output_dir(output_dir_name)

        # log_file_pathの作成
        log_file_path = os.path.join(log_save_path, "log.txt")

        # 学習済みモデルのロード
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)

        # データセットの定義
        train_dataset = CustomDataset("Train_GCC-training.tsv", "train_images", preprocess=preprocess, device=device)
        val_dataset = CustomDataset("Validation_GCC-1.1.0-Validation.tsv", "val_images", preprocess=preprocess, device=device)

        # テストモードで使用するデータの数
        num_samples = 1000

        # データセットからランダムにサンプルを選択
        train_indices = random.sample(range(len(train_dataset)), num_samples)
        val_indices = random.sample(range(len(val_dataset)), num_samples)

        # サブセットの作成
        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(val_dataset, val_indices)

        # データローダーの定義
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

        # 最適化アルゴリズムの定義
        optimizer = optim.SGD(model.parameters(), lr=float(config["learning_rate"]))
        
        # 損失関数の定義
        if config['loss_function'] == "CrossToTextSimilarityLoss":
            criterion = CrossToTextSimilarityLoss()
        elif config['loss_function'] == "ClipLoss":
            criterion = ClipLoss()
        elif config['loss_function'] == "CombineLoss":
            criterion = CombineLoss()
        elif config['loss_function'] == "DeltaClipLoss":
            criterion = DeltaClipLoss()
        else:
            raise ValueError(f"Invalid loss function: {config['loss_function']}")

        # パラメータ固定
        if config['freeze_text_params']:
            # テキストエンコーダの重みを固定
            freeze_text_params(model)
    
        # 学習ループ
        loss_list, val_loss_list = training_loop(num_epoch, batch_size, train_loader, val_loader, model, optimizer, criterion, log_file_path, model_save_path)

        # 学習曲線を保存
        save_training_curve(loss_list, val_loss_list, log_save_path)

        # 学習済みモデルの保存
        save_model(model, model_save_path)

    else:
        # outputディレクトリの作成とパスの取得
        output_dir_name = get_next_folder_name(config["set"])
        model_save_path, log_save_path = define_output_dir(output_dir_name)

        # log_file_pathの作成
        log_file_path = os.path.join(log_save_path, "log.txt")
        
        # 学習済みモデルのロード
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)

        # データローダーの定義
        train_dataset = CustomDataset("Train_GCC-training.tsv", "train_images", preprocess=preprocess, device=device)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

        val_dataset = CustomDataset("Validation_GCC-1.1.0-Validation.tsv", "val_images", preprocess=preprocess, device=device)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

        # 最適化アルゴリズムの定義
        optimizer = optim.SGD(model.parameters(), lr=float(config["learning_rate"]))
        
        # 損失関数の定義
        if config['loss_function'] == "CrossToTextSimilarityLoss":
            criterion = CrossToTextSimilarityLoss()
        elif config['loss_function'] == "ClipLoss":
            criterion = ClipLoss()
        elif config['loss_function'] == "CombineLoss":
            criterion = CombineLoss()
        elif config['loss_function'] == "DeltaClipLoss":
            criterion = DeltaLoss()
        else:
            raise ValueError(f"Invalid loss function: {config['loss_function']}")

        # パラメータ固定
        if config['freeze_text_params']:
            # テキストエンコーダの重みを固定
            freeze_text_params(model)
    
        # 学習ループ
        loss_list, val_loss_list = training_loop(num_epoch, batch_size, train_loader, val_loader, model, optimizer, criterion, log_file_path, model_save_path)

        # 学習曲線を保存
        save_training_curve(loss_list, val_loss_list, log_save_path)

        # 学習済みモデルの保存
        save_model(model, model_save_path)

if __name__ == "__main__":
    main()
