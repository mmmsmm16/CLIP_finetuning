"""
学習済みモデルを用いたテスト
- PCAを用いた埋め込み空間の可視化
- テストデータを用いたコサイン類似度の可視化
"""
import clip
import argparse
import torch
import numpy as np
import os
import skimage
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib import colors, cm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lm", type=str, default="set1", help="load model")
    args = parser.parse_args()
    return args
# メイン関数
def main():
    args = parse_args()
    ftmodel_path = f"/home/tsuneda/data_dl/CLIP_finetuning/output/{args.lm}/models/finetuned_clip_model_5_128.pth"
    # checkpoint = f"/home/tsuneda/data_dl/CLIP_finetuning/output/checkpoint_epoch_3.pth"
    output_path = f"./output/{args.lm}/test"
    os.makedirs(output_path, exist_ok=True)
    model, preprocess = load_ftmodel(ftmodel_path, "cuda")
    # model, preprocess = load_checkpoint(torch.load(checkpoint), "cuda")
    original_images, images, texts = create_test_image(preprocess)
    image_features, text_features, similarity = compute_similarity(model, images, texts)
    # print(f"image_features:", image_features.shape)
    # print(f"text_features:", text_features.shape)
    visualize_embedding_space(image_features, text_features, output_path)
    visualize_similarity(original_images, texts, similarity, output_path)

# ファインチューニングしたCLIPモデルをロードする関数
def load_ftmodel(model_path, device):
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.load_state_dict(torch.load(model_path))
    return model, preprocess

# チェックポイントのモデルをロードする関数
def load_checkpoint(checkpoint, device):
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, preprocess

# テストテキストの作成
descriptions = {
    "page": "a page of text about segmentation",
    "chelsea": "a facial photo of a tabby cat",
    "astronaut": "a portrait of an astronaut with the American flag",
    "rocket": "a rocket standing on a launchpad",
    "motorcycle_right": "a red motorcycle standing in a garage",
    "camera": "a person looking at a camera on a tripod",
    "horse": "a black-and-white silhouette of a horse", 
    "coffee": "a cup of coffee on a saucer"
}

# テスト画像の作成関数
def create_test_image(preprocess):
    original_images = []
    images = []
    texts = []

    for filename in [filename for filename in os.listdir(skimage.data_dir) if filename.endswith(".png") or filename.endswith(".jpg")]:
        name = os.path.splitext(filename)[0]
        if name not in descriptions:
            continue
        image = Image.open(os.path.join(skimage.data_dir, filename)).convert("RGB")
        original_images.append(image)
        images.append(preprocess(image))
        texts.append(descriptions[name])

    return original_images, images, texts

# 画像特徴量とテキスト特徴量の類似度を計算する関数
def compute_similarity(model, images, texts):
    image_input = torch.tensor(np.stack(images)).cuda()
    text_tokens = clip.tokenize(["This is " + desc for desc in texts]).cuda()
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_tokens)

    # 類似度を計算する
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
    return image_features, text_features, similarity

# image_featuresとtext_featuresを用いた埋め込み空間の可視化
# ペアの画像とテキストの埋め込みを同じ色でプロットする
def visualize_embedding_space(image_features, text_features, output_path):
    pca = PCA(n_components=2)
    vis_data = pca.fit_transform(np.concatenate([image_features.cpu().numpy(), text_features.cpu().numpy()]))
    image_vis_data = vis_data[:len(image_features)]
    text_vis_data = vis_data[len(image_features):]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("Image and Text Embeddings Aligned with PCA")

    # 寄与率を取得
    explained_variances = pca.explained_variance_ratio_
    variance_explained = f"PC1: {explained_variances[0]:.2f}, PC2: {explained_variances[1]:.2f}"

    # 寄与率をサブタイトルに追加
    ax.set_xlabel(f"Principal Component 1 ({explained_variances[0]:.2%} variance)")
    ax.set_ylabel(f"Principal Component 2 ({explained_variances[1]:.2%} variance)")

    ax.scatter(image_vis_data[:, 0], image_vis_data[:, 1], c="r", label='Image Features')
    ax.scatter(text_vis_data[:, 0], text_vis_data[:, 1], c="b", label='Text Features')

    for i, (x, y) in enumerate(image_vis_data):
        ax.annotate(i, (x, y))

    for i, (x, y) in enumerate(text_vis_data):
        ax.annotate(i, (x, y))

    # 凡例を追加
    ax.legend()

    # 寄与率を表示
    plt.figtext(0.5, 0.01, variance_explained, ha="center", fontsize=10)

    # 画像を保存する
    plt.savefig(f"{output_path}/embedding_space.png")
    plt.close(fig)  # ファイル保存後に図を閉じる

# コサイン類似度を可視化する関数
def visualize_similarity(original_images, texts, similarity, output_path):
    count = len(texts)
    fig, ax = plt.subplots(figsize=(20, 14))

    # カラーマップとして 'jet' を使用し、コントラストを高める
    cmap = cm.jet
    norm = colors.Normalize(vmin=0, vmax=1)

    for x in range(similarity.shape[1]):
        for y in range(similarity.shape[0]):
            ax.add_patch(plt.Rectangle((x-0.5, y-0.5), 1, 1, color=cmap(norm(similarity[y, x]))))

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)

    ax.set_yticks(range(count))
    ax.set_yticklabels(texts, fontsize=18)
    ax.set_xticks([])
    for i, image in enumerate(original_images):
        ax.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
    for x in range(similarity.shape[1]):
        for y in range(similarity.shape[0]):
            ax.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)

    for side in ["left", "top", "right", "bottom"]:
        ax.spines[side].set_visible(False)

    ax.set_xlim([-0.5, count - 0.5])
    ax.set_ylim([count + 0.5, -2])

    ax.set_title("Cosine similarity between text and image features", size=20)
    plt.savefig(f"{output_path}/similarity.png")
    plt.close()  # ファイル保存後に図を閉じる

# メイン関数の実行
if __name__ == "__main__":
    main()
