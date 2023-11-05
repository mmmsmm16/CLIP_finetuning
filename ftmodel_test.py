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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_model", type=str, default="set1", help="load model")
    args = parser.parse_args()
    return args
# メイン関数
def main():
    args = parse_args()
    ftmodel_path = f"/home/tsuneda/copy/CLIP_finetuning/output/{args.load_model}/models/finetuned_clip_model_5_128.pth"
    output_path = f"./output/{args.load_model}/test"
    os.makedirs(output_path, exist_ok=True)
    model, preprocess = load_ftmodel(ftmodel_path, "cuda")
    original_images, images, texts = create_test_image(preprocess)
    image_features, text_features, similarity = compute_similarity(model, images, texts)
    print(f"image_features:", image_features.shape)
    print(f"text_features:", text_features.shape)
    visualize_embedding_space(image_features, text_features, output_path)
    visualize_similarity(original_images, texts, similarity, output_path)

# ファインチューニングしたCLIPモデルをロードする関数
def load_ftmodel(model_path, device):
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.load_state_dict(torch.load(model_path))
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

    plt.figure(figsize=(10, 10))
    plt.title("Image and Text Embeddings Aligned with PCA")
    plt.scatter(image_vis_data[:, 0], image_vis_data[:, 1], c="r")
    plt.scatter(text_vis_data[:, 0], text_vis_data[:, 1], c="b")
    for i, (x, y) in enumerate(image_vis_data):
        plt.annotate(i, (x, y))

    for i, (x, y) in enumerate(text_vis_data):
        plt.annotate(i, (x, y))

    # 画像を保存する
    plt.savefig(f"{output_path}/embedding_space.png")

# コサイン類似度を可視化する関数
def visualize_similarity(original_images, texts, similarity, output_path):
    count = len(texts)
    plt.figure(figsize=(20, 14))
    plt.imshow(similarity, vmin=0.1, vmax=0.3)
    # plt.colorbar()
    plt.yticks(range(count), texts, fontsize=18)
    plt.xticks([])
    for i, image in enumerate(original_images):
        plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
    for x in range(similarity.shape[1]):
        for y in range(similarity.shape[0]):
            plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)

    for side in ["left", "top", "right", "bottom"]:
        plt.gca().spines[side].set_visible(False)

    plt.xlim([-0.5, count - 0.5])
    plt.ylim([count + 0.5, -2])

    plt.title("Cosine similarity between text and image features", size=20)
    plt.savefig(f"{output_path}/similarity.png")

# メイン関数の実行
if __name__ == "__main__":
    main()
