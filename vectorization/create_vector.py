from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image

# 画像の特徴量ベクトルを抽出する関数
def extract_feature_vector(image_path):
    # ResNet50モデルの読み込み（事前学習済みモデル）
    model = resnet50(pretrained=True)
    model.eval()  # 推論モードに設定

    # 特徴量ベクトルを取得するため、最終の分類層を除去
    model = torch.nn.Sequential(*list(model.children())[:-1])

    # 画像の前処理
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 画像の読み込みと前処理の適用
    image = Image.open(image_path)
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # バッチサイズの追加

    # GPUが利用可能な場合はGPUを使用
    if torch.cuda.is_available():
        model = model.to('cuda')
        input_batch = input_batch.to('cuda')

    # 推論
    with torch.no_grad():
        feature_vector = model(input_batch)

    # ベクトルを1次元に変換
    feature_vector = feature_vector.flatten().cpu().numpy()
    return feature_vector

def create_json_with_vector(json_data):
    # Sentence-BERTモデルのロード
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # 各キャプションのベクトル化
    for json_datum in json_data:
        caption = json_datum["caption"]
        embedding_sentence_vector = model.encode(caption)  # キャプションをSentenceTransformerでベクトル化
        json_datum["embedding_sentence_vector"] = embedding_sentence_vector.tolist()
        embedding_image_vector = extract_feature_vector(json_datum["path"])
        json_datum["embedding_image_vector"] = embedding_image_vector.tolist()
    return json_data
        
