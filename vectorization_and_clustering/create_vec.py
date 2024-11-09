from sentence_transformers import SentenceTransformer
import json

# JSONファイルのパス
json_file_path = 'caption.json'

# ファイルからJSONデータを読み込む
with open(json_file_path, 'r') as file:
    data = json.load(file)

# Sentence-BERTモデルのロード
model = SentenceTransformer('all-MiniLM-L6-v2')

# 各キャプションのベクトル化
for item in data:
    caption = item["caption"]
    embedding = model.encode(caption)  # キャプションをベクトル化
    item["embedding"] = embedding.tolist()  # numpy arrayをリストに変換して保存

# ベクトル化されたデータを確認（例）
for item in data:
    print(f"Filename: {item['filename']}")
    print(f"Embedding: {item['embedding'][:5]}...")  # 部分的に表示
    print()

# 結果を別のJSONファイルとして保存したい場合
with open("updated_embeddings.json", "w") as f:
    json.dump(data, f, indent=2)
