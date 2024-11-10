from sentence_transformers import SentenceTransformer
import numpy as np


def create_json_with_vector(json_data):
    # Sentence-BERTモデルのロード
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # 各キャプションのベクトル化
    for json_datum in json_data:
        caption = json_datum["caption"]
        embedding = model.encode(caption)  # キャプションをSentenceTransformerでベクトル化
        json_datum["embedding_sentence_vector"] = embedding.tolist()
    
    return json_data
        
