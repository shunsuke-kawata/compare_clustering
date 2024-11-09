from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import json
import numpy as np

# JSONファイルからベクトルデータを読み込む
with open('updated_embeddings.json', 'r') as file:
    data = json.load(file)

# ベクトルを抽出
vectors = [np.array(item['embedding']) for item in data]

# コサイン類似度マトリクスを計算
similarity_matrix = cosine_similarity(vectors)

# KMeansクラスタリングを実行
num_clusters = 5  # クラスタの数
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
clusters = kmeans.fit_predict(similarity_matrix)

# クラスタリングの結果を表示
for idx, cluster in enumerate(clusters):
    print(f'Filename: {data[idx]["filename"]} - Cluster: {cluster}')

