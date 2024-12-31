import json
import os
import shutil

json_file_path = './clustering_result.json'
base_folder_path = "./clustering_result_images/"
folders_path = [
    "class_by_sentence_kmeans",
    "class_by_feature_kmeans",
    "class_by_sentence_xmeans",
    "class_by_feature_xmeans",
    "class_by_sentence_hierarchical",
    "class_by_feature_hierarchical"
]

def get_unique_clusters(result_json, cluster_keys):
    cluster_info = {}
    for key in cluster_keys:
        unique_clusters = set()
        for ele in result_json:
            unique_clusters.add(ele[key])
        cluster_info[key] = sorted(list(unique_clusters))
    return cluster_info

def classify_result():
    # JSONデータをファイルから読み込む
    with open(json_file_path, 'r') as file:
        result_json = json.load(file)
    if os.path.exists(base_folder_path):
        shutil.rmtree(base_folder_path)
        
    cluster_info = get_unique_clusters(result_json,folders_path) # クラスタ数を仮に10としています。実際のクラスタ数に合わせて調整してください。
    for key,val in cluster_info.items():
        for v in val:
            os.makedirs(f"{base_folder_path}{key}/cluster_{v}", exist_ok=True)
    
    # 各データの画像を対応するフォルダにコピー
    for ele in result_json:
        image_path = ele["path"]  # 元の画像ファイルのパス
        filename = ele["filepath"]
        
        for folder_key in folders_path:
            # クラスタ番号を取得して対応フォルダへコピー
            cluster_num = ele[folder_key]
            target_folder = f"{base_folder_path}{folder_key}/cluster_{cluster_num}"
            
            # 対象フォルダに画像をコピー
            target_path = os.path.join(target_folder, filename)
            shutil.copy(image_path, target_path)

if __name__ == '__main__':
    classify_result()