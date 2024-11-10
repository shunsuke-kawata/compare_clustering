import json
from compare_clastering import compare_clustering_methods
from plot_result import save_clustered_images

def main():
    # JSONファイルのパス
    json_file_path = 'caption_embedding_vector.json'

    # ファイルからJSONデータを読み込む
    with open(json_file_path, 'r') as file:
        json_data_with_vector = json.load(file)

    json_added_result,num_clusters = compare_clustering_methods(json_data_with_vector)
    
        # クラスタリング結果を基に4種類のグラフを保存
    save_clustered_images(json_added_result, 'class_by_sentence_kmeans', 'sentence_kmeans_clusters.png')
    save_clustered_images(json_added_result, 'class_by_image_kmeans', 'image_kmeans_clusters.png')
    save_clustered_images(json_added_result, 'class_by_sentence_hierarchical', 'sentence_hierarchical_clusters.png')
    save_clustered_images(json_added_result, 'class_by_image_hierarchical', 'image_hierarchical_clusters.png')
    
    with open("clustering_result.json", "w") as f:
        json.dump(json_added_result, f, indent=2)

if __name__ =='__main__':
    main()

print(1)
