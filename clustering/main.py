import json
from compare_clastering import compare_clustering_methods

def main():
    # JSONファイルのパス
    json_file_path = 'caption_embedding_vector.json'

    # ファイルからJSONデータを読み込む
    with open(json_file_path, 'r') as file:
        json_data_with_vector = json.load(file)

    json_added_result,num_clusters = compare_clustering_methods(json_data_with_vector)
    
    with open("clustering_result.json", "w") as f:
        json.dump(json_added_result, f, indent=2)

if __name__ =='__main__':
    main()
