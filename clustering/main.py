import json
from compare_clastering import compare_clustering_methods
import csv
from classify_folder import classify_result

def main():
    # JSONファイルのパス
    json_file_path = 'caption_embedding_vector.json'

    # ファイルからJSONデータを読み込む
    with open(json_file_path, 'r') as file:
        json_data_with_vector = json.load(file)

    json_data_cp,evalation_dict = compare_clustering_methods(json_data_with_vector)
    
    with open("clustering_result.json", "w") as f:
        json.dump(json_data_cp, f, indent=2)
        
    with open("clustering_eval.json", "w") as f:
        json.dump(evalation_dict, f, indent=2)
        
    methods = list(evalation_dict.keys())
    metrics = list(evalation_dict[methods[0]].keys())

    # CSVに書き込み
    with open("clustering_evaluation_scores.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        
        # ヘッダー行として評価指標を追加
        writer.writerow(["Metric"] + metrics)
        
        # 各クラスタリング手法ごとに列を作成して書き込み
        for method in methods:
            row = [method] + [evalation_dict[method][metric] for metric in metrics]
            writer.writerow(row)
    
    classify_result()

if __name__ =='__main__':
    main()
