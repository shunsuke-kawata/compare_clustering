from flask import Flask, render_template, request, send_from_directory
import json
import os

# Flaskアプリのセットアップ
app = Flask(__name__)

# ファイルパス
JSON_FILE_PATH = "./clustering_result.json"
IMAGES_FOLDER_PATH = "./results/"

# クラスタリング結果を読み込み
with open(JSON_FILE_PATH, 'r') as file:
    result_json_data = json.load(file)

# クラスタリングメソッドのキー
CLUSTER_METHODS = [
    "class_by_sentence_kmeans",
    "class_by_feature_kmeans",
    "class_by_sentence_hierarchical",
    "class_by_feature_hierarchical"
]

def get_clusters(method_key):
    # 指定されたクラスタリングメソッドでのクラスタを取得
    print(method_key)
    return sorted(set(item[method_key] for item in result_json_data))

def get_cluster_images(method_key, cluster_id):
    # 指定されたクラスタリングメソッドとクラスタIDに属する画像のリストを取得
    return [
        os.path.join("results", item["filename"])  # パスを相対的に指定
        for item in result_json_data
        if str(item[method_key]) == str(cluster_id)
    ]

@app.route("/results/<path:filename>")
def serve_image(filename):
    # results フォルダ内の画像ファイルを提供
    return send_from_directory(IMAGES_FOLDER_PATH, filename)

@app.route("/", methods=["GET", "POST"])
def index():
    # 初期選択のメソッドとクラスタ番号を設定
    selected_method = request.form.get("method", CLUSTER_METHODS[0])
    selected_cluster = request.form.get("cluster")

    # メソッドに基づいたクラスタのリスト
    clusters = get_clusters(selected_method)

    # クラスタIDが未選択の場合は最初のクラスタに設定
    if selected_cluster is None:
        selected_cluster = clusters[0] if clusters else None

    # 選択されたクラスタの画像リストを取得
    cluster_images = get_cluster_images(selected_method, selected_cluster)

    return render_template(
        "index.html",
        methods=CLUSTER_METHODS,
        selected_method=selected_method,
        clusters=clusters,
        selected_cluster=selected_cluster,
        cluster_images=cluster_images,
        str=str  # strをテンプレートに渡す
    )

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=os.environ.get("PORT"))