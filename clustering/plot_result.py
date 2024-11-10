import os
import matplotlib.pyplot as plt
import cv2

def save_clustered_images(json_data, method_key, output_filename):
    # クラスタ数の取得
    clusters = set(item[method_key] for item in json_data)

    # クラスタごとにプロット
    fig, axes = plt.subplots(len(clusters), 1, figsize=(20, 5 * len(clusters)))
    if len(clusters) == 1:
        axes = [axes]  # クラスタが1つだけの場合の処理

    for cluster_index, ax in zip(clusters, axes):
        cluster_images = [item['path'] for item in json_data if item[method_key] == cluster_index]
        print(cluster_images)

        # 各クラスタごとの画像を横に並べて表示
        for idx, image_path in enumerate(cluster_images):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            ax.imshow(image)
            ax.set_title(f"Cluster {cluster_index}", fontsize=16)
            ax.axis('off')
            
        fig.tight_layout()
        plt.savefig(output_filename)

