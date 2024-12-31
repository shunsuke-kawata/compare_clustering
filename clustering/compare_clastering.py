from pyclustering.cluster import xmeans
from scipy.cluster.hierarchy import linkage, fcluster,dendrogram
from scipy.spatial.distance import pdist
import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans

def save_dendrogram(linkage_matrix, labels, filename):
    plt.figure(figsize=(120, 50))
    dendrogram(linkage_matrix, labels=labels, leaf_rotation=90, leaf_font_size=10)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def evaluate_clustering(embedding_vectors, labels):
    # シルエットスコアの計算
    silhouette_avg = silhouette_score(embedding_vectors, labels)

    # ダビース・ボルディン指数の計算
    davies_bouldin = davies_bouldin_score(embedding_vectors, labels)

    # カリンスキー・ハラバス指数の計算
    calinski_harabasz = calinski_harabasz_score(embedding_vectors, labels)
    
    tmp_eval_dict = {
        "Silhouette Score":silhouette_avg,
        "Davies-Bouldin Index":davies_bouldin,
        "Calinski-Harabasz Index":calinski_harabasz
    }
    return tmp_eval_dict

def execute_clustering_kmeans(json_data, evalation_dict, num_clusters,flag='both'):
    if(flag=='both' or flag=='sentense'):
        # 埋め込みベクトルのリストを取得
        embedding_sentence_vectors = [np.array(json_datum['embedding_sentence_vector']) for json_datum in json_data]

        # キャプションによるクラスタリングの実行
        kmeans_sentence = KMeans(n_clusters=num_clusters, random_state=42)
        sentence_labels = kmeans_sentence.fit_predict(embedding_sentence_vectors)
        
        # クラスタリング結果を json_data に追加
        for index, label in enumerate(sentence_labels):
            json_data[index]['class_by_sentence_kmeans'] = int(label)
    
        # キャプションベクトルに基づくクラスタリング評価
        evalation_dict["KMeans (Sentence Embeddings)"] = evaluate_clustering(embedding_sentence_vectors, sentence_labels)
    if(flag=='both' or flag=='image'):
        embedding_image_vectors = [np.array(json_datum['embedding_feature_vector']) for json_datum in json_data]
        # 画像特徴量によるクラスタリングの実行
        kmeans_image = KMeans(n_clusters=num_clusters, random_state=42)
        image_labels = kmeans_image.fit_predict(embedding_image_vectors)
        
        # クラスタリング結果を json_data に追加
        for index, label in enumerate(image_labels):
            json_data[index]['class_by_feature_kmeans'] = int(label)
        
        # 画像特徴量に基づくクラスタリング評価
        evalation_dict["KMeans (Image Embeddings)"] = evaluate_clustering(embedding_image_vectors, image_labels)
    
def execute_clustering_xmeans(json_data, evalation_dict,num_clusters,flag='both'):
    if(flag=='both' or flag=='sentense'):
        # 埋め込みベクトルのリストを取得
        embedding_sentence_vectors = [np.array(json_datum['embedding_sentence_vector']) for json_datum in json_data]

        # キャプションによるクラスタリングの実行
        i_centers_sentence = xmeans.kmeans_plusplus_initializer(embedding_sentence_vectors, num_clusters).initialize()
        xmeans_sentence_instance = xmeans.xmeans(data=embedding_sentence_vectors, initial_centers=i_centers_sentence, ccore=False)
        xmeans_sentence_instance.process()
        
        # クラスタリング結果を取得
        clusters_by_sentence = xmeans_sentence_instance.get_clusters()
        sentence_labels = np.zeros(len(embedding_sentence_vectors))
        for cluster_index, cluster in enumerate(clusters_by_sentence):
            for index in cluster:
                sentence_labels[index] = cluster_index
                json_data[index]['class_by_sentence_xmeans'] = cluster_index
        
        # キャプションベクトルに基づくクラスタリング評価
        evalation_dict["XMeans (Sentence Embeddings)"] =  evaluate_clustering(embedding_sentence_vectors, sentence_labels)
        
    if(flag=='both' or flag=='image'):
        embedding_image_vectors = [np.array(json_datum['embedding_feature_vector']) for json_datum in json_data]
        
        # 画像特徴量によるクラスタリングの実行
        i_centers_image = xmeans.kmeans_plusplus_initializer(embedding_image_vectors, num_clusters).initialize()
        xmeans_image_instance = xmeans.xmeans(data=embedding_image_vectors, initial_centers=i_centers_image, ccore=False)
        xmeans_image_instance.process()
        
        # クラスタリング結果を取得
        clusters_by_image = xmeans_image_instance.get_clusters()
        image_labels = np.zeros(len(embedding_image_vectors))
        for cluster_index, cluster in enumerate(clusters_by_image):
            for index in cluster:
                image_labels[index] = cluster_index
                json_data[index]['class_by_feature_xmeans'] = cluster_index
        
        # 画像特徴量に基づくクラスタリング評価
        evalation_dict["XMeans (Image Embeddings)"]  = evaluate_clustering(embedding_image_vectors, image_labels)

def execute_clustering_hierarchical(json_data, evalation_dict,num_clusters,flag='both'):
    if(flag=='both' or flag=='sentense'):
        # キャプションベクトルによる階層型クラスタリング
        sentence_vectors = [np.array(json_datum['embedding_sentence_vector']) for json_datum in json_data]
        sentence_distance_matrix = pdist(sentence_vectors, metric='cosine')
        sentence_linkage_matrix = linkage(sentence_distance_matrix, method='average')
        sentence_clusters = fcluster(sentence_linkage_matrix, num_clusters, criterion='maxclust')
    
        # クラスタリング結果を json_data に追加
        for idx, json_datum in enumerate(json_data):
            json_datum['class_by_sentence_hierarchical'] = int(sentence_clusters[idx])
            
        dendrogram_labels = [json_datum['path'] for json_datum in json_data]
        
        # キャプションベクトルに基づくクラスタリング評価
        evalation_dict["Hierarchical (Sentence Embeddings)"] = evaluate_clustering(sentence_vectors, sentence_clusters)
        save_dendrogram(sentence_linkage_matrix, labels=dendrogram_labels, filename="sentence_dendrogram.png")

    if(flag=='both' or flag=='image'):
        # 画像ベクトルによる階層型クラスタリング
        image_vectors = [np.array(json_datum['embedding_feature_vector']) for json_datum in json_data]
        image_distance_matrix = pdist(image_vectors, metric='cosine')
        image_linkage_matrix = linkage(image_distance_matrix, method='average')
        image_clusters = fcluster(image_linkage_matrix, num_clusters, criterion='maxclust')

        # クラスタリング結果を json_data に追加
        for idx, json_datum in enumerate(json_data):
            json_datum['class_by_feature_hierarchical'] = int(image_clusters[idx])
        
        dendrogram_labels = [json_datum['path'] for json_datum in json_data]
        # デンドログラムを保存
        
        # 画像特徴量に基づくクラスタリング評価
        evalation_dict["Hierarchical (Image Embeddings)"] = evaluate_clustering(image_vectors, image_clusters)
        save_dendrogram(image_linkage_matrix, labels=dendrogram_labels, filename="feature_dendrogram.png")

def compare_clustering_methods(json_data):
    # json_dataをディープコピーして操作用に作成
    json_data_cp = copy.deepcopy(json_data)
    
    evalation_dict = {}
    
    # KMeansによるクラスタリング結果を追加
    execute_clustering_kmeans(json_data_cp, evalation_dict, num_clusters=17,flag='both')
    # XMeansによるクラスタリング結果を追加
    execute_clustering_xmeans(json_data_cp, evalation_dict,num_clusters=17,flag='both')
    # 階層的クラスタリングによる結果を追加
    execute_clustering_hierarchical(json_data_cp, evalation_dict,num_clusters=12,flag='both')

    return json_data_cp,evalation_dict