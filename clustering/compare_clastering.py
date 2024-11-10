from pyclustering.cluster import xmeans
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import numpy as np
import copy

def execute_clustering_xmeans(json_data, num_clusters):
    # 埋め込みベクトルのリストを取得
    embedding_sentence_vectors = [np.array(json_datum['embedding_sentence_vector']) for json_datum in json_data]
    embedding_image_vectors = [np.array(json_datum['embedding_image_vector']) for json_datum in json_data]

    # キャプションによるクラスタリングの実行
    i_centers_sentence = xmeans.kmeans_plusplus_initializer(embedding_sentence_vectors, num_clusters).initialize()
    xmeans_sentence_instance = xmeans.xmeans(data=embedding_sentence_vectors, initial_centers=i_centers_sentence, ccore=False)
    xmeans_sentence_instance.process()
    
    # 画像特徴量によるクラスタリングの実行
    i_centers_image = xmeans.kmeans_plusplus_initializer(embedding_image_vectors, num_clusters).initialize()
    xmeans_image_instance = xmeans.xmeans(data=embedding_image_vectors, initial_centers=i_centers_image, ccore=False)
    xmeans_image_instance.process()
    
    # クラスタリング結果を取得
    clusters_by_sentence = xmeans_sentence_instance.get_clusters()
    clusters_by_image = xmeans_image_instance.get_clusters()

    # クラスタ番号を各データに格納
    for cluster_index, cluster in enumerate(clusters_by_sentence):
        for index in cluster:
            json_data[index]['class_by_sentence_kmeans'] = cluster_index

    for cluster_index, cluster in enumerate(clusters_by_image):
        for index in cluster:
            json_data[index]['class_by_feature_kmeans'] = cluster_index

def execute_clustering_hierarchical(json_data, num_clusters):
    # キャプションベクトルによる階層型クラスタリング
    sentence_vectors = [np.array(json_datum['embedding_sentence_vector']) for json_datum in json_data]
    sentence_distance_matrix = pdist(sentence_vectors, metric='cosine')
    sentence_linkage_matrix = linkage(sentence_distance_matrix, method='average')
    sentence_clusters = fcluster(sentence_linkage_matrix, num_clusters, criterion='maxclust')
    
    # 画像ベクトルによる階層型クラスタリング
    image_vectors = [np.array(json_datum['embedding_image_vector']) for json_datum in json_data]
    image_distance_matrix = pdist(image_vectors, metric='cosine')
    image_linkage_matrix = linkage(image_distance_matrix, method='average')
    image_clusters = fcluster(image_linkage_matrix, num_clusters, criterion='maxclust')

    # クラスタリング結果を json_data に追加
    for idx, json_datum in enumerate(json_data):
        json_datum['class_by_sentence_hierarchical'] = int(sentence_clusters[idx])
        json_datum['class_by_feature_hierarchical'] = int(image_clusters[idx])

def compare_clustering_methods(json_data, num_clusters=10):
    # json_dataをディープコピーして操作用に作成
    json_data_cp = copy.deepcopy(json_data)

    # XMeansによるクラスタリング結果を追加
    execute_clustering_xmeans(json_data_cp, num_clusters)

    # 階層的クラスタリングによる結果を追加
    execute_clustering_hierarchical(json_data_cp, num_clusters)

    return json_data_cp,num_clusters