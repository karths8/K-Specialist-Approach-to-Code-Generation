import numpy as np
from sklearn.cluster import KMeans
import joblib
import json
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer, util
import argparse

parser = argparse.ArgumentParser(description='Options')
parser.add_argument('--kmeans_base_path', default='/workspace/CS762_Project/Kmeans_data', type=str, help="kmeans output path")
parser.add_argument('--data_path', default='/workspace/CS762_Project/Data_files/final_combined_data_Dec15_mod_asserts.json', type=str, help="path for input data")
parser.add_argument('--emb_path', default='/workspace/CS762_Project/Notebooks/final_combined_data_Dec15_embeddings.npy', type=str, help="path for precomputed embeddings based on data_path")
parser.add_argument('--num_clusters', default=1, type=int, help="Number of clusters for kmeans")
parser.add_argument('--model_name', default='/workspace/CS762_Project/e5-large-v2', type=str, help="Name of the sentence t model")
args = parser.parse_args()


def load_save_embeddings(data_path,save_path):
    model = SentenceTransformer(args.model_name)
    with open(data_path, 'r') as json_file:
        data = json.load(json_file)
    questions = [i['question'] for i in data]
    print('Embeddings starting!')
    embeddings = model.encode(questions)
    print('Embeddings complete!')
    np.save(save_path, embeddings)

def kmeans_cluster_data(data_path, emb_path, num_clusters):
    with open(data_path, 'r') as json_file:
        data = json.load(json_file)
    print('Loading embeddings!')
    embeddings = np.load(emb_path)
    labels = ['' for i in embeddings]
    print(f'Fitting KMeans model with k={num_clusters}!')
    kmeans_path = os.path.join(args.kmeans_base_path, f'k_{num_clusters}')
    os.makedirs(kmeans_path, exist_ok=True)
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans_model = kmeans.fit(embeddings)
    cluster_labels = kmeans_model.predict(embeddings)
    cluster_centers = kmeans.cluster_centers_
    print(f'Saving KMeans model with k={num_clusters}!')
    model_filename = os.path.join(kmeans_path, 'kmeans_model.joblib')
    joblib.dump(kmeans_model, model_filename)
    
    cluster_dict = {}
    for idx,i in enumerate(cluster_labels):
        if i not in cluster_dict:
            cluster_dict[i] = []
        cluster_dict[i].append(data[idx])
    d_path = os.path.join(kmeans_path, 'clustered_data.json')
    print(f'Saving Data file at {d_path}!')
    cluster_dict = {str(k):v for k,v in cluster_dict.items()}
    # print(cluster_dict)
    with open(d_path, 'w') as data_file:
        json.dump(cluster_dict, data_file)

def main():
    if not os.path.exists(args.emb_path):
        load_save_embeddings(args.data_path,args.emb_path)
    kmeans_cluster_data(args.data_path,args.emb_path, args.num_clusters)

if __name__=='__main__':
    main()
