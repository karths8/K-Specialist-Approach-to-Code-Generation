from datasets import DatasetDict
from joblib import load
import numpy as np
import os
import json
import argparse

parser = argparse.ArgumentParser(description='Options')
parser.add_argument('--human_eval_path', default='/workspace/CS762_Project/Human_eval_prepared/phi-2/human_eval', type=str, help="kmeans output path")
args = parser.parse_args()


def make_dictionary(dataset):
    d_dict = dataset.to_dict()
    d_list = []
    for i in range(len(d_dict['task_id'])):
        d_list.append({k:d_dict[k][i] for k in d_dict})
    return d_list

def kmeans_cluster_data(kmeans_model_path,model_name, num_clusters, d_list):
    kmeans_model = load(kmeans_model_path)
    
    embeddings = np.load('/workspace/CS762_Project/Model/human_eval_embeddings.npy')
    cluster_labels = kmeans_model.predict(embeddings)
    cluster_dict = {}
    for idx,i in enumerate(cluster_labels):
        if i not in cluster_dict:
            cluster_dict[i] = []
        cluster_dict[i].append(d_list[idx])
    d_path = os.path.join(f'/workspace/CS762_Project/Model/human_eval_clustered/{model_name}/', f'clustered_data_{num_clusters}.json')
    os.makedirs(f'/workspace/CS762_Project/Model/human_eval_clustered/{model_name}/', exist_ok=True)
    print(f'Saving Data file at {d_path}!')
    cluster_dict = {str(k):v for k,v in cluster_dict.items()}
    with open(d_path, 'w') as data_file:
        json.dump(cluster_dict, data_file)

def main():
    dataset = DatasetDict.load_from_disk(args.human_eval_path)['test']
    d_list = make_dictionary(dataset)
    model_name = args.human_eval_path.split('/')[-2]
    kmeans_cluster_data('/workspace/CS762_Project/Kmeans_data/k_1/kmeans_model.joblib',model_name, 1,d_list)
    kmeans_cluster_data('/workspace/CS762_Project/Kmeans_data/k_5/kmeans_model.joblib',model_name, 5, d_list)
    kmeans_cluster_data('/workspace/CS762_Project/Kmeans_data/k_10/kmeans_model.joblib',model_name, 10, d_list)

if __name__=='__main__':
    main()