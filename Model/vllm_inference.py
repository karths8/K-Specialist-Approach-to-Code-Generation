from datasets import DatasetDict
from vllm import LLM, SamplingParams
import argparse
from joblib import load
import numpy as np
import json
import os

parser = argparse.ArgumentParser(description='Options')
parser.add_argument('--dataset_path', default='/workspace/CS762_Project/Human_eval_prepared/phi-2/human_eval', type=str, help="input file")
parser.add_argument('--model_dir', default='/workspace/CS762_Project/phi-2', type=str, help="tokenizer directory")
parser.add_argument('--num_samples', default=10, type=int, help="Num of samples to generate")
parser.add_argument('--num_clusters', default=1, type=int, help="Number of clusters")
parser.add_argument('--k_val', default=1, type=int, help="Number of clusters")
parser.add_argument('--temp', default=0.8, type=float, help="Temperature")

args = parser.parse_args()

def kmeans_cluster_data(kmeans_model_path, num_clusters):
    kmeans_model = load(kmeans_model_path)
    embeddings = np.load('/workspace/CS762_Project/Model/human_eval_embeddings.npy')
    cluster_labels = kmeans_model.predict(embeddings)
    cluster_centers = kmeans.cluster_centers_
    cluster_dict = {}
    for idx,i in enumerate(cluster_labels):
        if i not in cluster_dict:
            cluster_dict[i] = []
        cluster_dict[i].append(data[idx])
    d_path = os.path.join(kmeans_path, f'clustered_data_{num_clusters}.json')
    print(f'Saving Data file at {d_path}!')
    cluster_dict = {str(k):v for k,v in cluster_dict.items()}
    with open(d_path, 'w') as data_file:
        json.dump(cluster_dict, data_file)


def make_dictionary(dataset):
    d_dict = dataset.to_dict()
    d_list = []
    for i in range(len(d_dict['task_id'])):
        d_list.append({k:d_dict[k][i] for k in d_dict})
    return d_list

def read_json(json_path):
    with open(json_path, 'r') as fp:
        data = json.load(fp)
    return data

def main():
    dataset = DatasetDict.load_from_disk(args.dataset_path)['test']
    dataset = make_dictionary(dataset)
    merged_models_path = '/workspace/CS762_Project/Merged_models'
    model_name = args.model_dir.split('/')[-1]
    base_dir = f'/workspace/CS762_Project/Results/{model_name}/total_clusters_{args.num_clusters}/'
    with open(f'/workspace/CS762_Project/Model/human_eval_clustered/{model_name}/clustered_data_{args.num_clusters}.json', 'r') as fp:
        clustered_data = json.load(fp)
    
    for k in clustered_data:
        if int(k)==args.k_val:
            print(f'Starting for cluster_{k} out of total {args.num_clusters} clusters!')
            merged_model_path = f'/workspace/CS762_Project/Results/{model_name}/total_clusters_{args.num_clusters}/k_{k}/merged_model'
            human_eval_prompts = [p['prompt'] for p in clustered_data[k]]
            generations = []
            human_eval_ids = [t['task_id'] for t in clustered_data[k]]
            sample_temp = args.temp
            sampling_params = SamplingParams(temperature=sample_temp, top_p=0.95,max_tokens=1000, stop=['[/STOP]'])
            llm = LLM(model=merged_model_path,tokenizer=args.model_dir, trust_remote_code=True)
            for s in range(args.num_samples):
                print(f'Generating {s} Sample')
                outputs = llm.generate(human_eval_prompts, sampling_params)
                output_strs = [output.outputs[0].text for output in outputs]
                for p,q in zip(human_eval_ids, output_strs):
                    comp = q.replace('[/Code]','').strip()
                    mod_comp = '\n'.join(['    '+comp.split('\n')[0]] + comp.split('\n')[1:])
                    generations.append({'task_id':p, 'completion':mod_comp})
            gen_path = base_dir+f'temp_{str(sample_temp)}/'
            os.makedirs(gen_path, exist_ok=True)
            with open(gen_path+f'human_eval_generated_k_{k}.json', 'w') as json_file:
                json.dump(generations, json_file)
        # data_splits = []
        # for k in clustered_data:
        #     data_splits.extend(read_json(base_dir+f'human_eval_generated_k_{k}.json'))
        # with open(base_dir+f'human_eval_generated_combined.json', 'w') as json_file:
        #     json.dump(data_splits, json_file)
        
        # for output in outputs:
        #     prompt = output.prompt
        #     generated_text = output.outputs[0].text
        #     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__=='__main__':
    main()
        

