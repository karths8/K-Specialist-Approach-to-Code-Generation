import pandas as pd
import argparse
from transformers import AutoTokenizer
import json
import csv
import os
import random
from datasets import load_dataset
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='Options')
parser.add_argument('--input_file', default='/workspace/CS762_Project/Data_files/final_seed_data.json', type=str, help="input file")
parser.add_argument('--tokenizer_dir', default='/workspace/CS762_Project/CodeLlama-7b-Python-hf', type=str, help="tokenizer directory")
parser.add_argument('--output_file', default='generated_data', type=str, help="output directory")
parser.add_argument('--kmeans_data_path', default='/workspace/CS762_Project/Kmeans_data', type=str, help="kmeans data path")
# parser.add_argument('--total', default=5, type=int, help="total clusters")

args = parser.parse_args()

system_prompt = '''You are an assistant tasked with generating code given a question and some Examples and/or Explanations along with the question. The question as well as some examples with input and expected outputs will be between [Question] and [/Question]. You must answer the programming question with python code within the [Code] and [/Code] blocks'''

def make_chat_template(chat_type,data):
    if chat_type=='system':
        chat = [{"role": "system","content":data['input']}]
    elif chat_type=='sample':
        chat = [{"role": "user","content":data['input']},{"role": "assistant","content":data['output']}]
    return chat

def write_csv(rows,file_path):
    with open(file_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(rows)

def make_prompt_str(data):
    assert_num = min(random.randint(0,4), len(data['asserts']))
    
    assert_str = '\n'.join(random.sample(data['asserts'], assert_num))
    prompt_str=f'''
[Question]
        
{data['question']}

{assert_str}

[/Question]
'''
    code_str = f'''
\n[Code]

{data['code']} 

[/Code]
[/STOP]'''
    
    return prompt_str, code_str

def split_train_test(data, split_ratio=0.9):
    random.seed(42)
    train_split = int(len(data) * split_ratio)
    data_train = random.sample(data, train_split)
    data_test = [x for x in data if x not in data_train]
    return data_train, data_test

def get_data_list(data):
    data_list = []
    for i in data:
        prompt_str, code_str = make_prompt_str(i)
        data_list.append({'input':prompt_str,'output':code_str})
    return data_list

def get_llama_prompts(data, args, test=False):
    prompt_list = []
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    data_list = get_data_list(data)
    for i in data_list:
        prompt = make_chat_template('system',{'input':system_prompt})
        sample = make_chat_template('sample',i)
        prompt.extend(sample)
        prompt_list.append(prompt)
    llama_prompts = []
    # print(prompt_list[0])
    for idx,i in enumerate(prompt_list):
        chat_content = i[:-1] if test else i
        llama_prompts.append({'question':data[idx]['question'],'prompt':tokenizer.apply_chat_template(chat_content, tokenize=False), 'code':data[idx]['code'], 'asserts':str('\n'.join(data[idx]['asserts']))})
    return llama_prompts

def make_set_list(data_prompts):
    data_list = [['question','prompt', 'code', 'asserts']]
    for i in data_prompts:
        data_list.append([i['question'], i['prompt'], i['code'], i['asserts']])
    return data_list

def main():
    folders = os.listdir(args.kmeans_data_path)
    model_name = args.tokenizer_dir.split('/')[-1]
    base_path = f'/workspace/CS762_Project/Prepared_data/{model_name}'
    for folder in folders:
        total = int(folder[2:])
        data_path = os.path.join(args.kmeans_data_path, folder, 'clustered_data.json')
        with open(data_path, 'r') as json_file:
            clustered_data = json.load(json_file)

        for k in clustered_data:
            data_k = clustered_data[k]
            train_data, test_data = split_train_test(data_k)
            train_prompts = get_llama_prompts(train_data, args)
            test_prompts = get_llama_prompts(test_data, args, test=True)
            # print(train_prompts[0])
            train_list = make_set_list(train_prompts)
            test_list = make_set_list(test_prompts)
            # print(test_prompts[0])
            dir_path = os.path.join(base_path, f'k_{total}')
            os.makedirs(dir_path, exist_ok=True)
            train_op_path = os.path.join(dir_path,f'k_{total}_train_{k}.csv')
            val_op_path = os.path.join(dir_path,f'k_{total}_val_{k}.csv')
            write_csv(train_list,train_op_path)
            write_csv(test_list,val_op_path)
            dataset = load_dataset("csv", data_files={'train':train_op_path, 'val':val_op_path})
            print('Dataset Loaded: ')
            print(dataset)
            dataset_op_path = os.path.join(dir_path,args.output_file+f'_k_{total}_cluster_{k}')
            dataset.save_to_disk(dataset_op_path)


if __name__=='__main__':
    main()

