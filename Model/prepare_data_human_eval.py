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
parser.add_argument('--input_file', default='/workspace/CS762_Project/Data_files/human_eval.json', type=str, help="input file")
parser.add_argument('--tokenizer_dir', default='/workspace/CS762_Project/CodeLlama-7b-Python-hf', type=str, help="tokenizer directory")

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
    prompt_str=f'''
[Question]
        
{data['question']}

[/Question]
'''
    code_str = f'''
\n\n[Code]

{data['function_signature']} 
'''
    
    return prompt_str, code_str

def clean_prompt(p):
    new_p = '\n'.join(p.strip().split('\n')[:-1])
    return new_p.strip()

def get_data_list(data):
    data_list = []
    for i in data:
        prompt_str, code_str = make_prompt_str(i)
        data_list.append({'input':prompt_str,'output':code_str})
    return data_list

def get_llama_prompts(data, args):
    prompt_list = []
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    data_list = get_data_list(data)
    for i in data_list:
        prompt = make_chat_template('system',{'input':system_prompt})
        sample = make_chat_template('sample',i)
        prompt.extend(sample)
        prompt_list.append(prompt)
    llama_prompts = []
    for idx,i in enumerate(prompt_list):
        model_prompt = clean_prompt(tokenizer.apply_chat_template(i, tokenize=False))
        llama_prompts.append({'task_id':data[idx]['task_id'], 'question':data[idx]['question'],'prompt':model_prompt})
    return llama_prompts

def make_set_list(data_prompts):
    data_list = [['task_id','question', 'prompt']]
    for i in data_prompts:
        data_list.append([i['task_id'], i['question'], i['prompt']])
    return data_list

def main():
    model_name = args.tokenizer_dir.split('/')[-1]
    with open(args.input_file, 'r') as fp:
        data = json.load(fp)
    base_path = f'/workspace/CS762_Project/Human_eval_prepared/{model_name}'
    prompts = get_llama_prompts(data, args)
    prompts_list = make_set_list(prompts)
    os.makedirs(base_path, exist_ok=True)
    csv_path = os.path.join(base_path,'human_eval.csv')
    write_csv(prompts_list,csv_path)
    dataset = load_dataset("csv", data_files={'test':csv_path})
    print('Dataset Loaded: ')
    print(dataset)
    op_path = os.path.join(base_path, 'human_eval')
    dataset.save_to_disk(op_path)


if __name__=='__main__':
    main()

