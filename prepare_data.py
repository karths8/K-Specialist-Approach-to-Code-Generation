import pandas as pd
import argparse
from transformers import AutoTokenizer
import json
import csv
from datasets import load_dataset

parser = argparse.ArgumentParser(description='Options')
parser.add_argument('--input_file', default='Seed data - Karthik.csv', type=str, help="input file")
parser.add_argument('--tokenizer_dir', default='/workspace/CS762_Project/CodeLlama-7b-Python-hf', type=str, help="tokenizer directory")
parser.add_argument('--output_file', default='generated_data', type=str, help="output directory")

args = parser.parse_args()

system_prompt = '''You are an assistant tasked with generating code given a question and some Examples / Explanations along with the question. The question will be given under the heading "Question:" and the examples or explanations will be given under "Example / Explanation:". Your job is to generate the code and complete the content under the heading title "Code:". '''

def make_chat_template(chat_type,data):
    if chat_type=='system':
        chat = [{"role": "system","content":data['input']}]
    elif chat_type=='sample':
        chat = [{"role": "user","content":data['input']},{"role": "assistant","content":data['output']}]
    return chat

def write_csv(rows,file_path):
    with open(file_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows([['question','prompt', 'category']] + rows)

def make_prompt_str(data):
    prompt_str=f'''Question:
        
{data['Question']}

Example / Explanation:

{data['Question Example/Explanation']}'''
    code_str = f'''\nCode:

{data['Method 1']} [/STOP]'''
    
    return prompt_str, code_str

def split_train_test(df):
    categories_dict = df['Categories'].to_dict()
    bin_dict = {}
    for i in categories_dict:
        key_str = categories_dict[i].replace('[','').replace(']','')
        if key_str in bin_dict.keys():
            bin_dict[key_str].append(i)
        else:
            bin_dict[key_str] = [i]
    test_idxs = [bin_dict[i][len(bin_dict[i])//2] for i in bin_dict]
    train_idxs = [j for i in bin_dict for j in bin_dict[i] if j not in test_idxs]
    df_test = df.loc[test_idxs]
    df_train = df.loc[train_idxs]
    return df_train, df_test

def get_data_list(seed_samples):
    data_list = []
    for i in seed_samples:
        prompt_str, code_str = make_prompt_str(i)
        data_list.append({'input':prompt_str,'output':code_str})
    return data_list

def get_llama_prompts(df, args, test=False):
    prompt_list = []
    seed_samples = []
    for index, row in df.iterrows():
        seed_samples.append(row.to_dict())
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    data_list = get_data_list(seed_samples)
    for i in data_list:
        prompt = make_chat_template('system',{'input':system_prompt})
        sample = make_chat_template('sample',i)
        prompt.extend(sample)
        prompt_list.append(prompt)
    llama_prompts = []
    # print(prompt_list[0])
    for idx,i in enumerate(prompt_list):
        chat_content = i[:-1] if test else i
        # print(chat_content)
        # print(tokenizer.apply_chat_template(chat_content, tokenize=False))
        # print(idx)
        # print(seed_samples[idx])
        llama_prompts.append({'question':seed_samples[idx]['Question'],'prompt':tokenizer.apply_chat_template(chat_content, tokenize=False),'category':seed_samples[idx]['Categories'].replace('[','').replace(']','')})
    return llama_prompts

def make_set_list(data_prompts):
    data_list = [['question','prompt', 'category']]
    for i in data_prompts:
        data_list.append([i['question'], i['prompt'], i['category']])
    return data_list

def main():
    df = pd.read_csv(args.input_file)
    df_train, df_test = split_train_test(df)
    train_prompts = get_llama_prompts(df_train, args)
    test_prompts = get_llama_prompts(df_test, args, test=True)
    print(train_prompts[0])
    train_list = make_set_list(train_prompts)
    test_list = make_set_list(test_prompts)
    print(test_prompts[0])
    write_csv(train_list,'train.csv')
    write_csv(test_list,'test.csv')
    dataset = load_dataset("csv", data_files={'train':'train.csv', 'test':'test.csv'})
    print('Dataset Loaded: ')
    print(dataset)
    dataset.save_to_disk(args.output_file)
    

if __name__=='__main__':
    main()
    