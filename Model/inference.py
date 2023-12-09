from datasets import Dataset
import torch
from peft import PeftModel
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
import csv
import numpy as np
import pickle
import time
import pandas as pd
from transformers import StoppingCriteriaList, StoppingCriteria
import json
import re

# parser = argparse.ArgumentParser(description='Options')
# parser.add_argument('--data_path', default='/workspace/llama_data', type=str, help="Where the test data is stored")
# parser.add_argument('--model_dir', default='/workspace/Llama-2-13b-hf', type=str, help="Folder where model is stored")
# parser.add_argument('--lora_path', default='/workspace/results/checkpoint-489', type=str, help="Where the adapter weights are stored")
# parser.add_argument('--quant', action='store_true', help="whether to load 8-bit model instead of 16-bit")
# args = parser.parse_args()

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = [], encounters=1,tokenizer=None):
        super().__init__()
        self.stops = stops
        self.ENCOUNTERS = encounters
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        input_list = input_ids[0].tolist()
        sentence = self.tokenizer.decode(input_list,return_tensors="pt")
        stopping_word = self.stops[0]
        stop_count = sentence.count(stopping_word)
        if stop_count >= self.ENCOUNTERS:
            return True
        return False

def inference(data_path='./../Data_files/human_eval.json', model_dir = './../CodeLlama-7b-Python-hf', lora_path = 'results/checkpoint-120/', res_path = './../Data_files/human_eval_completions.json'):
    human_eval_data = {}
    with open(data_path, 'r') as file:
        human_eval_data = json.load(file)
    checkpoint = model_dir
    device = "cuda"
    print('Loading model for inference')
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.float16, device_map='auto')
    print('Loading Lora Weights')
    # model = PeftModel.from_pretrained(model, lora_path)
    # model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    print('started')
    stop_words_ids = ["[/STOP]"]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids, encounters=1,tokenizer=tokenizer)])
    start_time = time.time()
    result = []
    for data in human_eval_data:
        print(data['task_id'])
        input_str1 = tokenizer(data['prompt'])
        print('generate started!')
        predict = model.generate(inputs=torch.tensor([input_str1['input_ids']]).to('cuda'), max_new_tokens=1200, stopping_criteria = stopping_criteria)
        print('generate ended')
        # predict = model.generate(torch.tensor([input_str1]).to('cuda'), max_length=1024)
        predict_str = tokenizer.decode(predict[0],skip_special_tokens=True)
        print('Prediction String:\n')
        print(predict_str)
        print('Answer:\n')
        print(data['canonical_solution'])
        result.append({
            'task_id' : data['task_id'],
            'completion' : predict_str
        })
    with open(res_path, 'w') as res_file:
        json.dump(result, res_file, indent = 4)
    

inference()