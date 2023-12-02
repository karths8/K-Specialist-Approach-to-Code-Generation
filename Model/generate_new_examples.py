from datasets import Dataset, DatasetDict
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

# class StoppingCriteriaSub(StoppingCriteria):

#     def __init__(self, stops = [], encounters=1,tokenizer=None):
#         super().__init__()
#         self.stops = stops
#         self.ENCOUNTERS = encounters
#         self.tokenizer = tokenizer

#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
#         input_list = input_ids[0].tolist()
#         sentence = self.tokenizer.decode(input_list,return_tensors="pt")
#         stopping_word = self.stops[0]
#         stop_count = sentence.count(stopping_word)
#         if stop_count >= self.ENCOUNTERS:
#             return True
#         return False

def inference(data_path='/workspace/CS762_Project/Data_files/prompt_list.json', model_dir = '/workspace/CS762_Project/CodeLlama-34b-Python-hf'):
    # dataset = DatasetDict.load_from_disk(data_path)
    with open(data_path, 'r') as json_file:
        data = json.load(json_file)
    checkpoint = model_dir
    device = "cuda"
    print('Loading model for inference')
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.float16, device_map='auto')
    # print('Loading Lora Weights')
    # model = PeftModel.from_pretrained(model, lora_path)
    # model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    print('started')
    # stop_words_ids = ["[/STOP]"]
    # stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids, encounters=1,tokenizer=tokenizer)])
    start_time = time.time()
    predictions = []
    for idx,i in enumerate(data):
        print(idx)
        input_str1 = tokenizer(i['prompt'])
        print('generate started!')
        predict = model.generate(inputs=torch.tensor([input_str1['input_ids']]).to('cuda'), max_new_tokens=500)
        print('generate ended')
        # predict = model.generate(torch.tensor([input_str1]).to('cuda'), max_length=1024)
        predict_str = tokenizer.decode(predict[0],skip_special_tokens=True)
        predictions.append({'prompt':i['prompt'],'examples':i['examples'],'prediction':predict_str})
        print('Prediction String:\n')
        print(predict_str)
        # print('Answer:\n')
        # print(i['answer'])
    end_time = time.time()
    elapsed_time = start_time - end_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")
inference()