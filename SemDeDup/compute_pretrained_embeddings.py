# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from tqdm import tqdm

def get_embeddings(model, dataloader, emd_memmap):
    """
    function to compute and store representations for the data from pretrained model. It is preferable to parallelize this function on mulitiple devices (GPUs). Each device will process part of the data.
    model: pretrained model
    dataloader: should return   1) data_batch: batch of data examples
                                2) batch_indices: global index for each example (between 0 and of size <dataset_size>-1).
    emd_memmap: numpy memmap to store embeddings of size <dataset_size>.

    """

    # -- Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -- model
    model = model.to(device)
    model = model.eval()

    # -- Get and store encodings
    print("Get encoding...")
    with torch.no_grad():
        # Removing paths_batch from below
        for batch in tqdm(dataloader):
            data_batch = batch['data']
            batch_indices = batch['index']
            data_batch = data_batch.to(device)
            outputs = model(data_batch)
            encoding = outputs.logits
            eps = 1e-8
            p = 2
            dim = -1
            emd_memmap[batch_indices] = torch.norm(encoding, p=p, dim=dim, keepdim=True).clamp_min(eps).expand_as(encoding).cpu()

### Completely new code

import pandas as pd
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader

class CodeJsonDataset(Dataset):
    def __init__(self, dataset_json_file_path: str, tokenizer: AutoTokenizer):
        self.dataset_json_file_path = dataset_json_file_path
        self.code_data = self.load_json_data()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.code_data)

    def __getitem__(self, idx: int):
        sample = self.code_data[idx]
        # return {'data': self.tokenizer(sample, padding=True, truncation=True, return_tensors='pt'), 'index': idx}
        return {'data': self.tokenizer(sample, return_tensors='pt')['input_ids'], 'index': idx}

    def load_json_data(self):
        df = pd.read_json(self.dataset_json_file_path)
        merged_column = df.apply(lambda row: '\n'.join(f'{col}: {value}' for col, value in row.items()), axis=1)
        df['data'] = merged_column
        df = df['data']
        return df.values

if __name__ == '__main__':
    dataset_json_file_path = './../../codealpaca/data/code_alpaca_2k.json'
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
    model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-mono")
    code_alpaca_dataset = CodeJsonDataset(dataset_json_file_path, tokenizer)
    dataloader = DataLoader(code_alpaca_dataset, batch_size=1, shuffle=True)
    path_str_type = '<S50'
    emb_memory_loc = './code_alpaca_results/emb_mmap.dat'
    paths_memory_loc = './code_alpaca_results/paths_mmap.dat'
    dataset_size = len(code_alpaca_dataset)
    # fix this
    emb_size = 51200
    emb_mmap = np.memmap(emb_memory_loc, dtype='float32', mode='w+', shape=(dataset_size, emb_size))
    
    get_embeddings(model, dataloader, emb_mmap)