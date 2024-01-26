import torch
import pandas as pd
import numpy as np

from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader

def get_embeddings(model, dataloader, emd_memmap, paths_memmap):
    """
    Function to compute and store representations for the data from pre-trained sentence transformers. 
    It is preferable to parallelize this function on multiple devices (GPUs). Each device will process part of the data.
    
    model: pre-trained sentence transformer
    dataloader: should return   1) data_batch: batch of data examples
                                2) batch_indices: global index for each example (between 0 and of size <dataset_size>-1)
    emd_memmap: numpy memmap to store embeddings of size <dataset_size>.
    """

    # -- Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -- model
    model = model.to(device)
    model = model.eval()

    # -- Get and store embeddings
    print("Get emedding...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            data_batch = batch['data']
            batch_indices = batch['index']
            paths_batch = batch['path']
            embeddings = model.encode(data_batch)
            emd_memmap[batch_indices] = embeddings
            paths_memmap[batch_indices] = paths_batch

class CodeJsonDataset(Dataset):
    def __init__(self, dataset_json_file_path: str):
        self.dataset_json_file_path = dataset_json_file_path
        self.code_data = self.load_json_data()
        
    def __len__(self):
        return len(self.code_data)

    def __getitem__(self, idx: int):
        sample = self.code_data['instruction'][idx]
        path = self.code_data['ID'][idx]
        return {'data': sample, 'index': idx, 'path': path}

    def load_json_data(self):
        df = pd.read_json(self.dataset_json_file_path)
        df['ID'] = range(0, len(df))
        # remove the .head(200) post validating
        return df.head(200)

if __name__ == '__main__':
    # dataset to be deduped
    dataset_json_file_path = './../../codealpaca/data/code_alpaca_2k.json'
    
    # plug in your favorite encoder: using https://huggingface.co/BAAI/bge-large-en-v1.5
    model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    
    code_dataset = CodeJsonDataset(dataset_json_file_path)
    # plug in optimal batch size
    dataloader = DataLoader(code_dataset, batch_size=64, shuffle=False)

    # params for storing embeddings
    emb_memory_loc = './code_alpaca_results/emb_mmap.dat'
    paths_memory_loc = './code_alpaca_results/paths_mmap.dat'
    dataset_size = len(code_dataset)
    
    # model embedding size
    emb_size = 1024
    emb_mmap = np.memmap(emb_memory_loc, dtype='float32', mode='w+', shape=(dataset_size, emb_size))
    path_mmap = np.memmap(paths_memory_loc, dtype='float32', mode='w+', shape=(dataset_size,))
    
    get_embeddings(model, dataloader, emb_mmap, path_mmap)