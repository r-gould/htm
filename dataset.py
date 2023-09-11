import torch
import numpy as np

from tqdm import tqdm
from datasets import load_dataset
from typing import List

class Dataset:

    def __init__(self, tokenizer, sdrs: np.ndarray[np.bool_], dataset_str: str="NeelNanda/pile-10k"):
        self.tokenizer = tokenizer
        self.dataset_str = dataset_str
        self.sdrs = sdrs

    def __iter__(self):
        
        dataset: List[str] = load_dataset(self.dataset_str)['train']['text']
        size = len(dataset)

        bos_token: str = self.tokenizer.bos_token

        idxs = np.random.permutation(size)

        for i in tqdm(idxs):
            prompt: str = bos_token + dataset[i]
            tokens: List[int] = self.tokenizer.encode(prompt)

            for tok in tokens:
                sdr: np.ndarray[np.bool_] = self.sdrs[tok] # (N)
                yield sdr