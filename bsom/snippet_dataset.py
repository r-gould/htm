import torch
import numpy as np
from typing import List

from datasets import load_dataset

class SnippetDataset:

    def __init__(self, tokenizer, bsize: int, device: str, dataset_str: str="NeelNanda/pile-10k"):
        self.tokenizer = tokenizer
        self.bsize = bsize
        self.dataset_str = dataset_str
        self.device = device

    def __iter__(self):
        
        dataset: List[str] = load_dataset(self.dataset_str)['train']['text']
        size = len(dataset)

        vocab_size: int = self.tokenizer.vocab_size
        bos_token: str = self.tokenizer.bos_token
        bullet_point_id: int = self.tokenizer.encode(".")[0]

        idxs = np.random.permutation(size)
        curr_snippets: List[List[int]] = []
        for i in idxs:
            prompt: str = bos_token + dataset[i]
            tokens: List[int] = self.tokenizer.encode(prompt)
            snippets: List[List[int]] = self.to_snippets(tokens, bullet_point_id)

            curr_snippets.extend(snippets)

            while len(curr_snippets) >= self.bsize:
                batch_snippets = curr_snippets[:self.bsize]
                curr_snippets = curr_snippets[self.bsize:]

                batch = torch.FloatTensor([[1 if j in batch_snippets[i] else 0 for j in range(vocab_size)] 
                                          for i in range(self.bsize)]) # (b, vocab_size)

                yield batch.to(self.device)

    @staticmethod
    def to_snippets(tokens: List[int], bullet_point_id: int, min_size: int = 48, 
                    max_size: int = 96) -> List[List[int]]:
        snippets: List[List[int]] = []
        last_idx = 0
        curr_count = 0
        for i in range(len(tokens)):
            if ((curr_count >= min_size) and (tokens[i] == bullet_point_id)) or (curr_count == max_size):
                snippets.append(tokens[last_idx:i+1])
                last_idx = i+1
                curr_count = 0
            else:
                curr_count += 1
                
        return snippets