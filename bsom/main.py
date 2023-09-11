import torch
import numpy as np

from transformers import GPTNeoXTokenizerFast

from snippet_dataset import SnippetDataset
from codebook import Codebook

@torch.no_grad()
def main():

    tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/pythia-70m")
    bsize = 8
    epochs = 10
    device = "cuda"
    
    dataset = SnippetDataset(tokenizer, bsize, device)
    codebook = Codebook(tokenizer.vocab_size, device)

    codebook.run_bsom(dataset, epochs, k=4)

if __name__ == "__main__":
    main()