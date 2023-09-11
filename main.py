import pickle
import torch
import numpy as np

from typing import List, Optional
from transformers import GPTNeoXTokenizerFast

from components import Layer, Minicolumn
from dataset import Dataset


def train(tokenizer, sdrs: np.ndarray[np.bool_], layer_params: dict, spatial_pooling_params: dict, temporal_memory_params: dict) -> Layer:
    
    layer = Layer(**layer_params)

    dataset = Dataset(tokenizer, sdrs)
    epochs = 5

    for epoch in range(epochs):
        for x in dataset:
            x: np.ndarray[np.bool_]

            active_minicolumns = layer.run_spatial_pooling(x, **spatial_pooling_params, learning_mode=True)

            layer.run_temporal_memory(active_minicolumns, **temporal_memory_params, learning_mode=True)
    
        with open(f"layer_{epoch}.pickle", "wb") as file_:
            pickle.dump(layer, file_, -1)
    
    #layer = pickle.load(open(f"layer_{epochs}.pickle", "rb", -1)
    return layer

def inference(prompt: str, layer: Layer, tokenizer, sdrs: np.ndarray[np.bool_], spatial_pooling_params: dict, 
              temporal_memory_params: dict) -> List[int]:
    # sdrs: (V, N)

    bos_token: str = tokenizer.bos_token
    prompt_: str = bos_token + prompt
    tokens: List[int] = tokenizer.encode(prompt_)

    for i, tok in enumerate(tokens):

        x: np.ndarray[np.bool_] = sdrs[tok]

        active_minicolumns = layer.run_spatial_pooling(x, **spatial_pooling_params, learning_mode=False)

        predicted: Optional[List[Minicolumn]] = layer.run_temporal_memory(active_minicolumns, **temporal_memory_params, 
                                                                          learning_mode=False, return_prediction=(i == len(tokens)-1))
    
    def minicols_to_idxs(minicols: List[Minicolumn], layer: Layer) -> List[int]:
        return [layer.minicolumns.index(minicol) for minicol in minicols]

    def overlap(a: List[int], b: List[int]) -> int:
        return len(set(a) & set(b))

    predicted_idxs = minicols_to_idxs(predicted, layer)
    overlap_scores: List[int] = []

    for i in range(tokenizer.vocab_size):
        tok_sdr: np.ndarray[np.bool_] = sdrs[i]

        active_minicolumns = layer.run_spatial_pooling(tok_sdr, **spatial_pooling_params, learning_mode=False)

        tok_idxs = minicols_to_idxs(active_minicolumns, layer)
        overlap_scores.append(
            overlap(predicted_idxs, tok_idxs)
        )

    return overlap_scores # len = vocab_size




if __name__ == "__main__":

    from bsom.codebook import Codebook

    tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/pythia-70m")

    codebook = Codebook(tokenizer.vocab_size, device="cuda")
    codebook.codes = torch.load("codes_4.pt")
    sdrs: np.ndarray[np.bool_] = codebook.get_sdrs().cpu().detach().numpy() # (V, h, w)
    sdrs = sdrs.reshape(sdrs.shape[0], -1)

    input_size = sdrs.shape[-1]

    layer_params = {
        "input_size": input_size, 
        "num_minicolumns": input_size, 
        "num_neurons_per_minicolumn": 32, 
        "num_proximal_synapses": input_size//2,
        "num_distal_segments": 0, 
        "max_distal_segments": 256, 
        "max_distal_synapses": 256, 
        "proximal_overlap_thresh": 0,
        "proximal_thresh_perm": 0.1, 
        "distal_init_perm": 0.21, 
        "distal_thresh_perm": 0.5,
    }

    spatial_pooling_params = {
        "sparsity": 0.02,
        "delta_pos": 0.05, 
        "delta_neg": 0.008, 
        "boosting": False,
    }

    temporal_memory_params = {
        "sample_size": 32,
        "delta_pos": 0.1, 
        "delta_neg": 0.1, 
        "active_thresh": 16, 
        "match_thresh": 10,
    }


    layer = train(tokenizer, sdrs, layer_params, spatial_pooling_params, temporal_memory_params)

    overlap_scores = inference("Hello, my name is", layer, tokenizer, sdrs,
                               spatial_pooling_params, temporal_memory_params)

    vocab_idx = np.argmax(overlap_scores)
    pred_tok = tokenizer.decode(vocab_idx)
    print("Pred:", pred_tok)