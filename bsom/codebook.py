import math
import torch
import torch.nn.functional as F
from torch import IntTensor
from torch import Tensor
from typing import Optional

from snippet_dataset import SnippetDataset

SQRT_E = math.sqrt(math.e)

class Codebook:

    def __init__(self, vocab_size: int, device: str, width: int=64, height: int=64, sigma: float=64.0, update_exponent: float=0.883286):

        self.width = width
        self.height = height
        self.num_codes = width * height
        self.vocab_size = vocab_size
        self.device = device

        self.sigma = sigma
        self.update_exponent = update_exponent

        self.coords = torch.IntTensor([[i, j] for i in range(height) for j in range(width)]).to(device).unsqueeze(0) # (1, num_codes, 2)

        self.codes = torch.rand(self.num_codes, vocab_size, device=device)
        
    @torch.no_grad()
    def run_bsom(self, dataset: SnippetDataset, epochs: int, k: Optional[int]=None):
        
        if k is not None:
            self.codes = torch.load(f"codes_{k}.pt")

        for epoch in range(epochs):
            print(f"Epoch: {epoch}")

            if (k is not None) and (epoch <= k):
                self.sigma = self.sigma**self.update_exponent
                continue

            n = torch.zeros(self.vocab_size, self.num_codes, device=self.device)
            y = torch.zeros(1, self.num_codes, device=self.device)

            for i, batch in enumerate(dataset):
                # batch: (b, V)
                dists = torch.sum((batch.unsqueeze(1) - self.codes.unsqueeze(0))**2, dim=-1) # (b, M)
                batch_closest_idxs = torch.argmin(dists, dim=-1) # (b)

                h = self.get_total_dists(batch_closest_idxs) # (b, num_codes)


                n += batch.T @ h # (V, len(codebook))

                y += torch.sum(h, dim=0, keepdim=True) # (1, len(codebook))

                del batch
                del dists
                del batch_closest_idxs
                del h
                torch.cuda.empty_cache()
            
            self.codes = (n / y).T # (len(codebook), V)

            self.sigma = self.sigma**self.update_exponent

            torch.save(self.codes, f'codes_{epoch}.pt')

        torch.save(self.codes, f'codes.pt')

        sdrs = self.get_sdrs()
        torch.save(sdrs, 'sdrs.pt')

    @torch.no_grad()
    def get_sdrs(self, sparsity: float=0.02, normalize: bool=True, 
                 resize: bool=True) -> Tensor:

        num_active = int(self.num_codes * sparsity)

        map_ = self.codes.T # (V, num_codes)
        if normalize:
            map_ = F.normalize(map_, dim=0) # (V, num_codes)
        active_idxs = torch.topk(map_, k=num_active, dim=-1).indices # (V, num_active)

        sdrs = torch.zeros(self.vocab_size, self.num_codes, dtype=torch.bool, device=self.device)
        sdrs[torch.arange(self.vocab_size).unsqueeze(1), active_idxs] = 1

        if resize:
            sdrs = sdrs.reshape(self.vocab_size, self.height, self.width)

        return sdrs

    @torch.no_grad()
    def get_total_dists(self, idxs: IntTensor, torus=True) -> Tensor:

        # idxs: (b)
        batch_coords = torch.stack([idxs // self.width, idxs % self.width], dim=1).unsqueeze(1) # (b, 1, 2)

        xy_dists = torch.abs(batch_coords - self.coords) # (b, num_codes, 2)

        if torus:
            bounds = torch.IntTensor([self.width, self.height]).to(self.device).unsqueeze(0).unsqueeze(0) # (1, 1, 2)
            xy_dists = torch.minimum(xy_dists, bounds - xy_dists) # (b, num_codes, 2)

        cb_distances = torch.max(xy_dists, dim=-1).values # (b, num_codes)

        neighbour_dists = (1 - SQRT_E * torch.exp(-0.5 * cb_distances**2 / self.sigma**2)) / (self.sigma * (1 - SQRT_E))
        neighbour_dists[cb_distances > self.sigma] = 0.0
        return neighbour_dists # (b, num_codes)