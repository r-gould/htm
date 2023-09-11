import random
import numpy as np

from typing import List

class Synapse:

    def __init__(self, thresh_perm: float):
        
        self.thresh_perm = thresh_perm
        self.perm: float = self._create_perm() #Normal(init_perm, sigma)

    def receptive(self) -> bool:
        return (self.perm >= self.thresh_perm)
    
    def hebbian_update(self, delta_pos: float, delta_neg: float, *args, **kwargs):
        
        if self.is_active(*args, **kwargs):
            self.perm = min(self.perm + delta_pos, 1.0)
        else:
            self.perm = max(self.perm - delta_neg, 0.0)

    def is_active(self, *args, **kwargs) -> bool:
        raise NotImplementedError()

    def _create_perm(self) -> float:
        raise NotImplementedError()


class ProximalSynapse(Synapse):

    def __init__(self, input_idx: int, thresh_perm: float):
        
        super().__init__(thresh_perm)
        self.idx = input_idx
    
    def hebbian_update(self, delta_pos: float, delta_neg: float, x: np.ndarray[np.bool_]):
        super().hebbian_update(delta_pos, delta_neg, x)

    def is_active(self, x: np.ndarray[np.bool_]) -> bool:
        return x[self.idx]

    def _create_perm(self) -> float:
        
        if random.random() > 0.5:
            # make receptive
            return self.thresh_perm + (1.0 - self.thresh_perm) * random.random()
        
        # make non-receptive
        return self.thresh_perm * random.random()



class DistalSynapse(Synapse):

    def __init__(self, pre_neuron: "Neuron", init_perm: float, thresh_perm: float):
        
        self.init_perm = init_perm
        super().__init__(thresh_perm)
        self.pre_neuron = pre_neuron

    def hebbian_update(self, delta_pos: float, delta_neg: float, active_neurons: "List[Neuron]"):
        super().hebbian_update(delta_pos, delta_neg, active_neurons)
    
    def is_active(self, active_neurons: "List[Neuron]") -> bool:
        return (self.pre_neuron in active_neurons)

    def _create_perm(self) -> float:
        return self.init_perm
