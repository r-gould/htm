import random
import numpy as np

from typing import List, Tuple

from .synapses import ProximalSynapse, DistalSynapse

class ProximalSegment:

    def __init__(self, input_size: int, num_synapses: int, overlap_thresh: int, thresh_perm: float):
        
        self.input_size = input_size
        self.overlap_thresh = overlap_thresh
        self.synapses: List[ProximalSynapse] = self._generate_synapses(input_size, num_synapses, thresh_perm)

    def forward(self, x: np.ndarray[np.bool_]) -> Tuple[bool, int]:

        receptive_mask = self._generate_receptive_mask()
        score = int(np.sum(receptive_mask * x))
        return (score >= self.overlap_thresh), score

    def _generate_receptive_mask(self) -> np.ndarray[np.bool_]:
        
        mask = np.zeros(self.input_size, dtype=np.bool_)
        mask[[synapse.idx for synapse in self.synapses if synapse.receptive()]] = 1
        return mask

    @staticmethod
    def _generate_synapses(input_size: int, num_synapses: int, thresh_perm: float) -> List[ProximalSynapse]:

        synapse_idxs = random.sample([i for i in range(input_size)], num_synapses)
        return [ProximalSynapse(i, thresh_perm) for i in synapse_idxs]


class DistalSegment:

    def __init__(self, origin_neuron: "Neuron", max_num_synapses: int,
                 init_perm: float, thresh_perm: float):

        self.init_perm = init_perm
        self.thresh_perm = thresh_perm

        self.origin_neuron = origin_neuron
        self.max_num_synapses = max_num_synapses
        self.synapses: List[DistalSynapse] = []
        self.num_active_synapses: int = 0
        self.num_active_receptive_synapses: int = 0

    def receptive_synapses(self) -> List[DistalSynapse]:
        return [synapse for synapse in self.synapses if synapse.receptive()]

    def grow_synapses(self, to_neurons: "List[Neuron]", sample_size: int):
        
        to_neurons_copy = to_neurons.copy()
        random.shuffle(to_neurons_copy)

        count = max(0, sample_size - self.num_active_synapses)
        to_add = min(count, len(to_neurons), self.max_num_synapses - len(self.synapses))

        for i in range(to_add):
            to_neuron = to_neurons_copy[i]
            new_synapse = DistalSynapse(to_neuron, self.init_perm, self.thresh_perm)
            self.synapses.append(new_synapse)
            
    def update_active_count(self, active_neurons: "List[Neuron]"):
        self.num_active_synapses = sum([synapse.is_active(active_neurons) for synapse in self.synapses])
        self.num_active_receptive_synapses = sum([synapse.is_active(active_neurons) for synapse in self.receptive_synapses()])