import random
import numpy as np

from typing import List, Optional

from .segments import ProximalSegment, DistalSegment

class Neuron:

    def __init__(self, num_distal_segments: int, max_distal_segments: int, max_distal_synapses: int, 
                 distal_init_perm: float, distal_thresh_perm: float):
        # parameters are associated with distal connections
        assert(num_distal_segments <= max_distal_segments)
    
        self.num_distal_segments = num_distal_segments
        self.max_distal_segments = max_distal_segments
        self.distal_init_perm = distal_init_perm
        self.distal_thresh_perm = distal_thresh_perm
        self.distal_segments = [DistalSegment(self, max_distal_synapses, distal_init_perm, distal_thresh_perm)
                                for _ in range(num_distal_segments)]

    def grow_segment(self) -> Optional[DistalSegment]:
        if len(self.distal_segments) >= self.max_distal_segments:
            return None
        new_segment = DistalSegment(self, self.max_distal_segments, self.distal_init_perm, self.distal_thresh_perm)
        self.distal_segments.append(new_segment)
        return new_segment

class Minicolumn:

    def __init__(self, input_size: int, num_neurons_per_minicolumn: int, num_proximal_synapses: int,
                 num_distal_segments: int, max_distal_segments: int, max_distal_synapses: int, proximal_overlap_thresh: int, 
                 proximal_thresh_perm: float, distal_init_perm: float, distal_thresh_perm: float):

        self.neurons = [Neuron(num_distal_segments, max_distal_segments, max_distal_synapses, 
                               distal_init_perm, distal_thresh_perm) 
                        for _ in range(num_neurons_per_minicolumn)]

        self.proximal_segment = ProximalSegment(input_size, num_proximal_synapses, proximal_overlap_thresh, proximal_thresh_perm)
        
        self.distal_segments: List[DistalSegment] = []
        self.update_distal_info()

    def update_distal_info(self):

        self.distal_segments = [segment for neuron in self.neurons for segment in neuron.distal_segments]


class Layer:

    def __init__(self, input_size: int, num_minicolumns: int, num_neurons_per_minicolumn: int, num_proximal_synapses: int,
                 num_distal_segments: int, max_distal_segments: int, max_distal_synapses: int, proximal_overlap_thresh: int, 
                 proximal_thresh_perm: float, distal_init_perm: float, distal_thresh_perm: float):

        self.minicolumns = [Minicolumn(input_size, num_neurons_per_minicolumn, num_proximal_synapses, num_distal_segments, 
                                       max_distal_segments, max_distal_synapses, proximal_overlap_thresh, 
                                       proximal_thresh_perm, distal_init_perm, distal_thresh_perm)
                            for _ in range(num_minicolumns)]

        self.active_neurons = []
        self.winner_neurons = []
        self.active_segments = []
        self.matching_segments = []

    def run_spatial_pooling(self, x: np.ndarray[np.bool_], sparsity: float, learning_mode: bool,
                            delta_pos: float, delta_neg: float, boosting: bool) -> List[Minicolumn]:
        
        overlapped = []
        for minicolumn in self.minicolumns:
            is_active, score = minicolumn.proximal_segment.forward(x)
            if is_active:
                overlapped.append((minicolumn, score))

        overlapped.sort(key=lambda pair: pair[1], reverse=True)
        take_top = int(sparsity * len(self.minicolumns))
        active_minicolumns = [pair[0] for pair in overlapped[:take_top]]

        if learning_mode:
            for minicolumn in active_minicolumns:
                for synapse in minicolumn.proximal_segment.synapses:
                    synapse.hebbian_update(delta_pos, delta_neg, x)

                    if boosting:
                        raise NotImplementedError()
                        # increase boost coeff for cols that have low active duty   
                        # across all columns

        return active_minicolumns


    def run_temporal_memory(self, active_minicolumns: List[Minicolumn], sample_size: int, learning_mode: bool, 
                            delta_pos: float, delta_neg: float, active_thresh: int, match_thresh: int,
                            return_prediction: bool=False) -> Optional[List[Minicolumn]]:
        
        new_active_neurons = []
        new_winner_neurons = []
        new_active_segments = []
        new_matching_segments = []

        for minicolumn in self.minicolumns:
            if minicolumn in active_minicolumns:
                # minicolumn active
                active_distal_segments = set(minicolumn.distal_segments) & set(self.active_segments)
                if len(active_distal_segments) > 0:
                    # was correctly in predictive mode

                    for segment in active_distal_segments:
                        # activate neurons in predictive state
                        new_active_neurons.append(segment.origin_neuron)
                        new_winner_neurons.append(segment.origin_neuron)

                        # segments active => strengthen
                        # segments inactive => weaken
                        # grow synapses to previous winner neurons to strengthen pattern matching
                        if learning_mode:
                            for synapse in segment.receptive_synapses():
                                synapse.hebbian_update(delta_pos, delta_neg, self.active_neurons)

                            segment.grow_synapses(self.winner_neurons, sample_size)
                
                else:
                    # was not in predictive mode => bursting

                    # activate all neurons
                    for neuron in minicolumn.neurons:
                        new_active_neurons.append(neuron)

                    matching_distal_segments = set(minicolumn.distal_segments) & set(self.matching_segments)
                    if len(matching_distal_segments) > 0:
                        # if at least one matching segment,
                        # set the learning segment to the maximally matching segment
                        learning_segment = max(matching_distal_segments, key=lambda segment: (segment.num_active_synapses, random.random()))
                        winner_neuron = learning_segment.origin_neuron

                    else:
                        # if no matching segments,
                        # set the learning segment to a newly grown segment on the neuron with the fewest segments
                        winner_neuron = min(minicolumn.neurons, key=lambda neuron: (len(neuron.distal_segments), random.random()))
                        learning_segment = winner_neuron.grow_segment()

                    new_winner_neurons.append(winner_neuron)

                    if learning_mode:
                        for synapse in learning_segment.receptive_synapses():
                            synapse.hebbian_update(delta_pos, delta_neg, self.active_neurons)

                        learning_segment.grow_synapses(self.winner_neurons, sample_size)


            else:
                # minicolumn inactive
                if not learning_mode:
                    continue

                # this punishes (since minicol did not actually 
                # activate) segments that matched (which includes active segments)
                
                matching_distal_segments = set(minicolumn.distal_segments) & set(self.matching_segments)
                
                for segment in matching_distal_segments:
                    for synapse in segment.synapses:
                        if synapse.is_active(self.active_neurons):
                            synapse.perm = max([synapse.perm - delta_neg, 0.0])

        # update segment info using new info
        for minicolumn in self.minicolumns:
            minicolumn.update_distal_info()
            for segment in minicolumn.distal_segments:
                segment.update_active_count(new_active_neurons)

                # match_thresh < active_thresh, so since
                # segment.num_active_synapses > segment.num_active_receptive_synapses
                # then if a segment is active, it also matches.
                
                if segment.num_active_synapses >= match_thresh:
                    new_matching_segments.append(segment)

                if segment.num_active_receptive_synapses >= active_thresh:
                    new_active_segments.append(segment)

        
        # save new info
        self.active_neurons = new_active_neurons
        self.winner_neurons = new_winner_neurons
        self.active_segments = new_active_segments
        self.matching_segments = new_matching_segments

        if return_prediction:
            predicted_minicolumns: List[Minicolumn] = []
            for minicolumn in self.minicolumns:
                active_distal_segments = set(minicolumn.distal_segments) & set(self.active_segments)
                if len(active_distal_segments) > 0:
                    # minicolumn has been predicted to fire next
                    predicted_minicolumns.append(minicolumn)
            return predicted_minicolumns