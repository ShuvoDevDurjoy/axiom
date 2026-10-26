import numpy as np
from axiom.nn.functional import log_softmax
from axiom.tensor import Tensor

def cross_entropy(logits, targets): 
    assert isinstance(logits, Tensor)
    assert isinstance(targets, Tensor)
    log_probs = log_softmax(logits)
    class_indices = targets.data.astype(int).reshape(-1)
    row_offsets = logits.data.shape[1] * np.arange(class_indices.size)
    corrected_log_probs = log_probs.gather(row_offsets + class_indices)
    return -corrected_log_probs.mean()