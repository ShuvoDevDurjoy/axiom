
import numpy as np
from nn.tensor import Tensor
from nn.modules.module import Module
import nn.functional as F
from nn.init import _get_fan_in_and_fan_out, kaiming_uniform, uniform


class Upsample(Module): 
    def __init__(self, scale_factor: int = 1, mode = 'nearest'): 
        self.scale_factor = scale_factor
        self.mode = mode

    def _nearest_interpolate(self, x): 
        pass

    def _uniform_interpolate(self, x): 
        pass

    def parameters(self): 
        return []

    def forward(self, x): 
        return x