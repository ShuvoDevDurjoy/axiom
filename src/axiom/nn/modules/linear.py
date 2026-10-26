from axiom.nn.modules.module import Module
from axiom.init import kaiming_uniform, uniform, _get_fan_in_and_fan_out
import numpy as np
from axiom.tensor import Tensor

class Linear(Module): 
    def __init__(self, in_features, out_features, bias = False):
        super().__init__()
        self.shape = (out_features, in_features)
        self.weights = Tensor(kaiming_uniform(self.shape, np.sqrt(5.0)), requires_grad=True)
        self.bias = None
        if bias: 
            fan_in, _ = _get_fan_in_and_fan_out(self.shape)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            self.bias = Tensor(uniform((out_features, ), -bound, bound), requires_grad=True)

    def parameters(self): 
        if self.bias: 
            return [self.weights, self.bias]
        return [self.weights]

    def forward(self, x): 
        out = x @ self.weights.T
        if self.bias: 
            out = out + self.bias
        return out




