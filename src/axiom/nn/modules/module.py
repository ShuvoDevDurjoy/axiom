from axiom.tensor import Tensor
class Module: 
    def parameters(self): 
        return 0

    def zero_grad(): 
        raise NotImplementedError

    def __call__(self, *args, **kwars): 
        return self.forward(*args, **kwars)

    def forward(self, *args, **kwargs): 
        raise NotImplementedError

    def parameters(self): 
        return []