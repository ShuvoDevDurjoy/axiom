class SGD: 
    def __init__(self, parameters, lr = 0.001): 
        self.parameters = parameters
        self.lr = lr

    def step(self): 
        for p in self.parameters: 
            if p.grad is not None: 
                p.data -= self.lr * p.grad