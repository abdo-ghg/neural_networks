import numpy as np

class Gradient:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self):
        for p in self.parameters:
            p.grad = np.zeros_like(p.grad)

    def step(self): 
        for p in self.parameters:
            p.data -= self.lr * p.grad