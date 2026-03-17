import numpy as np
from micrograd.engine import Tensor


class Module:
    def __init__(self):
        self.requires_grad = True

    def parameters(self):

        params = []

        for attr in self.__dict__.values():

            if isinstance(attr, Tensor):
                params.append(attr)

            elif isinstance(attr, Module):
                params += attr.parameters()

            elif isinstance(attr, list):
                for item in attr:
                    if isinstance(item, Module):
                        params += item.parameters()

        return params

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.grad)

    def train(self):
        self.requires_grad = True

    def eval(self):
        self.requires_grad = False

    def check_requires_grad(self):
        for p in self.parameters():
            p.requires_grad = self.requires_grad

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        self.W = Tensor(np.random.randn(in_features, out_features) * np.sqrt(2/in_features), requires_grad=self.requires_grad)

        if bias:
            self.b = Tensor(np.random.randn(1, out_features) * 0.01, requires_grad=self.requires_grad)

    def forward(self, x):
        return x @ self.W + (self.b if hasattr(self, 'b') else 0)


class ReLU(Module):
    def forward(self, x):
        return x.relu()

class Tanh(Module):
    def forward(self, x):
        return x.tanh()

class Sigmoid(Module):
    def forward(self, x):
        return x.sigmoid()