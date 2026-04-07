import numpy as np

def sum_to_shape(grad, shape):
    # remove extra dimensions
    while len(grad.shape) > len(shape):
        grad = grad.sum(axis=0)

    # sum along broadcasted axes
    for i in reversed(range(len(shape))):
        if shape[i] == 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad


class Tensor:
    def __init__(self, data, _children=(), _op="", requires_grad=True):
        self.data = np.array(data, dtype=float)
        self.grad = np.zeros_like(self.data)
        self.requires_grad = requires_grad

        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op


    def __add__(self, other):

        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += sum_to_shape(out.grad, self.data.shape)
            other.grad += sum_to_shape(out.grad, other.data.shape)

        if self.requires_grad or other.requires_grad:
            out._backward = _backward

        return out

    def __mul__(self, other):

        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():

            self.grad += sum_to_shape(other.data * out.grad, self.data.shape)
            other.grad += sum_to_shape(self.data * out.grad, other.data.shape)

        if self.requires_grad or other.requires_grad:
            out._backward = _backward

        return out


    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, (self, other), "matmul")

        def _backward():
            self_grad = out.grad @ other.data.T
            other_grad = self.data.T @ out.grad

            self.grad += sum_to_shape(self_grad, self.data.shape)
            other.grad += sum_to_shape(other_grad, other.data.shape)

        if self.requires_grad or other.requires_grad:
            out._backward = _backward

        return out

    def tanh(self):
        out = Tensor(np.tanh(self.data), (self,), "tanh")

        def _backward():
            self.grad += (1 - np.tanh(self.data)**2) * out.grad

        if self.requires_grad:
            out._backward = _backward
        return out

    def sigmoid(self):
        out = Tensor(1 / (1 + np.exp(-self.data)), (self,), "sigmoid")

        def _backward():
            self.grad += out.data * (1 - out.data) * out.grad

        if self.requires_grad:
            out._backward = _backward
        return out


    def mean(self):
        out = Tensor(self.data.mean(), (self,), "mean")

        def _backward():
            self.grad += np.ones_like(self.data) * out.grad / self.data.size

        if self.requires_grad:
            out._backward = _backward
        return out


    def __neg__(self): 
        return self * -1
    def __radd__(self, other): 
        return self + other
    def __sub__(self, other): 
        return self + (-other)
    def __rsub__(self, other): 
        return other + (-self)
    def __rmul__(self, other): 
        return self * other
    def __truediv__(self, other): 
        return self * other**-1
    def __rtruediv__(self, other): 
        return other * self**-1

    def log(self):
        out = Tensor(np.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1 / self.data) * out.grad

        if self.requires_grad:
            out._backward = _backward

        return out
        
    def exp(self):
        out = Tensor(np.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        if self.requires_grad:
            out._backward = _backward

        return out

    def __pow__(self, power):
        out = Tensor(self.data ** power, (self,), f"pow{power}")

        def _backward():
            self.grad += power * (self.data ** (power - 1)) * out.grad

        if self.requires_grad:
            out._backward = _backward
        return out
    

    def backward(self):

        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)

        self.grad = np.ones_like(self.data)

        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"