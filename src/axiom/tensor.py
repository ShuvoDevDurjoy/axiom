import numpy as np
import cupy as cp
from . import *

class Tensor:
    float32 = float32
    float64 = float64
    float16 = float16
    int32 = int32
    int64 = int64
    bool_ = bool_
    def __init__(self, data: np.ndarray, _childs: tuple = (), requires_grad = False, _op = None, device = DEVICE, dtype = None): 
        self.device = DEVICE if device not in ['cpu', 'cuda'] else device
        self.xp = cp if self.device == 'cuda' else np
        self.dtype = dtype
        self._init_data(data)
        self._backward = lambda: None
        self.grad = self.xp.zeros_like(self.data) if requires_grad else None
        self._prevs = set(_childs)
        self.requires_grad = requires_grad
        self._op = _op

    def _init_data(self, data):
        if self.dtype is None: 
            self.dtype = getattr(data, 'dtype', self.float32)
        if isinstance(data, self.xp.ndarray):
            self.data = data.astype(self.dtype)
        else: 
            self.data = self.xp.asarray(data, dtype = self.dtype) 

    @property
    def T(self): 
        return self.transpose()
    
    def transpose(self, *axes):
        if not axes:
            axes = tuple(reversed(range(self.data.ndim)))
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        else:
            axes = tuple(axes)
        out = Tensor(self.xp.transpose(self.data, axes), (self,), self.requires_grad, _op='transpose', device = self.device)
        def _backward(): 
            if self.requires_grad: 
                inverse_axes = self.xp.argsort(self.xp.asarray(axes))
                if hasattr(inverse_axes, 'tolist'):
                    inverse_axes = tuple(inverse_axes.tolist())
                else:
                    inverse_axes = tuple(inverse_axes)
                self._accumulate_grad(self.xp.transpose(out.grad, inverse_axes))
        out._backward = _backward
        return out

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = Tensor(self.data.reshape(*shape), (self,), self.requires_grad, _op='reshape', device = self.device)
        def _backward():
            # self.grad += out.grad.reshape(self.shape)
            self._accumulate_grad(out.grad.reshape(self.shape))
        out._backward = _backward
        return out

    def gather(self, indices):
        indices = self.xp.asarray(indices)
        if not self.xp.issubdtype(indices.dtype, self.xp.integer):
            raise TypeError("gather indices must be integers")
        indices = indices.astype(self.xp.intp, copy=False)
        if self.xp.any(indices < 0) or self.xp.any(indices >= self.data.size):
            indices_zero = [i for i in indices if i < 0]
            indices_more = [i for i in indices if i >= self.data.size]
            print(indices_zero)
            print(indices_more)
            raise IndexError("gather index is outside the flattened Tensor")

        out = Tensor(self.data.reshape(-1)[indices], (self,), self.requires_grad, _op='gather', device = self.device)

        def _backward():
            self.xp.add.at(self.grad.reshape(-1), indices.reshape(-1), out.grad.reshape(-1))

        out._backward = _backward
        return out

    def broadcast_to(self, shape: tuple): 
        if len(shape) < self.ndim: 
            raise ValueError(f"Can not broadcast to fewer dimension from {self.ndim} to {len(shape)}")

        for source, target in zip((1,) * (len(shape) - self.ndim) + self.shape, shape): 
            if source != target and source != 1: 
                raise ValueError(f"Can not Broadcast shape: from {self.shape} to {shape}")

        out = Tensor(self.xp.broadcast_to(self.data, shape), (self, ), requires_grad= self.requires_grad, _op ='broadcast_to', device = self.device)
        def _backward(): 
            self._accumulate_grad(out.grad)
        out._backward = _backward
        return out


    def __getitem__(self, key):
        out = Tensor(self.data[key], (self, ), requires_grad= self.requires_grad, _op=f'getitem: {key}', device = self.device)
        def _backward(): 
            self.xp.add.at(self.grad, key, out.grad)
        out._backward = _backward
        return out

    def requires_grad_(self, requires_grad: bool):
        self.requires_grad = requires_grad
        self.grad = self.xp.zeros_like(self.data) if self.requires_grad else None

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=self.requires_grad, device = self.device)
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data + other.data, (self, other), requires_grad=requires_grad, _op='+', device = self.device)
        def _backward(): 
            self._accumulate_grad(out.grad)
            other._accumulate_grad(out.grad)
        out._backward = _backward
        return out

    def __mul__(self, other): 
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=self.requires_grad, device = self.device)
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data * other.data, (self, other), requires_grad=requires_grad, _op='*', device = self.device)
        def _backward(): 
            self._accumulate_grad(other.data * out.grad)
            other._accumulate_grad(self.data * out.grad)
        out._backward = _backward
        return out

    def __matmul__(self, other): 
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=self.requires_grad, device = self.device)
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data @ other.data, (self, other), requires_grad=requires_grad, _op='@', device = self.device)
        def _backward(): 
            self._accumulate_grad(out.grad @ other.data.T)
            other._accumulate_grad(self.data.T @ out.grad)
        out._backward = _backward
        return out

    def __pow__(self, value): 
        assert isinstance(value, (int, float))
        out = Tensor(self.xp.pow(self.data, value), (self, ), requires_grad = self.requires_grad, _op=f'^{value}', device = self.device)
        def _backward(): 
            # self.grad += (value * self.xp.pow(self.data, value - 1) * out.grad)
            self._accumulate_grad(value * self.xp.pow(self.data, value - 1) * out.grad)
        out._backward = _backward
        return out

    def reciprocal(self): 
        out = Tensor(self.xp.pow(self.data, -1), (self, ), requires_grad=self.requires_grad, _op='reciprocal', device = self.device)
        def _backward(): 
            # self.grad += (-1 * self.xp.pow(self.data, -2) * out.grad)
            self._accumulate_grad(-1 * self.xp.pow(self.data, -2) * out.grad)
        out._backward = _backward
        return out 

    def __div__(self, other): 
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=self.requires_grad, device = self.device)
        return self * other.reciprocal()

    def __truediv__(self, other):
        return self.__div__(other)

    def leaky_relu(self, lvalue = 0.1):
        cdata = self.data.copy()
        lindexes = self.data <= 0
        cdata[lindexes] *= lvalue
        out = Tensor(cdata, (self, ), requires_grad=self.requires_grad, _op=f'leaky_relu: {lvalue}', device = self.device)
        def _backward():
            if self.requires_grad:
                grad = self.xp.ones_like(self.data)
                grad[self.data <= 0] = lvalue
                # self.grad += out.grad * grad
                self._accumulate_grad(out.grad * grad)
        out._backward = _backward
        return out

    def relu(self): 
        return self.leaky_relu(0)

    def exp(self):
        out = Tensor(self.xp.exp(self.data), (self,), requires_grad=self.requires_grad, _op='exp', device = self.device)
        def _backward():
            self._accumulate_grad(out.grad * out.data)
        out._backward = _backward
        return out

    def log(self): 
        out = Tensor(self.xp.log(self.data), (self, ), requires_grad = self.requires_grad, _op='log', device = self.device)
        def _backward(): 
            # if self.requires_grad: 
            #     self.grad += out.grad / self.data
            self._accumulate_grad(out.grad / self.data)
        out._backward = _backward
        return out

    def tanh(self): 
        out = Tensor(self.xp.tanh(self.data), (self, ), requires_grad= self.requires_grad, _op='tanh', device = self.device)
        def _backward(): 
            # if self.requires_grad: 
            #     self.grad += out.grad * (1 - out.data ** 2)
            self._accumulate_grad(out.grad * (1 - out.data ** 2))
        out._backward = _backward
        return out

    def sigmoid(self): 
        return 1 / (1 + (-self).exp())

    def max(self, axis = None, keepdims = False): 
        out = Tensor(self.xp.max(self.data, axis = axis, keepdims = keepdims), (self, ), requires_grad=self.requires_grad, _op='max', device = self.device)
        def _backward(): 
            if not self.requires_grad: 
                return None
            grad = out.grad
            if axis is None: 
                mask = self.data == self.xp.max(self.data)

                if not keepdims: 
                    grad = self.xp.broadcast_to(grad, self.data.shape)
                # self.grad += mask * grad
                self._accumulate_grad(mask * grad)
                return

            axes = axis if isinstance(axis, tuple) else (axis,)
            axes = tuple(
                ax if ax >= 0 else self.data.ndim + ax 
                for ax in axes
            )
            if not keepdims: 
                for ax in axes: 
                    grad = self.xp.expand_dims(grad, ax)
            max_values = self.xp.max(self.data, axis = axis, keepdims=True)
            mask = self.data == max_values
            grad = self.xp.broadcast_to(grad, self.data.shape)
            # self.grad += grad * mask
            self._accumulate_grad(grad * mask)
        out._backward = _backward
        return out

    def min(self, axis = None, keepdims = False): 
        out = Tensor(self.xp.min(self.data, axis = axis, keepdims = keepdims), (self, ), requires_grad = self.requires_grad, _op = 'min', device = self.device)
        def _backward(): 
            if not self.requires_grad: 
                return None
            grad = out.grad
            if axis is None: 
                mask = self.data == self.xp.min(self.data)

                if not keepdims: 
                    grad = self.xp.broadcast_to(grad, self.data.shape)
                # self.grad += mask * grad
                self._accumulate_grad(mask * grad)
                return
            axes = axis if isinstance(axis, tuple) else (axis, )
            axes = tuple(
                ax if ax >= 0 else self.data.ndim + ax
                for ax in axes
            )
            if not keepdims: 
                for ax in axes: 
                    grad = self.xp.expand_dims(grad, ax)
            min_values = self.xp.min(self.data, axis = axes, keepdims = True)
            mask = self.data == min_values
            grad = self.xp.broadcast_to(grad, self.data.shape)
            # self.grad += grad * mask
            self._accumulate_grad(grad * mask)
        out._backward = _backward
        return out


    def sum(self, axis = None, keepdims = False):
        out = Tensor(self.xp.sum( self.data, axis = axis, keepdims = keepdims),(self,),self.requires_grad, _op='sum', device = self.device)

        def _backward():
            if not self.requires_grad:
                return
            grad = out.grad
            if axis is None:
                grad = self.xp.broadcast_to(grad,self.data.shape)
            else:
                axes = axis if isinstance(axis, tuple) else (axis,)
                axes = tuple(ax if ax >= 0 else self.data.ndim + ax for ax in axes)

                if not keepdims:
                    for ax in sorted(axes):
                        grad = self.xp.expand_dims(grad, ax)

                grad = self.xp.broadcast_to(grad,self.data.shape)

            # self.grad += grad
            self._accumulate_grad(grad)

        out._backward = _backward
        return out

    def mean(self, axis = None, keepdims = False):
        if axis is None:
            count = self.data.size
        else:
            axes = axis if isinstance(axis, tuple) else (axis,)
            axes = tuple( ax if ax >= 0 else self.data.ndim + ax for ax in axes)
            count = self.xp.prod([self.data.shape[ax] for ax in axes])
        return self.sum(axis = axis, keepdims = keepdims) / count


    def __neg__(self): 
        return self * Tensor(-1, requires_grad = self.requires_grad)
    def __sub__(self, other): 
        return self + (-other)
    def __radd__(self, other): 
        return self + other
    def __rsub__(self, other): 
        return (-self) + other
    def __rmul__(self, other): 
        return self * other
    def __rtruediv__(self, other):
        return other * self.reciprocal()

    def zero_grad(self):
        for node in self._build_topo():
            if node.requires_grad:
                node.grad = self.xp.zeros_like(node.data)
    
    def backward(self): 
        nodes = self._build_topo()
        self.grad = self.xp.ones_like(self.grad)
        for node in nodes:
            node._backward() 

    def _build_topo(self): 
        nodes, visited = [], set()
        def build_topo(node): 
            if (node not in visited) and (node.requires_grad): 
                visited.add(node)
                for child in node._prevs: 
                    build_topo(child)
                nodes.append(node)
        build_topo(self)
        return reversed(nodes)
    
    @staticmethod
    def _accumulate_grad(self, grad):
        if self.requires_grad: 
            while grad.ndim > self.grad.ndim:
                grad = grad.sum(axis=0)
            axes = tuple(
                i
                for i, (gdim, tdim) in enumerate(zip(grad.shape, self.grad.shape))
                if tdim == 1 and gdim != 1
            )
            if axes:
                grad = grad.sum(axis=axes, keepdims=True)

            self.grad += grad

    @property
    def shape(self): 
        return self.data.shape

    @property
    def ndim(self): 
        return self.data.ndim

    def __repr__(self): 
        return f'Tensor({self.data}, dtype: {self.data.dtype}, Shape: {list(self.data.shape)})'
