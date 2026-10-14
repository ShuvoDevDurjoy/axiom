import numpy as np

class Tensor:
    def __init__(self, data: np.ndarray, _childs: tuple = (), requires_grad = False, _op = None): 
        self.data = np.asarray(data, dtype = np.float32)
        self._backward = lambda: None
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._prevs = set(_childs)
        self.requires_grad = requires_grad
        self._op = _op

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
        out = Tensor(np.transpose(self.data, axes), (self,), self.requires_grad, _op='transpose')
        def _backward(): 
            if self.requires_grad: 
                inverse_axes = np.argsort(axes)
                self.grad += np.transpose(out.grad, inverse_axes)
        out._backward = _backward
        return out

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = Tensor(self.data.reshape(*shape), (self,), self.requires_grad, _op='reshape')
        def _backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(self.shape)
        out._backward = _backward
        return out

    def gather(self, indices):
        indices = np.asarray(indices)
        if not np.issubdtype(indices.dtype, np.integer):
            raise TypeError("gather indices must be integers")
        indices = indices.astype(np.intp, copy=False)
        if np.any(indices < 0) or np.any(indices >= self.data.size):
            indices_zero = [i for i in indices if i < 0]
            indices_more = [i for i in indices if i >= self.data.size]
            print(indices_zero)
            print(indices_more)
            raise IndexError("gather index is outside the flattened Tensor")

        out = Tensor(self.data.reshape(-1)[indices], (self,), self.requires_grad, _op='gather')

        def _backward():
            if self.requires_grad:
                np.add.at(self.grad.reshape(-1), indices.reshape(-1), out.grad.reshape(-1))

        out._backward = _backward
        return out

    def broadcast_to(self, shape: tuple): 
        if len(shape) < self.ndim: 
            raise ValueError(f"Can not broadcast to fewer dimension from {self.ndim} to {len(shape)}")

        for source, target in zip((1,) * (len(shape) - self.ndim) + self.shape, shape): 
            if source != target and source != 1: 
                raise ValueError(f"Can not Broadcast shape: from {self.shape} to {shape}")

        out = Tensor(np.broadcast_to(self.data, shape), (self, ), requires_grad= self.requires_grad, _op ='broadcast_to')
        def _backward(): 
            if self.requires_grad: 
                self._accumulate_grad(out.grad, self)
        out._backward = _backward
        return out


    def __getitem__(self, key):
        out = Tensor(self.data[key], (self, ), requires_grad= self.requires_grad, _op=f'getitem: {key}')
        def _backward(): 
            if self.requires_grad: 
                np.add.at(self.grad, key, out.grad)
        out._backward = _backward
        return out

    def requires_grad_(self, requires_grad: bool):
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if self.requires_grad else None

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=self.requires_grad)
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data + other.data, (self, other), requires_grad=requires_grad, _op='+')
        def _backward(): 
            if(self.requires_grad):
                self._accumulate_grad(out.grad, self)
            if(other.requires_grad): 
                self._accumulate_grad(out.grad, other)
        out._backward = _backward
        return out

    def __mul__(self, other): 
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=self.requires_grad)
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data * other.data, (self, other), requires_grad=requires_grad, _op='*')
        def _backward(): 
            if self.requires_grad: 
                self._accumulate_grad(other.data * out.grad, self)
            if other.requires_grad:
                self._accumulate_grad(self.data * out.grad, other)
        out._backward = _backward
        return out

    def __matmul__(self, other): 
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=self.requires_grad)
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data @ other.data, (self, other), requires_grad=requires_grad, _op='@')
        def _backward(): 
            if self.requires_grad: 
                self._accumulate_grad(out.grad @ other.data.T, self)
            if other.requires_grad:
                self._accumulate_grad(self.data.T @ out.grad, other)
        out._backward = _backward
        return out

    def __pow__(self, value): 
        assert isinstance(value, (int, float))
        out = Tensor(np.pow(self.data, value), (self, ), requires_grad = self.requires_grad, _op='**')
        def _backward(): 
            if self.requires_grad: 
                self.grad += (value * np.pow(self.data, value - 1) * out.grad)
        out._backward = _backward
        return out

    def reciprocal(self): 
        out = Tensor(np.pow(self.data, -1), (self, ), requires_grad=self.requires_grad, _op='reciprocal')
        def _backward(): 
            if self.requires_grad:
                self.grad += (-1 * np.pow(self.data, -2) * out.grad)
        out._backward = _backward
        return out 

    def __div__(self, other): 
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=self.requires_grad)
        return self * other.reciprocal()

    def __truediv__(self, other):
        return self.__div__(other)

    def leaky_relu(self, lvalue = 0.1):
        cdata = self.data.copy()
        lindexes = self.data <= 0
        cdata[lindexes] *= lvalue
        out = Tensor(cdata, (self, ), requires_grad=self.requires_grad, _op=f'leaky_relu: {lvalue}')
        def _backward():
            if self.requires_grad:
                grad = np.ones_like(self.data)
                grad[self.data <= 0] = lvalue
                self.grad += out.grad * grad
        out._backward = _backward
        return out

    def relu(self): 
        return self.leaky_relu(0)

    def exp(self):
        out = Tensor(np.exp(self.data), (self,), requires_grad=self.requires_grad, _op='exp')
        def _backward():
            if self.requires_grad:
                self.grad += out.grad * out.data
        out._backward = _backward
        return out

    def log(self): 
        out = Tensor(np.log(self.data), (self, ), requires_grad = self.requires_grad, _op='log')
        def _backward(): 
            if self.requires_grad: 
                self.grad += out.grad / self.data
        out._backward = _backward
        return out

    def tanh(self): 
        out = Tensor(np.tanh(self.data), (self, ), requires_grad= self.requires_grad, _op='tanh')
        def _backward(): 
            if self.requires_grad: 
                self.grad += out.grad * (1 - out.data ** 2)
        out._backward = _backward
        return out

    def sigmoid(self): 
        return 1 / (1 + (-self).exp())

    def max(self, axis = None, keepdims = False): 
        out = Tensor(np.max(self.data, axis = axis, keepdims = keepdims), (self, ), requires_grad=self.requires_grad, _op='max')
        def _backward(): 
            if not self.requires_grad: 
                return None
            grad = out.grad
            if axis is None: 
                mask = self.data == np.max(self.data)

                if not keepdims: 
                    grad = np.broadcast_to(grad, self.data.shape)
                self.grad += mask * grad
                return

            axes = axis if isinstance(axis, tuple) else (axis,)
            axes = tuple(
                ax if ax >= 0 else self.data.ndim + ax 
                for ax in axes
            )
            if not keepdims: 
                for ax in axes: 
                    grad = np.expand_dims(grad, ax)
            max_values = np.max(self.data, axis = axis, keepdims=True)
            mask = self.data == max_values
            grad = np.broadcast_to(grad, self.data.shape)
            self.grad += grad * mask
        out._backward = _backward
        return out

    def min(self, axis = None, keepdims = False): 
        out = Tensor(np.min(self.data, axis = axis, keepdims = keepdims), (self, ), requires_grad = self.requires_grad, _op = 'min')
        def _backward(): 
            if not self.requires_grad: 
                return None
            grad = out.grad
            if axis is None: 
                mask = self.data == np.min(self.data)

                if not keepdims: 
                    grad = np.broadcast_to(grad, self.data.shape)
                self.grad += mask * grad
            axes = axis if isinstance(axis, tuple) else (axis, )
            axes = tuple(
                ax if ax >= 0 else self.data.ndim + ax
                for ax in axes
            )
            if not keepdims: 
                for ax in axes: 
                    grad = np.expand_dims(grad, ax)
            min_values = np.min(self.data, axis = axes, keepdims = True)
            mask = self.data == min_values
            grad = np.broadcast_to(grad, self.data.shape)
            self.grad += grad * mask
        out._backward = _backward
        return out


    def sum(self, axis = None, keepdims = False):
        out = Tensor(np.sum( self.data, axis = axis, keepdims = keepdims),(self,),self.requires_grad, _op='sum')

        def _backward():
            if not self.requires_grad:
                return
            grad = out.grad
            if axis is None:
                grad = np.broadcast_to(grad,self.data.shape)
            else:
                axes = axis if isinstance(axis, tuple) else (axis,)
                axes = tuple(ax if ax >= 0 else self.data.ndim + ax for ax in axes)

                if not keepdims:
                    for ax in sorted(axes):
                        grad = np.expand_dims(grad, ax)

                grad = np.broadcast_to(grad,self.data.shape)

            self.grad += grad

        out._backward = _backward
        return out

    def mean(self, axis = None, keepdims = False):
        if axis is None:
            count = self.data.size
        else:
            axes = axis if isinstance(axis, tuple) else (axis,)
            axes = tuple( ax if ax >= 0 else self.data.ndim + ax for ax in axes)
            count = np.prod([self.data.shape[ax] for ax in axes])
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
        nodes = self._build_topo()
        for node in nodes: 
            node.grad = np.zeros_like(node.grad)
    
    def backward(self): 
        nodes = self._build_topo()
        self.grad = np.ones_like(self.grad)
        for node in nodes:
            node._backward() 

    def _build_topo(self): 
        nodes, visited = [], set()
        def build_topo(node): 
            if node not in visited: 
                visited.add(node)
                for child in node._prevs: 
                    build_topo(child)
                nodes.append(node)
        build_topo(self)
        return reversed(nodes)

    def _accumulate_grad(self, grad, target):
        while grad.ndim > target.grad.ndim:
            grad = grad.sum(axis=0)
        axes = tuple(
            i
            for i, (gdim, tdim) in enumerate(zip(grad.shape, target.grad.shape))
            if tdim == 1 and gdim != 1
        )
        if axes:
            grad = grad.sum(axis=axes, keepdims=True)

        target.grad += grad

    @property
    def shape(self): 
        return self.data.shape

    @property
    def ndim(self): 
        return self.data.ndim

    def __repr__(self): 
        return f'Tensor({self.data}, Shape: {list(self.data.shape)})'
