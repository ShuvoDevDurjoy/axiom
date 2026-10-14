from nn.tensor import Tensor 
from nn.init import *
from nn.modules.linear import Linear

def main(): 
    input = Tensor([1, 2], requires_grad=True)
    out = input ** 2
    print(out)


if __name__ == '__main__': 
    main()


