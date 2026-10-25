from nn.tensor import Tensor as tf
from nn.init import *
from nn.modules.linear import Linear

def main(): 
    input = tf([1.3, 2.6], dtype=tf.float32)
    out = input ** 9
    print(out)


if __name__ == '__main__': 
    main()


