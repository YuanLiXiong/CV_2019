import numpy as np
from numpy import *
# import random

def gen_sample_data(num_samples=200):

    # X range: [-100, 100]
    X = random.rand(num_samples) * num_samples - num_samples // 2
    X = sort(X)
    Y = np.zeros_like(X)
    Y[X>0] = 1
    return X, Y


if __name__ == "__main__":
    X, Y = gen_sample_data()
    print(X)
    print(Y)
