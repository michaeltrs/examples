import numpy as np



def softmax(x):
    return np.exp(x) / np.exp(x).sum()


N = 128
x = np.zeros(N)
x[0] = 1.0

softmax(x)  # .sum()
x