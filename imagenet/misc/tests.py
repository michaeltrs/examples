import numpy as np



# def softmax(x):
#     return np.exp(x) / np.exp(x).sum()
#
#
# N = 128
# x = np.zeros(N)
# x[0] = 1.0
#
# softmax(x)  # .sum()
# x

N = 1000
x = -10 * np.ones((N, N)) + 20 * np.eye(N)

z = 1/(1 + np.exp(-x))

z.mean(axis=0)[0]


