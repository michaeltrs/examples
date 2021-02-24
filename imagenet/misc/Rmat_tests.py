import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from torch.autograd import Variable
import matplotlib.pyplot as plt


# def train_step(net, sample, loss_fn, optimizer, device, model_input_fn, loss_input_fn):
#     # sample = next(trainiter)
#     # zero the parameter gradients
#     optimizer.zero_grad()
#     # model forward pass
#     outputs = net(model_input_fn(sample, device))
#     outputs = outputs.permute(0, 2, 3, 1)
#     # print(outputs.shape)
#     # model backward pass
#     ground_truth = loss_input_fn(sample, device)
#     loss = loss_fn['mean'](outputs, ground_truth)
#     loss.backward()
#     # run optimizer
#     optimizer.step()
#     return outputs, ground_truth, loss



# decay_fn = lambda epoch: dr ** epoch


N = 1000
a = 1000
w = Variable(torch.rand(N, a, requires_grad=True), requires_grad=True)

I = torch.eye(N)


optimizer = optim.SGD([w], lr=0.01, momentum=0.9)

losses = []
for i in range(1000):
    optimizer.zero_grad()
    w1 = nn.functional.normalize(w, dim=1)
    R = torch.matmul(w1, w1.permute(1, 0))
    loss = torch.sum((I - R) ** 2)
    losses.append(loss)
    print(loss)
    loss.backward(retain_graph=True)
    # run optimizer
    optimizer.step()

plt.plot(losses)
plt.show()