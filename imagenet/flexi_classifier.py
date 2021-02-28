import torch
import torch.nn as nn
import torch.nn.functional as F



class FlexiSoftmaxClassifier(nn.Module):
    def __init__(self, N):
        super(FlexiSoftmaxClassifier, self).__init__()
        self.N = N
        self.R = nn.Parameter(torch.rand(N, N))  #  + 1000 * torch.eye(N))
        # nn.init.xavier_uniform(self.R_)
        self.I = torch.eye(N)
        # self.CE = nn.CrossEntropyLoss()

    def forward(self, l):
        # R_norm = self.R_  # F.normalize(self.R_, dim=1)
        # self.R = torch.matmul(R_norm, R_norm.permute(1, 0))
        # self.R = F.softmax(self.R, dim=1)
        # self.R = F.sigmoid(self.R)
        # print("l: ", l.shape)
        # print("R: ", self.R.shape)
        # print("R: ", self.R.sum(dim=1).mean(), self.R.sum(dim=0).mean())
        R = self.R  # F.sigmoid(self.R)
        loh = torch.zeros(l.shape[0], self.N).to(l.device)
        loh = loh.scatter(1, l.unsqueeze(-1).to(torch.int64), 1)
        laff = torch.matmul(loh, R)  # .permute(1, 0))
        # ce_term = self.CE(y, laff)
        penalty = torch.mean((self.I.to(self.R.device) - R) ** 2)
        return laff, penalty  # ce_term, pen_term, laff, self.R

    # def penalty(self):
    #     return torch.sum((self.I - self.R) ** 2)
