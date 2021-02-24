import torch
import torch.nn as nn
import torch.nn.functional as F



class FlexiSoftmaxClassifier(nn.Module):
    def __init__(self, N):
        super(FlexiSoftmaxClassifier, self).__init__()
        self.R_ = nn.Parameter(torch.randn(N, N))  # torch.cuda.FloatTensor(N, N))
        # nn.init.xavier_uniform(self.R_)
        self.I = torch.eye(N)
        # self.CE = nn.CrossEntropyLoss()

    def forward(self, l):
        self.R_ = nn.Parameter(F.normalize(self.R_, dim=1))
        self.R = torch.matmul(self.R_, self.R_.permute(1, 0))
        print("l: ", l.shape)
        print("R: ", self.R.shape)
        laff = torch.matmul(l, self.R.permute(1, 0))
        # ce_term = self.CE(y, laff)
        # pen_term = torch.sum((self.I - self.R) ** 2)
        return laff  # ce_term, pen_term, laff, self.R

    def penalty(self):
        return torch.sum((self.I - self.R) ** 2)
