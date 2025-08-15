import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class MM(nn.Module):
    def __init__(self, C, D, device):
        super(MM, self).__init__()
        self.register_buffer('S', torch.randn(C, D, 15).to(device))
        self.register_buffer('H', torch.randn(15, 16).to(device))
        # number of codeblocks that input is divided into,
        # for example, if input size is 14, and C is 7, then input vector is divided into 7 codeblocks, each of size 2.
        self.C = C
        # size of each codeblock
        self.D = D
        # thresholds table.
        self.T = nn.Parameter(torch.randn(C, 15).to(device))
        self.LUT = nn.Parameter(torch.randn(C, 16, D).to(device))

    def forward(self, x):
        x = x.view(-1, self.C, self.D)
        #print(x.shape)
        x = torch.einsum('bcd,cdk->bck', x, self.S)
        x = x - self.T - 0.0001
        tanh = torch.tanh(x)
        sign = torch.sign(x)
        x = (sign - tanh).detach() + tanh
        x = torch.einsum('bcd,dk->bck', x, self.H)
        one_hot = F.one_hot(torch.argmax(x, dim=-1), num_classes=16).float()
        # temperature = 0.1
        temperature = 1
        softmax = torch.softmax(x / temperature, dim=-1)
        # x = (one_hot - softmax).detach() + softmax
        # x = torch.softmax(x, dim=-1)
        # x = torch.matmul(x, self.LUT)
        x_one_hot = torch.einsum('bck,ckd->bcd', one_hot, self.LUT)
        x_softmax = torch.einsum('bck,ckd->bcd', softmax, self.LUT)
        x = (x_one_hot - x_softmax).detach() + x_softmax
        #print(one_hot)
        #print(one_hot.shape)
        x = x.reshape(-1, self.C * self.D)
        return x




class TwoLayerPerceptron_norm1(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2, num_classes, device):
        super(TwoLayerPerceptron_norm1, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(input_size, hidden_size).to(self.device)
        self.fc2 = nn.Linear(hidden_size, hidden_size2).to(self.device)
        self.output = nn.Linear(hidden_size2, num_classes).to(self.device)
        # self.fc1 = nn.Linear(input_size, hidden_size,bias=False)
        # self.fc2 = nn.Linear(hidden_size, hidden_size2,bias=False)
        # self.output = nn.Linear(hidden_size2, num_classes,bias=False)
        # self.hardsigmoid = nn.Hardtanh()
        self.hardsigmoid = nn.ReLU()
        # self.hardsigmoid = nn.Sigmoid()
        self.hs1 = hidden_size
        self.hs2 = hidden_size2
        self.norm0 = nn.BatchNorm1d(input_size, eps=1e-5).to(self.device)
        self.norm1 = nn.BatchNorm1d(hidden_size).to(self.device)
        self.norm2 = nn.BatchNorm1d(hidden_size2).to(self.device)
    def forward(self, x):
        x = x.to(self.device)
        x = self.norm0(x)
        res = [x]
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.hardsigmoid(x)
        x = self.fc2(x)
        x = self.norm2(x)
        x = self.hardsigmoid(x)
        x = self.output(x)
        return F.log_softmax(x, dim=-1),res



class TwoLayerPerceptron_norm2(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2, num_classes, device,
                D1
                ):
        super(TwoLayerPerceptron_norm2, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(input_size, hidden_size).to(self.device)
        self.fc2 = nn.Linear(hidden_size, hidden_size2).to(self.device)
        self.output = nn.Linear(hidden_size2, num_classes).to(self.device)
        # self.fc1 = nn.Linear(input_size, hidden_size,bias=False)
        # self.fc2 = nn.Linear(hidden_size, hidden_size2,bias=False)
        # self.output = nn.Linear(hidden_size2, num_classes,bias=False)
        # self.hardsigmoid = nn.Hardtanh()
        self.hardsigmoid = nn.ReLU()
        # self.hardsigmoid = nn.Sigmoid()
        self.hs1 = hidden_size
        self.hs2 = hidden_size2
        self.norm0 = nn.BatchNorm1d(input_size, eps=1e-5).to(self.device)
        self.norm1 = nn.BatchNorm1d(hidden_size).to(self.device)
        self.norm2 = nn.BatchNorm1d(hidden_size2).to(self.device)

        self.MM1 = MM(input_size//D1,D1,device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.norm0(x)
        x = self.MM1(x)
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.hardsigmoid(x)
        res = [x]
        x = self.fc2(x)
        x = self.norm2(x)
        x = self.hardsigmoid(x)
        x = self.output(x)
        return F.log_softmax(x, dim=-1),res



class TwoLayerPerceptron_norm3(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2, num_classes, device,
                D1,D2
                ):
        super(TwoLayerPerceptron_norm3, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(input_size, hidden_size).to(self.device)
        self.fc2 = nn.Linear(hidden_size, hidden_size2).to(self.device)
        self.output = nn.Linear(hidden_size2, num_classes).to(self.device)
        # self.fc1 = nn.Linear(input_size, hidden_size,bias=False)
        # self.fc2 = nn.Linear(hidden_size, hidden_size2,bias=False)
        # self.output = nn.Linear(hidden_size2, num_classes,bias=False)
        # self.hardsigmoid = nn.Hardtanh()
        self.hardsigmoid = nn.ReLU()
        # self.hardsigmoid = nn.Sigmoid()
        self.hs1 = hidden_size
        self.hs2 = hidden_size2
        self.norm0 = nn.BatchNorm1d(input_size, eps=1e-5).to(self.device)
        self.norm1 = nn.BatchNorm1d(hidden_size).to(self.device)
        self.norm2 = nn.BatchNorm1d(hidden_size2).to(self.device)

        self.MM1 = MM(input_size//D1,D1,device)
        self.MM2 = MM(hidden_size//D2,D2,device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.norm0(x)
        x = self.MM1(x)
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.hardsigmoid(x)
        x = self.MM2(x)
        x = self.fc2(x)
        x = self.norm2(x)
        x = self.hardsigmoid(x)
        res = [x]
        x = self.output(x)
        return F.log_softmax(x, dim=-1),res


class TwoLayerPerceptron_norm4(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2, num_classes, device,
                D1,D2,D3
                ):
        super(TwoLayerPerceptron_norm4, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(input_size, hidden_size).to(self.device)
        self.fc2 = nn.Linear(hidden_size, hidden_size2).to(self.device)
        self.output = nn.Linear(hidden_size2, num_classes).to(self.device)
        # self.fc1 = nn.Linear(input_size, hidden_size,bias=False)
        # self.fc2 = nn.Linear(hidden_size, hidden_size2,bias=False)
        # self.output = nn.Linear(hidden_size2, num_classes,bias=False)
        # self.hardsigmoid = nn.Hardtanh()
        self.hardsigmoid = nn.ReLU()
        # self.hardsigmoid = nn.Sigmoid()
        self.hs1 = hidden_size
        self.hs2 = hidden_size2
        self.norm0 = nn.BatchNorm1d(input_size, eps=1e-5).to(self.device)
        self.norm1 = nn.BatchNorm1d(hidden_size).to(self.device)
        self.norm2 = nn.BatchNorm1d(hidden_size2).to(self.device)

        self.MM1 = MM(input_size//D1,D1,device)
        self.MM2 = MM(hidden_size//D2,D2,device)
        self.MM3 = MM(hidden_size2//D3,D3,device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.norm0(x)
        x = self.MM1(x)
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.hardsigmoid(x)
        x = self.MM2(x)
        x = self.fc2(x)
        x = self.norm2(x)
        x = self.hardsigmoid(x)
        x = self.MM3(x)
        x = self.output(x)
        return F.log_softmax(x, dim=-1),0





class MM_final(nn.Module):
    def __init__(self, C, D, output_size, device):
        super(MM_final, self).__init__()
        self.register_buffer('S', torch.randn(C, D, 15).to(device))
        self.register_buffer('H', torch.randn(15, 16).to(device))
        # number of codeblocks that input is divided into,
        # for example, if input size is 14, and C is 7, then input vector is divided into 7 codeblocks, each of size 2.
        self.C = C
        # size of each codeblock
        self.D = D
        # thresholds table.
        self.T = nn.Parameter(torch.randn(C, 15).to(device))
        self.LUT = nn.Parameter(torch.randn(C, 16, output_size).to(device))

    def forward(self, x):
        x = x.view(-1, self.C, self.D)
        x = torch.einsum('bcd,cdk->bck', x, self.S)
        x = x - self.T
        x[x == 0] = -1
        x = torch.sign(x)

        x = torch.einsum('bcd,dk->bck', x, self.H)
        one_hot = F.one_hot(torch.argmax(x, dim=-1), num_classes=16).float()
        x = torch.einsum('bck,cko->bco', one_hot, self.LUT)

        x = torch.sum(x, dim=1)
        return x
    





class mlp_lut(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2, num_classes, device,
                D1,D2,D3
                ):
        super(mlp_lut, self).__init__()
        self.device = device

        self.MM1 = MM_final(input_size//D1,D1,hidden_size,device)
        self.MM2 = MM_final(hidden_size//D2,D2,hidden_size2,device)
        self.MM3 = MM_final(hidden_size2//D3,D3,num_classes,device)

    def forward(self, x):

        x = x.to(self.device)
        x = self.MM1(x)
        x = self.MM2(x)
        x = self.MM3(x)

        return F.log_softmax(x, dim=-1)