import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

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

class TextCNN1(nn.Module):

    def __init__(self,input_size, num_classes,
                 len_vocab,ipd_vocab,
                 len_embedding_bits,ipd_embedding_bits,
                 nk, ebdin,
                 device):
        super(TextCNN1, self).__init__()
        self.ebdin = ebdin
        self.input_size = input_size
        self.len_embedding_bits = len_embedding_bits
        self.ipd_embedding_bits = ipd_embedding_bits
        ebdbit = len_embedding_bits + ipd_embedding_bits
        self.len_vocab = len_vocab
        self.ipd_vocab = ipd_vocab
        self.num_classes = num_classes
        self.device = device
        self.nk = nk
        self.len_embedding = nn.Embedding(self.len_vocab, self.len_embedding_bits).to(device)
        self.ipd_embedding = nn.Embedding(self.ipd_vocab, self.ipd_embedding_bits).to(device)

        
        self.fc1 = nn.Linear(ebdbit,ebdin).to(device)
        self.conv3 = nn.Conv2d(1, nk, (3, ebdin)).to(device)
        self.conv4 = nn.Conv2d(1, nk, (4, ebdin)).to(device)
        self.conv5 = nn.Conv2d(1, nk, (5, ebdin)).to(device)
        self.fc2 = nn.Linear(15*nk, num_classes).to(device)


    def forward(self, x):
        len_x = x[:,:,0].to(self.device)
        ipd_x = x[:,:,1].to(self.device)
        len_x = len_x.to(self.device).long()
        ipd_x = ipd_x.to(self.device).long()
        len_x = self.len_embedding(len_x)
        ipd_x = self.ipd_embedding(ipd_x)
        batch = x.shape[0]
        x = torch.cat((len_x, ipd_x), dim=-1)
        x = self.fc1(x)
        res = [x.view(batch,-1)]
        x = x .view(x.shape[0],1,x.shape[1],x.shape[2])
        x1 = F.relu(self.conv3(x))
        x2 = F.relu(self.conv4(x))
        x3 = F.relu(self.conv5(x))
        x1 = x1.view(batch,-1)
        x2 = x2.view(batch,-1)
        x3 = x3.view(batch,-1)
        x = torch.cat((x1, x2, x3), -1)
        x = x.view(batch, 1, -1)
        x = self.fc2(x)
        x = x.view(-1, self.num_classes)
        return F.log_softmax(x, dim=1),res



class TextCNN2(nn.Module):

    def __init__(self,input_size, num_classes,
                 len_vocab,ipd_vocab,
                 len_embedding_bits,ipd_embedding_bits,
                 nk, ebdin,
                 device,
                 D1):
        super(TextCNN2, self).__init__()
        self.ebdin = ebdin
        self.input_size = input_size
        self.len_embedding_bits = len_embedding_bits
        self.ipd_embedding_bits = ipd_embedding_bits
        ebdbit = len_embedding_bits + ipd_embedding_bits
        self.len_vocab = len_vocab
        self.ipd_vocab = ipd_vocab
        self.num_classes = num_classes
        self.device = device
        self.nk = nk
        self.len_embedding = nn.Embedding(self.len_vocab, self.len_embedding_bits).to(device)
        self.ipd_embedding = nn.Embedding(self.ipd_vocab, self.ipd_embedding_bits).to(device)

        
        self.fc1 = nn.Linear(ebdbit,ebdin).to(device)
        self.conv3 = nn.Conv2d(1, nk, (3, ebdin)).to(device)
        self.conv4 = nn.Conv2d(1, nk, (4, ebdin)).to(device)
        self.conv5 = nn.Conv2d(1, nk, (5, ebdin)).to(device)
        self.fc2 = nn.Linear(15*nk, num_classes).to(device)

        self.MM1 = MM(32//D1,D1,device)


    def forward(self, x):
        len_x = x[:,:,0].to(self.device)
        ipd_x = x[:,:,1].to(self.device)
        len_x = len_x.to(self.device).long()
        ipd_x = ipd_x.to(self.device).long()
        len_x = self.len_embedding(len_x)
        ipd_x = self.ipd_embedding(ipd_x)
        batch = x.shape[0]
        x = torch.cat((len_x, ipd_x), dim=-1)
        x = self.fc1(x)
        x = self.MM1(x.view(batch,-1))
        x = x .view(batch,1,self.input_size,self.ebdin)
        x1 = F.relu(self.conv3(x))
        x2 = F.relu(self.conv4(x))
        x3 = F.relu(self.conv5(x))
        x1 = x1.view(batch,-1)
        x2 = x2.view(batch,-1)
        x3 = x3.view(batch,-1)
        x = torch.cat((x1, x2, x3), -1)
        x = x.view(batch, -1)
        if self.nk%4==3:
            zero_column = torch.zeros(batch, 3).to(self.device)
            x = torch.cat((x, zero_column), dim=1)
        res = [x]
        if self.nk%4==3:
            x = x[:, :-3]
        x = self.fc2(x)
        x = x.view(-1, self.num_classes)
        return F.log_softmax(x, dim=1),res





class TextCNN3(nn.Module):

    def __init__(self,input_size, num_classes,
                 len_vocab,ipd_vocab,
                 len_embedding_bits,ipd_embedding_bits,
                 nk, ebdin,
                 device,
                 D1,D2):
        super(TextCNN3, self).__init__()
        self.ebdin = ebdin
        self.input_size = input_size
        self.len_embedding_bits = len_embedding_bits
        self.ipd_embedding_bits = ipd_embedding_bits
        ebdbit = len_embedding_bits + ipd_embedding_bits
        self.len_vocab = len_vocab
        self.ipd_vocab = ipd_vocab
        self.num_classes = num_classes
        self.device = device
        self.nk = nk
        self.len_embedding = nn.Embedding(self.len_vocab, self.len_embedding_bits).to(device)
        self.ipd_embedding = nn.Embedding(self.ipd_vocab, self.ipd_embedding_bits).to(device)

        
        self.fc1 = nn.Linear(ebdbit,ebdin).to(device)
        self.conv3 = nn.Conv2d(1, nk, (3, ebdin)).to(device)
        self.conv4 = nn.Conv2d(1, nk, (4, ebdin)).to(device)
        self.conv5 = nn.Conv2d(1, nk, (5, ebdin)).to(device)
        self.fc2 = nn.Linear(15*nk, num_classes).to(device)

        self.MM1 = MM(32//D1,D1,device)
        if nk!=3:
            self.MM2 = MM((15*nk)//D2,D2,device)
        else:
            self.MM2 = MM((15*nk+3)//D2,D2,device)

    def forward(self, x):
        len_x = x[:,:,0].to(self.device)
        ipd_x = x[:,:,1].to(self.device)
        len_x = len_x.to(self.device).long()
        ipd_x = ipd_x.to(self.device).long()
        len_x = self.len_embedding(len_x)
        ipd_x = self.ipd_embedding(ipd_x)
        batch = x.shape[0]
        x = torch.cat((len_x, ipd_x), dim=-1)
        x = self.fc1(x)
        x = self.MM1(x.view(batch,-1))
        x = x .view(batch,1,self.input_size,self.ebdin)
        x1 = F.relu(self.conv3(x))
        x2 = F.relu(self.conv4(x))
        x3 = F.relu(self.conv5(x))
        x1 = x1.view(batch,-1)
        x2 = x2.view(batch,-1)
        x3 = x3.view(batch,-1)
        x = torch.cat((x1, x2, x3), -1)
        x = x.view(batch, -1)
        if self.nk%4==3:
            zero_column = torch.zeros(batch, 3).to(self.device)
            x = torch.cat((x, zero_column), dim=1)
        x = self.MM2(x)
        if self.nk%4==3:
            x = x[:, :-3]
        x = self.fc2(x)
        x = x.view(-1, self.num_classes)
        return F.log_softmax(x, dim=1),0



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




class MM_final_conv(nn.Module):
    def __init__(self, C, D, output_size, nk, device):
        super(MM_final_conv, self).__init__()
        self.register_buffer('S', torch.randn(C, D, 15).to(device))
        self.register_buffer('H', torch.randn(15, 16).to(device))
        # number of codeblocks that input is divided into,
        # for example, if input size is 14, and C is 7, then input vector is divided into 7 codeblocks, each of size 2.
        self.C = C
        # size of each codeblock
        self.D = D
        # thresholds table.
        self.T = nn.Parameter(torch.randn(C, 15).to(device))
        #self.LUT = nn.Parameter(torch.randn(C, 16, output_size, nk).to(device))
        self.LUT = nn.ParameterList([
            nn.Parameter(torch.randn(C, 16, output_size).to(device)) for _ in range(nk)
        ])

    def forward(self, x):
        x = x.view(-1, self.C, self.D)
        x = torch.einsum('bcd,cdk->bck', x, self.S)
        x = x - self.T
        x[x == 0] = -1
        x = torch.sign(x)
        
        x = torch.einsum('bcd,dk->bck', x, self.H)
        one_hot = F.one_hot(torch.argmax(x, dim=-1), num_classes=16).float()
        result = []
        for i in range(len(self.LUT)):
            x0 = torch.einsum('bck,cko->bco', one_hot, self.LUT[i])
            x0 = torch.sum(x0, dim=1)
            result.append(x0)
        x = torch.stack(result,dim=1)
        return x






class cnn_lut(nn.Module):

    def __init__(self,input_size, num_classes,
                 len_vocab,ipd_vocab,
                 len_embedding_bits,ipd_embedding_bits,
                 nk, ebdin,
                 device,
                 D1,D2):
        super(cnn_lut, self).__init__()
        self.ebdin = ebdin
        self.input_size = input_size
        self.len_embedding_bits = len_embedding_bits
        self.ipd_embedding_bits = ipd_embedding_bits
        ebdbit = len_embedding_bits + ipd_embedding_bits
        self.len_vocab = len_vocab
        self.ipd_vocab = ipd_vocab
        self.num_classes = num_classes
        self.device = device
        self.nk = nk

        self.convMM3 = MM_final_conv(ebdin*input_size//D1, D1, input_size-3+1, nk, device)
        self.convMM4 = MM_final_conv(ebdin*input_size//D1, D1, input_size-4+1, nk, device)
        self.convMM5 = MM_final_conv(ebdin*input_size//D1, D1, input_size-5+1, nk, device)

        if nk!=3:
            self.MM2 = MM_final((15*nk)//D2,D2,num_classes,device)
        else:
            self.MM2 = MM_final((15*nk+3)//D2,D2,num_classes,device)
        
        self.lenebdLUT = nn.Parameter(torch.randn(len_vocab, ebdin).to(device))
        self.ipdebdLUT = nn.Parameter(torch.randn(ipd_vocab, ebdin).to(device))

    def forward(self, x):
        len_x = x[:,:,0].to(self.device)
        ipd_x = x[:,:,1].to(self.device)
        len_x = len_x.to(self.device).long()
        ipd_x = ipd_x.to(self.device).long()
        len_x = self.lenebdLUT[len_x]
        ipd_x = self.ipdebdLUT[ipd_x]
        x = ipd_x+len_x
        batch = x.shape[0]

        x1 = self.convMM3(x)
        x1 = x1.view(batch,-1)
        x2 = self.convMM4(x)
        x2 = x2.view(batch,-1)
        x3 = self.convMM5(x)
        x3 = x3.view(batch,-1)
        
        x = torch.cat((x1, x2, x3), -1)
        x = x.view(batch, -1)
        
        if self.nk%4==3:
            zero_column = torch.zeros(batch, 3).to(self.device)
            x = torch.cat((x, zero_column), dim=1)
        x = self.MM2(x)
        if self.nk%4==3:
            x = x[:, :-3]
        return F.log_softmax(x, dim=1)
