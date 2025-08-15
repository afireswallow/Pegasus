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

class originenlargeTextCNN(nn.Module):

    def __init__(self,input_size, largein,num_classes,
                 len_vocab,ipd_vocab,
                 len_embedding_bits,ipd_embedding_bits,
                 nk, ebdin,
                 device):
        super(originenlargeTextCNN, self).__init__()
        self.largein = largein
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
        self.enlarge = nn.Linear(ebdin,largein).to(device)
        self.conv3 = nn.Conv2d(1, nk, (3, largein),bias=False).to(device)
        self.conv4 = nn.Conv2d(1, nk, (4, largein),bias=False).to(device)
        self.conv5 = nn.Conv2d(1, nk, (5, largein),bias=False).to(device)
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
        x = self.enlarge(x)
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
        return F.log_softmax(x, dim=1),0

class enlargesegTextCNN1(nn.Module):

    def __init__(self,input_size, largein,num_classes,
                 len_vocab,ipd_vocab,
                 len_embedding_bits,ipd_embedding_bits,
                 nk, ebdin,
                 device):
        super(enlargesegTextCNN1, self).__init__()
        self.largein = largein
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
        self.enlarge = nn.Linear(ebdin,largein).to(device)
        #[N,8,4]
        self.conv3 = nn.Conv2d(1, nk, (3, largein),bias=False).to(device)
        self.conv4 = nn.Conv2d(1, nk, (4, largein),bias=False).to(device)
        self.conv5 = nn.Conv2d(1, nk, (5, largein),bias=False).to(device)
        #[N,2,6,1]
        #[N,2,5,1]
        #[N,2,4,1]
        #[N,2,1,1]
        #[N,2,1,1]
        #[N,2,1,1]
        self.fc2 = nn.Linear(15*nk, num_classes).to(device)

        self.len_embedding.weight.requires_grad = False
        self.ipd_embedding.weight.requires_grad = False


    def segmentConv(self,x):
        batch = x.shape[0]
        feature1 = []
        feature2 = []
        feature3 = []
        x = x.view(batch,self.largein,8)
        x = x.view(batch,-1)

        seg_length = 64  # 每段长度
        num_segments = x.shape[1] // seg_length  # 完整的段数
        remainder = x.shape[1] % seg_length  # 剩余的长度
        # 遍历完整段
        for i in range(num_segments):
            segx = x[:, seg_length * i: seg_length * (i + 1)].clone()
            o = torch.zeros_like(x)
            o[:,seg_length * i: seg_length * (i + 1)] = segx
            o = o.view(batch,self.largein,8)
            o = o.view(batch,1,8,self.largein)
            e1 = F.relu(self.conv3(o))
            e2 = F.relu(self.conv4(o))
            e3 = F.relu(self.conv5(o))

            e1 = e1.view(batch,-1)
            e2 = e2.view(batch,-1)
            e3 = e3.view(batch,-1)

            feature1.append(e1)
            feature2.append(e2)
            feature3.append(e3)


        f1 = torch.stack(feature1)
        x1 = torch.sum(f1,dim=0)
        f2 = torch.stack(feature2)
        x2 = torch.sum(f2,dim=0)
        f3 = torch.stack(feature3)
        x3 = torch.sum(f3,dim=0)

        #print(x1.shape)
        #print(x2.shape)
        #print(x3.shape)

        x = torch.cat((x1, x2, x3), -1)
        x = x.view(batch, 1, -1)
        return x

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
        x = self.enlarge(x)
        x = self.segmentConv(x)
        x = self.fc2(x)
        x = x.view(-1, self.num_classes)
        return F.log_softmax(x, dim=1),res


class enlargesegTextCNN2(nn.Module):

    def __init__(self,input_size, largein,num_classes,
                 len_vocab,ipd_vocab,
                 len_embedding_bits,ipd_embedding_bits,
                 nk, ebdin,D1,
                 device):
        super(enlargesegTextCNN2, self).__init__()
        self.largein = largein
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
        self.enlarge = nn.Linear(ebdin,largein).to(device)
        self.conv3 = nn.Conv2d(1, nk, (3, largein),bias=False).to(device)
        self.conv4 = nn.Conv2d(1, nk, (4, largein),bias=False).to(device)
        self.conv5 = nn.Conv2d(1, nk, (5, largein),bias=False).to(device)
        self.fc2 = nn.Linear(15*nk, num_classes).to(device)

        self.MM1 =MM( (input_size*ebdin)//D1, D1,device)
        self.len_embedding.weight.requires_grad = False
        self.ipd_embedding.weight.requires_grad = False


    def segmentConv(self,x):
        batch = x.shape[0]
        feature1 = []
        feature2 = []
        feature3 = []
        x = x.view(batch,-1)

        seg_length = 64  
        num_segments = x.shape[1] // seg_length 
        remainder = x.shape[1] % seg_length 

        x = x.view(batch,8,-1)
        for i in range(8):
            segx = x[:, i, :].clone()
            o = torch.zeros_like(x)
            o[:,i,:] = segx
            o = o.view(batch,1,8,self.largein)
            
            e1 = F.relu(self.conv3(o))
            e2 = F.relu(self.conv4(o))
            e3 = F.relu(self.conv5(o))

            e1 = e1.view(batch,-1)
            e2 = e2.view(batch,-1)
            e3 = e3.view(batch,-1)

            feature1.append(e1)
            feature2.append(e2)
            feature3.append(e3)

        f1 = torch.stack(feature1)
        x1 = torch.sum(f1,dim=0)
        f2 = torch.stack(feature2)
        x2 = torch.sum(f2,dim=0)
        f3 = torch.stack(feature3)
        x3 = torch.sum(f3,dim=0)
        x = torch.cat((x1, x2, x3), -1)
        x = x.view(batch, 1, -1)
        return x

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
        x = x.view(batch,1,8,4)
        x = self.enlarge(x)
        x = self.segmentConv(x)
        x = self.fc2(x)
        x = x.view(-1, self.num_classes)
        return F.log_softmax(x, dim=1),0




class newMM(nn.Module):
    def __init__(self, C, D, numclass, device):
        super(newMM, self).__init__()
        
        # number of codeblocks that input is divided into,
        # for example, if input size is 14, and C is 7, then input vector is divided into 7 codeblocks, each of size 2.
        self.C = 16
        # size of each codeblock
        self.D = 2
        # thresholds table.
        self.register_buffer('S', torch.randn(16, 2, 15).to(device))
        self.register_buffer('H', torch.randn(30, 256).to(device))
        self.T = nn.Parameter(torch.randn(16, 15).to(device))
        self.LUT = nn.Parameter(torch.randn(8, 256, numclass).to(device))
        

    def forward(self, x):
        x = x.view(-1, 16, 2)
        x = torch.einsum('bcd,cdk->bck', x, self.S)
        x = x - self.T - 0.0001
        tanh = torch.tanh(x)
        sign = torch.sign(x)
        x = (sign - tanh).detach() + tanh    #(batch, 16, 15)
        x = x.view(-1, 8, 2, 15)
        x = x.reshape(-1, 8, 30)
        x = x.view(-1,8, 30)
        x = torch.einsum('bcd,dk->bck', x, self.H) #(batch, 8, 256)

        one_hot = F.one_hot(torch.argmax(x, dim=-1), num_classes=256).float()
        x = torch.einsum('bck,ckd->bcd', one_hot, self.LUT)
        return x



class enlargesegTextCNN_LUT(nn.Module):

    def __init__(self,input_size, largein,num_classes,
                 len_vocab,ipd_vocab,
                 len_embedding_bits,ipd_embedding_bits,
                 nk, ebdin,D1,
                 device):
        super(enlargesegTextCNN_LUT, self).__init__()
        self.largein = largein
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

        self.MM1 = newMM( (input_size*ebdin)//D1, D1,num_classes,device)
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
        x = self.MM1(x.view(batch,-1)) 
        x = x.sum(dim=1) 
        x = x.view(-1, self.num_classes)
        return F.log_softmax(x, dim=1)

