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





class bigcnn1_linear(nn.Module):

    def __init__(self,num_classes,device,seglength,convdim):
        super(bigcnn1_linear, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(60, 512).to(self.device)
        self.fc2 = nn.Linear(512, 256).to(self.device)
        self.fc3 = nn.Linear(256, 64).to(self.device)
        self.fc4 = nn.Linear(64,16).to(self.device)
        self.norm1 = nn.BatchNorm1d(60, eps=1e-5).to(self.device)
        self.norm2 = nn.BatchNorm1d(512).to(self.device)
        self.norm3 = nn.BatchNorm1d(256).to(self.device)
        self.norm4 = nn.BatchNorm1d(64).to(self.device)
        nk = 16
        self.activate = nn.ReLU()
        self.fc5 = nn.Linear(16,2).to(self.device)
        self.fc6 = nn.Linear(16,num_classes).to(self.device)
    
    def forward(self, x):
        x = x.to(self.device)
        x = x.to(dtype=torch.float32)
        batch = x.shape[0]
        res = [x.view(-1,60)]
        x = x.view(batch*8, 60)
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.norm2(x)
        x = self.fc2(x)
        x = self.norm3(x)
        x = self.fc3(x)
        x = self.norm4(x)
        x = self.fc4(x)
        
        x = self.activate(x)
        x = self.fc5(x)
        x = self.activate(x)
        x = x.view(batch, 16)
        x = self.fc6(x)
        return F.log_softmax(x, dim=-1),res


class bigcnn1_quantilinear(nn.Module):

    def __init__(self,num_classes,device,seglength,convdim):
        super(bigcnn1_quantilinear, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(60, 512).to(self.device)
        self.fc2 = nn.Linear(512, 256).to(self.device)
        self.fc3 = nn.Linear(256, 64).to(self.device)
        self.fc4 = nn.Linear(64,16).to(self.device)
        self.norm1 = nn.BatchNorm1d(60, eps=1e-5).to(self.device)
        self.norm2 = nn.BatchNorm1d(512).to(self.device)
        self.norm3 = nn.BatchNorm1d(256).to(self.device)
        self.norm4 = nn.BatchNorm1d(64).to(self.device)
        nk = 16
        self.activate = nn.ReLU()
        self.fc5 = nn.Linear(16,2).to(self.device)
        self.fc6 = nn.Linear(16,num_classes).to(self.device)
        self.MM1 = MM(60//2,2,device)
    
    def forward(self, x):
        x = x.to(self.device)
        x = x.to(dtype=torch.float32)
        batch = x.shape[0]
        res = [x.view(-1,60)]
        x = x.view(batch*8, 60)
        x = self.MM1(x)
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.norm2(x)
        x = self.fc2(x)
        x = self.norm3(x)
        x = self.fc3(x)
        x = self.norm4(x)
        x = self.fc4(x)
        
        x = self.activate(x)
        x = self.fc5(x)
        x = self.activate(x)
        x = x.view(batch, 16)
        x = self.fc6(x)
        return F.log_softmax(x, dim=-1),res


class bigcnn2_getdata_w(nn.Module):

    def __init__(self,num_classes,device,seglength,convdim):
        super(bigcnn2_getdata_w, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(60, 512).to(self.device)
        self.fc2 = nn.Linear(512, 256).to(self.device)
        self.fc3 = nn.Linear(256, 64).to(self.device)
        self.fc4 = nn.Linear(64,16).to(self.device)
        self.norm1 = nn.BatchNorm1d(60, eps=1e-5).to(self.device)
        self.norm2 = nn.BatchNorm1d(512).to(self.device)
        self.norm3 = nn.BatchNorm1d(256).to(self.device)
        self.norm4 = nn.BatchNorm1d(64).to(self.device)
        nk = 16
        self.seg_length =seglength
        self.activate = nn.ReLU()
        self.fc5 = nn.Linear(16,2).to(self.device)
        self.fc6 = nn.Linear(16,num_classes).to(self.device)
        self.MM1 = MM(60//2,2,device)
    
    def forward(self, x):
        x = x.to(self.device)
        x = x.to(dtype=torch.float32)
        batch = x.shape[0]
        res = [x.view(-1,60)]
        x = x.view(batch*8, 60)
        x = self.MM1(x)
        x = x.view(batch*8, 60)
        seg_length = self.seg_length
        numsplit = 60//seg_length
        feature = []
        for i in range(numsplit-1):
            segx = x[:, seg_length * i: seg_length * (i + 1)].clone()
            o = torch.zeros_like(x)
            o[:,seg_length * i: seg_length * (i + 1)] = segx

            o = self.norm1(o)
            o = self.fc1(o)
            o = self.norm2(o)
            o = self.fc2(o)
            o = self.norm3(o)
            o = self.fc3(o)
            o = self.norm4(o)
            o = self.fc4(o)
            feature.append(o)
        features = torch.cat(feature, dim=-1)
        return 0,[features]





class bigcnn2_getdata_b(nn.Module):

    def __init__(self,num_classes,device,seglength,convdim):
        super(bigcnn2_getdata_b, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(60, 512).to(self.device)
        self.fc2 = nn.Linear(512, 256).to(self.device)
        self.fc3 = nn.Linear(256, 64).to(self.device)
        self.fc4 = nn.Linear(64,16).to(self.device)
        self.norm1 = nn.BatchNorm1d(60, eps=1e-5).to(self.device)
        self.norm2 = nn.BatchNorm1d(512).to(self.device)
        self.norm3 = nn.BatchNorm1d(256).to(self.device)
        self.norm4 = nn.BatchNorm1d(64).to(self.device)
        nk = 16
        self.seg_length =seglength
        self.activate = nn.ReLU()
        self.fc5 = nn.Linear(16,2).to(self.device)
        self.fc6 = nn.Linear(16,num_classes).to(self.device)
        self.MM1 = MM(60//2,2,device)
    
    def forward(self, x):
        x = x.to(self.device)
        x = x.to(dtype=torch.float32)
        batch = x.shape[0]
        res = [x.view(-1,60)]
        x = x.view(batch*8, 60)
        x = self.MM1(x)
        x = x.view(batch*8, 60)
        seg_length = self.seg_length
        numsplit = 60//seg_length
        feature = []
        for i in range(numsplit-1,numsplit):
            segx = x[:, seg_length * i: seg_length * (i + 1)].clone()
            o = torch.zeros_like(x)
            o[:,seg_length * i: seg_length * (i + 1)] = segx

            o = self.norm1(o)
            o = self.fc1(o)
            o = self.norm2(o)
            o = self.fc2(o)
            o = self.norm3(o)
            o = self.fc3(o)
            o = self.norm4(o)
            o = self.fc4(o)
            feature.append(o)
        features = torch.cat(feature, dim=-1)
        return 0,[features]


class bigcnn2_seg_conv(nn.Module):

    def __init__(self,num_classes,device,seglength,convdim):
        super(bigcnn2_seg_conv, self).__init__()
        self.seg_length = seglength
        self.device = device
        self.convdim = convdim
        self.activate = nn.ReLU()
        
        nk=32
        self.nk = nk

        self.conv1_1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1, bias=True).to(device)
        self.conv1_2 = nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2, bias=True).to(device)
        self.conv1_3 = nn.Conv1d(1, 32, kernel_size=7, stride=1, padding=3, bias=True).to(device)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2).to(device)
        self.conv3 = nn.Conv2d(3, nk, (7, 8)).to(device)
        self.conv4 = nn.Conv2d(3, nk, (9, 8)).to(device)
        self.conv5 = nn.Conv2d(3, nk, (11, 8)).to(device)
        self.convconv3 = nn.Conv1d(in_channels=nk, out_channels=nk//2, kernel_size=3, padding=1).to(device)
        self.convconv4 = nn.Conv1d(in_channels=nk, out_channels=nk//2, kernel_size=3, padding=1).to(device)
        self.convconv5 = nn.Conv1d(in_channels=nk, out_channels=nk//2, kernel_size=3, padding=1).to(device)
        
        self.concretconv1 = nn.Conv1d(in_channels=nk//2, out_channels=nk//4, kernel_size=3, padding=1).to(device)
        self.concretconv2 = nn.Conv1d(in_channels=nk//4, out_channels=nk//8, kernel_size=3, padding=1).to(device)
        
        self.fc1 = nn.Linear( 36,16).to(self.device)
        self.fc2 = nn.Linear( 16,2).to(self.device)
        self.fc3 = nn.Linear(16,num_classes).to(self.device)

    
    def forward(self, x):
        x = x.to(self.device)
        x = x.to(dtype=torch.float32)
        seg_length = self.seg_length
        numsplit = 480//seg_length
        origin_batch = x.shape[0]
        batch = origin_batch*numsplit
        x = x.view(batch,1,self.convdim)
        x1 = self.conv1_1(x)
        x1 = self.maxpool(x1)
        x2 = self.conv1_2(x)
        x2 = self.maxpool(x2)
        x3 = self.conv1_3(x)
        x3 = self.maxpool(x3)
        x = torch.stack([x1, x2, x3], dim=1)
        x = x.view(batch,3,32,8)
        x1 = self.conv3(x)
        x2 = self.conv4(x)
        x3 = self.conv5(x)
        x1 = x1.view(batch,self.nk,-1)
        x2 = x2.view(batch,self.nk,-1)
        x3 = x3.view(batch,self.nk,-1)
        x1 = self.convconv3(x1)
        x1 = self.maxpool(x1)
        x2 = self.convconv4(x2)
        x2 = self.maxpool(x2)
        x3 = self.convconv5(x3)
        x3 = self.maxpool(x3)
        x = torch.cat([x1, x2, x3], dim=-1)
        x = self.concretconv1(x)
        x = self.maxpool(x)
        x = self.concretconv2(x)
        x = self.maxpool(x)
        x = x.view(batch,-1)
        x = self.fc1(x)
        x = self.activate(x)
        x = self.fc2(x)
        x = self.activate(x)
        x = x.view(origin_batch,8,10,2)
        x = x.sum(dim=2)
        x = x.view(origin_batch,-1)
        res = [x.view(-1,2)]
        x = self.fc3(x)
        return F.log_softmax(x, dim=-1),res
    








class bigcnn3_linear_after_conv(nn.Module):

    def __init__(self,num_classes,device,seglength,convdim):
        super(bigcnn3_linear_after_conv, self).__init__()
        self.seg_length = seglength
        self.device = device
        self.convdim = convdim
        self.activate = nn.ReLU()
        
        nk=32
        self.nk = nk

        self.conv1_1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1, bias=True).to(device)
        self.conv1_2 = nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2, bias=True).to(device)
        self.conv1_3 = nn.Conv1d(1, 32, kernel_size=7, stride=1, padding=3, bias=True).to(device)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2).to(device)
        self.conv3 = nn.Conv2d(3, nk, (7, 8)).to(device)
        self.conv4 = nn.Conv2d(3, nk, (9, 8)).to(device)
        self.conv5 = nn.Conv2d(3, nk, (11, 8)).to(device)
        self.convconv3 = nn.Conv1d(in_channels=nk, out_channels=nk//2, kernel_size=3, padding=1).to(device)
        self.convconv4 = nn.Conv1d(in_channels=nk, out_channels=nk//2, kernel_size=3, padding=1).to(device)
        self.convconv5 = nn.Conv1d(in_channels=nk, out_channels=nk//2, kernel_size=3, padding=1).to(device)
        
        self.concretconv1 = nn.Conv1d(in_channels=nk//2, out_channels=nk//4, kernel_size=3, padding=1).to(device)
        self.concretconv2 = nn.Conv1d(in_channels=nk//4, out_channels=nk//8, kernel_size=3, padding=1).to(device)
        
        self.fc1 = nn.Linear( 36,16).to(self.device)
        self.fc2 = nn.Linear( 16,2).to(self.device)

        self.MM2 = MM(2//2,2,device)

        self.fc3 = nn.Linear(16,num_classes).to(self.device)

    
    def forward(self, x):
        x = x.to(self.device)
        x = x.to(dtype=torch.float32)
        seg_length = self.seg_length
        numsplit = 480//seg_length
        origin_batch = x.shape[0]
        batch = origin_batch*numsplit
        x = x.view(batch,1,self.convdim)
        x1 = self.conv1_1(x)
        x1 = self.maxpool(x1)
        x2 = self.conv1_2(x)
        x2 = self.maxpool(x2)
        x3 = self.conv1_3(x)
        x3 = self.maxpool(x3)
        x = torch.stack([x1, x2, x3], dim=1)
        x = x.view(batch,3,32,8)
        x1 = self.conv3(x)
        x2 = self.conv4(x)
        x3 = self.conv5(x)
        x1 = x1.view(batch,self.nk,-1)
        x2 = x2.view(batch,self.nk,-1)
        x3 = x3.view(batch,self.nk,-1)
        x1 = self.convconv3(x1)
        x1 = self.maxpool(x1)
        x2 = self.convconv4(x2)
        x2 = self.maxpool(x2)
        x3 = self.convconv5(x3)
        x3 = self.maxpool(x3)
        x = torch.cat([x1, x2, x3], dim=-1)
        x = self.concretconv1(x)
        x = self.maxpool(x)
        x = self.concretconv2(x)
        x = self.maxpool(x)
        x = x.view(batch,-1)
        x = self.fc1(x)
        x = self.activate(x)
        x = self.fc2(x)
        x = self.activate(x)
        x = x.view(origin_batch,8,10,2)
        x = x.sum(dim=2)
        res = [x.view(-1,2)]
        x = x.view(-1,2)
        x = self.MM2(x)
        x = x.view(origin_batch,-1)
        x = self.fc3(x)
        return F.log_softmax(x, dim=-1),res
    







class newMM(nn.Module):
    def __init__(self, C, D, device):
        super(newMM, self).__init__()
        
        # number of codeblocks that input is divided into,
        # for example, if input size is 14, and C is 7, then input vector is divided into 7 codeblocks, each of size 2.
        self.C = 10
        # size of each codeblock
        self.D = 6
        # thresholds table.
        self.register_buffer('S', torch.randn(30, 2, 15).to(device))
        self.register_buffer('H', torch.randn(45, 4096).to(device))
        self.T = nn.Parameter(torch.randn(30, 15).to(device))
        self.LUT = nn.Parameter(torch.randn(10, 4096, 2).to(device))

    def forward(self, x):
        x = x.view(-1, 8, 30, 2)
        x = torch.einsum('abcd,cdk->abck', x, self.S)
        x = x - self.T - 0.0001
        tanh = torch.tanh(x)
        sign = torch.sign(x)
        x = (sign - tanh).detach() + tanh 
        x = x.view(-1, 8, 10, 3, 15)
        x = x.reshape(-1, 8, 10, 45)
        x = x.view(-1,8, 10, 45)
        x = torch.einsum('bpcd,dk->bpck', x, self.H)

        one_hot = F.one_hot(torch.argmax(x, dim=-1), num_classes=4096).float()
        x_one_hot = torch.einsum('bpck,ckd->bpcd', one_hot, self.LUT)
        x = x_one_hot
        return x











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



class bigcnn_lut(nn.Module):

    def __init__(self,num_classes,device,seglength,convdim):
        super(bigcnn_lut, self).__init__()
        self.device = device
        self.MM1 = newMM(80,6,device)
        #self.fc1 = nn.Linear(16,num_classes).to(self.device)
        self.activate = nn.ReLU()
        self.MM2 = MM_final(16//2,2,num_classes,device)
        self.num_classes = num_classes
    
    def forward(self, x):
        x = x.to(self.device)   #[batch, 480]
        x = x.to(dtype=torch.float32)
        batch = x.shape[0]
        x = self.MM1(x)    #[batch, 80, 2]
        #print(x)
        x = x.view(batch,8,10,2)
        x = x.sum(dim=2)  #[batch, 8, 2]
        
        x = x.view(batch,16)#[batch, 16]
        x = self.MM2(x)
        x = x.view(batch,self.num_classes)#[batch, 3]
        #x = self.fc1(x)

        return F.log_softmax(x, dim=-1)












