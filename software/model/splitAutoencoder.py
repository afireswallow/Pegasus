import torch
import torch.nn as nn
import torch.nn.functional as F

class original_splitautoencoder(nn.Module):
    def __init__(self, input_size, len_vocab, ipd_vocab,
                 len_embedding_bits, ipd_embedding_bits, 
                 ebdin, hs1, hs2, hs3, hs4,
                 device):
        super(original_splitautoencoder, self).__init__()
        self.ebdin = ebdin
        self.hs1 = hs1
        self.hs2 = hs2
        self.hs3 = hs3
        self.hs4 = hs4
        
        # Embedding layers
        self.len_embedding = nn.Embedding(len_vocab, len_embedding_bits).to(device)
        self.ipd_embedding = nn.Embedding(ipd_vocab, ipd_embedding_bits).to(device)
        
        # Feature extraction layer
        self.fc1 = nn.Linear((len_embedding_bits + ipd_embedding_bits), ebdin).to(device)
        self.activate = nn.ReLU()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Unflatten(1, (1, ebdin*8)).to(device),
            nn.Conv1d(
                in_channels=1,
                out_channels=hs1 // (ebdin*8),
                kernel_size=3,
                padding=1
            ).to(device), 
            self.activate,
            nn.Flatten().to(device),
            nn.Linear(hs1, hs2).to(device),
            self.activate,
            nn.Linear(hs2, hs3).to(device),
            self.activate,
            nn.Linear(hs3, hs4).to(device),
            self.activate,
            nn.Linear(hs4, 8).to(device),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(8, hs4).to(device),
            self.activate,  
            nn.Linear(hs4, hs3).to(device),
            self.activate,
            nn.Linear(hs3, hs2).to(device),
            self.activate,
            nn.Linear(hs2, hs1).to(device),
            self.activate,
            nn.Linear(hs1, ebdin*8).to(device),
        )
        
        self.device = device
    
    def _create_padded_feature(self, x, start_idx, length, total_length):
        # 创建全0张量
        batch_size = x.size(0)
        padded = torch.zeros(batch_size, total_length).to(self.device)
        # 将特征放在指定位置
        padded[:, start_idx:start_idx+length] = x
        return padded
        
    def forward(self, x):
        # Extract features
        x = x.view(-1, 8, 2)
        len_x = x[:,:,0].to(self.device).long()
        ipd_x = x[:,:,1].to(self.device).long()
        
        # Apply embeddings
        len_x = self.len_embedding(len_x)
        ipd_x = self.ipd_embedding(ipd_x)
        batch = x.shape[0]
        
        # Concatenate features
        x = torch.cat((len_x, ipd_x), dim=-1)
        
        # Feature extraction
        shared_feature = self.fc1(x)
        shared_feature = shared_feature.view(batch, -1)  # [batch, 32]
        
        # 初始化重建特征
        final_decoded = torch.zeros_like(shared_feature).to(self.device)
        encoded_slices = []
        
        # 处理5个6维切片和1个2维切片
        start_indices = [0, 6, 12, 18, 24, 30]
        lengths = [6, 6, 6, 6, 6, 2]
        
        for i in range(6):
            # 获取切片
            start_idx = start_indices[i]
            length = lengths[i]
            slice_feature = shared_feature[:, start_idx:start_idx+length]
            
            # 创建填充后的特征
            padded_feature = self._create_padded_feature(slice_feature, start_idx, length, 32)
            
            # 编码和解码
            encoded = self.encoder(padded_feature)
            # encoded_slices.append(encoded)
            decoded = self.decoder(encoded)
            
            # 累加解码结果
            final_decoded += decoded
        
        # encoded = torch.stack(encoded_slices, dim=1)  # [batch, 6, 8]
        # print(shared_feature.shape)
        return final_decoded, [shared_feature]














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






class MM_splitautoencoder(nn.Module):
    def __init__(self, input_size, len_vocab, ipd_vocab,
                 len_embedding_bits, ipd_embedding_bits, 
                 ebdin, hs1, hs2, hs3, hs4,
                 device):
        super(MM_splitautoencoder, self).__init__()
        self.ebdin = ebdin
        self.hs1 = hs1
        self.hs2 = hs2
        self.hs3 = hs3
        self.hs4 = hs4
        
        # Embedding layers
        self.len_embedding = nn.Embedding(len_vocab, len_embedding_bits).to(device)
        self.ipd_embedding = nn.Embedding(ipd_vocab, ipd_embedding_bits).to(device)
        
        # Feature extraction layer
        self.fc1 = nn.Linear((len_embedding_bits + ipd_embedding_bits), ebdin).to(device)
        self.activate = nn.ReLU()
        
        self.MM = MM(32//2, 2, device)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Unflatten(1, (1, ebdin*8)).to(device),
            nn.Conv1d(
                in_channels=1,
                out_channels=hs1 // (ebdin*8),
                kernel_size=3,
                padding=1
            ).to(device), 
            self.activate,
            nn.Flatten().to(device),
            nn.Linear(hs1, hs2).to(device),
            self.activate,
            nn.Linear(hs2, hs3).to(device),
            self.activate,
            nn.Linear(hs3, hs4).to(device),
            self.activate,
            nn.Linear(hs4, 8).to(device),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(8, hs4).to(device),
            self.activate,  
            nn.Linear(hs4, hs3).to(device),
            self.activate,
            nn.Linear(hs3, hs2).to(device),
            self.activate,
            nn.Linear(hs2, hs1).to(device),
            self.activate,
            nn.Linear(hs1, ebdin*8).to(device),
        )
        
        self.device = device
    
    def _create_padded_feature(self, x, start_idx, length, total_length):
        # 创建全0张量
        batch_size = x.size(0)
        padded = torch.zeros(batch_size, total_length).to(self.device)
        # 将特征放在指定位置
        padded[:, start_idx:start_idx+length] = x
        return padded
        
    def forward(self, x):
        # Extract features
        x = x.view(-1, 8, 2)
        len_x = x[:,:,0].to(self.device).long()
        ipd_x = x[:,:,1].to(self.device).long()
        
        # Apply embeddings
        len_x = self.len_embedding(len_x)
        ipd_x = self.ipd_embedding(ipd_x)
        batch = x.shape[0]
        
        # Concatenate features
        x = torch.cat((len_x, ipd_x), dim=-1)
        
        # Feature extraction
        shared_feature = self.fc1(x)
        shared_feature = shared_feature.view(batch, -1)  # [batch, 32]
        shared_feature = self.MM(shared_feature)
        
        # 初始化重建特征
        final_decoded = torch.zeros_like(shared_feature).to(self.device)
        encoded_slices = []
        
        # 处理5个6维切片和1个2维切片
        start_indices = [0, 6, 12, 18, 24, 30]
        lengths = [6, 6, 6, 6, 6, 2]
        
        for i in range(6):
            # 获取切片
            start_idx = start_indices[i]
            length = lengths[i]
            slice_feature = shared_feature[:, start_idx:start_idx+length]
            
            # 创建填充后的特征
            padded_feature = self._create_padded_feature(slice_feature, start_idx, length, 32)
            
            # 编码和解码
            encoded = self.encoder(padded_feature)
            # encoded_slices.append(encoded)
            decoded = self.decoder(encoded)
            
            # 累加解码结果
            final_decoded += decoded
        
        # encoded = torch.stack(encoded_slices, dim=1)  # [batch, 6, 8]
        
        return final_decoded, [shared_feature]
    



class MM_splitautoencoder_template(nn.Module):
    def __init__(self, input_size, len_vocab, ipd_vocab,
                 len_embedding_bits, ipd_embedding_bits, 
                 ebdin, hs1, hs2, hs3, hs4,
                 device):
        super(MM_splitautoencoder_template, self).__init__()
        self.ebdin = ebdin
        self.hs1 = hs1
        self.hs2 = hs2
        self.hs3 = hs3
        self.hs4 = hs4
        
        # Embedding layers
        self.len_embedding = nn.Embedding(len_vocab, len_embedding_bits).to(device)
        self.ipd_embedding = nn.Embedding(ipd_vocab, ipd_embedding_bits).to(device)
        
        # Feature extraction layer
        self.fc1 = nn.Linear((len_embedding_bits + ipd_embedding_bits), ebdin).to(device)
        self.activate = nn.ReLU()
        
        self.MM = MM(32//2, 2, device)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Unflatten(1, (1, ebdin*8)).to(device),
            nn.Conv1d(
                in_channels=1,
                out_channels=hs1 // (ebdin*8),
                kernel_size=3,
                padding=1
            ).to(device), 
            self.activate,
            nn.Flatten().to(device),
            nn.Linear(hs1, hs2).to(device),
            self.activate,
            nn.Linear(hs2, hs3).to(device),
            self.activate,
            nn.Linear(hs3, hs4).to(device),
            self.activate,
            nn.Linear(hs4, 8).to(device),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(8, hs4).to(device),
            self.activate,  
            nn.Linear(hs4, hs3).to(device),
            self.activate,
            nn.Linear(hs3, hs2).to(device),
            self.activate,
            nn.Linear(hs2, hs1).to(device),
            self.activate,
            nn.Linear(hs1, ebdin*8).to(device),
        )
        
        self.device = device
    
    def _create_padded_feature(self, x, start_idx, length, total_length):
        # 创建全0张量
        batch_size = x.size(0)
        padded = torch.zeros(batch_size, total_length).to(self.device)
        # 将特征放在指定位置
        padded[:, start_idx:start_idx+length] = x
        return padded
        
    def forward(self, x):
        encoded = self.encoder(x)
        # encoded_slices.append(encoded)
        decoded = self.decoder(encoded)
        return decoded
    





class newMM(nn.Module):
    def __init__(self, C, D, device):
        super(newMM, self).__init__()
        
        # number of codeblocks that input is divided into,
        # for example, if input size is 14, and C is 7, then input vector is divided into 7 codeblocks, each of size 2.
        self.C = C
        # size of each codeblock
        self.D = D
        # thresholds table.
        # self.register_buffer('S', torch.randn(30, 2, 15).to(device))
        # self.register_buffer('H', torch.randn(45, 4096).to(device))
        # self.T = nn.Parameter(torch.randn(30, 15).to(device))
        # self.LUT = nn.Parameter(torch.randn(80, 4096, 2).to(device))
        self.register_buffer('S', torch.randn(15, 2, 15).to(device))
        self.register_buffer('H', torch.randn(45, 4096).to(device))
        self.T = nn.Parameter(torch.randn(15, 15).to(device))
        self.LUT = nn.Parameter(torch.randn(5, 4096, 32).to(device))

    def forward(self, x):
        #x = x.view(-1, 8, 30, 2)
        x = x.view(-1,1,15,2)
        #print(x.shape)
        #print(self.S.shape)
        x = torch.einsum('abcd,cdk->abck', x, self.S)
        x = x - self.T - 0.0001
        tanh = torch.tanh(x)
        sign = torch.sign(x)
        x = (sign - tanh).detach() + tanh    #(batch, 8, 30, 15)
        #print(x)
        #x = x.view(-1, 8, 10, 3, 15)
        x = x.view(-1,1,5,3,15)
        #x = x.reshape(-1, 8, 10, 45)
        x = x.reshape(-1,1,5,45)
        #x = x.view(-1, 80, 45)
        x = x.view(-1,5,45)
        x = torch.einsum('bcd,dk->bck', x, self.H)

        one_hot = F.one_hot(torch.argmax(x, dim=-1), num_classes=4096).float()
        # print(one_hot.shape)
        # indices = torch.nonzero(one_hot, as_tuple=True)
        # selected_indices = indices[2].reshape(80)  # 形状变为 [80]
        # print(selected_indices)
        # print(self.LUT.shape)
        # print("---------------------------------------------")

        x_one_hot = torch.einsum('bck,ckd->bcd', one_hot, self.LUT)
        x = x_one_hot
        #print(x.shape)
        #print(one_hot)
        #print(one_hot.shape)
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











class table_splitautoencoder_template(nn.Module):
    def __init__(self, input_size, len_vocab, ipd_vocab,
                 len_embedding_bits, ipd_embedding_bits, 
                 ebdin, hs1, hs2, hs3, hs4,
                 device):
        super(table_splitautoencoder_template, self).__init__()
        self.ebdin = ebdin
        self.hs1 = hs1
        self.hs2 = hs2
        self.hs3 = hs3
        self.hs4 = hs4
        
        self.lenebdLUT = nn.Parameter(torch.randn(len_vocab, ebdin).to(device))
        self.ipdebdLUT = nn.Parameter(torch.randn(ipd_vocab, ebdin).to(device))

        self.MM_1 = newMM(C=5, D=6, device=device)
        self.MM_2 = MM_final(C=1, D=2, output_size=32, device=device)
        
        self.device = device
    
        
    def forward(self, x):
        len_x = x[:,:,0].to(self.device)
        ipd_x = x[:,:,1].to(self.device)
        len_x = len_x.to(self.device).long()
        ipd_x = ipd_x.to(self.device).long()
        len_x = self.lenebdLUT[len_x]
        ipd_x = self.ipdebdLUT[ipd_x]
        x = ipd_x+len_x
        batch = x.shape[0]
        x = x.view(-1,32)
        ebd = x
        #print(x.shape)
        x1 = x[:,0:30]
        x2 = x[:,30:32]
        x1 = self.MM_1(x1)
        x2 = self.MM_2(x2)
        x2 = x2.view(batch,1,32)
        #print(x1.shape)
        #print(x2.shape)
        reconstruct = torch.cat([x1, x2], dim=1)
        reconstruct = reconstruct.sum(dim=1)
        #print(x.shape)
        return reconstruct,[ebd]