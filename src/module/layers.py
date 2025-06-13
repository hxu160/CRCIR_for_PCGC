import math
import torch
import torch.nn as nn
from pytorch3d.ops import knn_gather
from pytorch3d.ops import knn_points, knn_gather

class mean_scale_estimation(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim) -> None:
        super().__init__()
        self.block_1 = ResnetBlockFC(in_dim, hidden_dim)
        self.block_2 = ResnetBlockFC(hidden_dim, out_dim)
    
    def forward(self, pi, ps, fs, scales = 1000):
        dis, idx, _ = knn_points(pi, ps, K=4, return_nn=False)
        #print(fs.shape)
        #print(idx.shape)
        grouped_feat = knn_gather(fs, idx)#B*N*K*C
        grouped_feat = self.block_1(grouped_feat)#B*N*K*C
        soft_max = torch.nn.Softmax(dim=2)
        weights = soft_max(-1*scales*dis).unsqueeze(2)#B*N*1*K
        fused_feat = torch.matmul(weights, grouped_feat).squeeze(2)#B*N*C
        return self.block_2(fused_feat)

class simple_resnet_graph_conv(nn.Module):
    def __init__(self, k, in_dim, out_dim):
        super().__init__()
        self.k = k
        self.conv = ResnetBlockFC(2*in_dim, out_dim)
        
    def forward(self, x, x_subset, knn_index):
        #B,N,C = x.shape
        #B,M,C = sub_x.shape
        grouped_feat = knn_gather(x, knn_index)#B*M*K*C
        repeated_feat = x_subset.unsqueeze(2).repeat(1, 1, self.k, 1)
        feat = torch.cat([grouped_feat - repeated_feat, repeated_feat], dim=3)#B*M*K*(2*C)
        return torch.max(self.conv(feat), dim=-2)[0]#B*C*M

class simple_resnet_graph_conv2(nn.Module):
    def __init__(self, k, in_dim, out_dim):
        super().__init__()
        self.k = k
        self.block_0 = ResnetBlockFC(2*in_dim, 2*out_dim)
        self.block_1 = ResnetBlockFC(2*out_dim, out_dim)
    def forward(self, x, x_subset, knn_index):
        #B,N,C = x.shape
        #B,M,C = sub_x.shape
        grouped_feat = knn_gather(x, knn_index)#B*M*K*C
        repeated_feat = x_subset.unsqueeze(2).repeat(1, 1, self.k, 1)
        feat = torch.cat([grouped_feat - repeated_feat, repeated_feat], dim=3)#B*M*K*(2*C)
        pooled_feat = torch.max(self.block_0(feat), dim=-2)[0] #B*M*C
        return self.block_1(pooled_feat)

class simple_graph_conv(nn.Module):
    def __init__(self, k, in_dim, out_dim, flatten = False):
        super().__init__()
        self.k = k
        if flatten:
            self.conv = nn.Sequential(
                nn.Conv2d(2*in_dim, out_dim, 1)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(2*in_dim, out_dim, 1),
                nn.ReLU()
            )
    def forward(self, x, x_subset, knn_index):
        #B,N,C = x.shape
        #B,M,C = sub_x.shape
        grouped_feat = knn_gather(x, knn_index)#B*M*K*C
        repeated_feat = x_subset.unsqueeze(2).repeat(1, 1, self.k, 1)
        feat = torch.cat([grouped_feat - repeated_feat, repeated_feat], dim=3).permute(0, 3, 1, 2).contiguous()
        return torch.max(self.conv(feat), dim=-1)[0]#B*C*M
    
# Resnet Blocks
class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class ResnetBlockFC_1D_Conv(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Conv1d(size_in, size_h, 1)
        self.fc_1 = nn.Conv1d(size_h, size_out, 1)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in, size_out, 1, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx