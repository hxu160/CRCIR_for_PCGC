import torch
import logging
class ResNet_Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(out_channels, out_channels, 1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU()
        )
        if in_channels == out_channels:
            self.shortcut = None
        else:
            self.shortcut = torch.nn.Conv2d(in_channels, out_channels, 1)
    
    def forward(self, x):
        if self.shortcut is not None:
            return self.fc(x) + self.shortcut(x)
        else:
            return self.fc(x) + x

class ResNet_Block_1D(torch.nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, 1),
            torch.nn.BatchNorm1d(out_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(out_channels, out_channels, 1),
            torch.nn.BatchNorm1d(out_channels)
            #torch.nn.LeakyReLU()
        )
        if in_channels == out_channels:
            self.shortcut = None
        else:
            self.shortcut = torch.nn.Conv1d(in_channels, out_channels, 1)
    
    def forward(self, x):
        if self.shortcut is not None:
            return self.fc(x) + self.shortcut(x)
        else:
            return self.fc(x) + x

class InterpAttentionNet(torch.nn.Module):

    def __init__(self, in_channels, latent_size, out_channels, K=16):
        super().__init__()
        
        logging.info(f"InterpNet - Simple - K={K}")
        self.resnet_block1 = ResNet_Block(in_channels+3, latent_size)
        self.resnet_block2 = ResNet_Block(latent_size, latent_size)
        self.resnet_block_1D = ResNet_Block_1D(latent_size, out_channels)

        self.fc_query = torch.nn.Conv2d(latent_size, 1, 1)
        self.fc_value = torch.nn.Conv2d(latent_size, latent_size,1)

        self.k = K

    def forward(self, feat, coords):
        '''
            grouped_features: [B, M, K, C]
            grouped_neighbors: [B, M, K, 3]
        '''
        pos = coords.permute(0, 3, 1, 2).contiguous()#B*3*M*K         
        x = feat.permute(0, 3, 1, 2).contiguous()#B*C*M*K
        x = torch.cat([x,pos], dim=1)#B*(C+3)*M*K
        x = self.resnet_block1(x)
        x = self.resnet_block2(x)
        #   B*C*M*K -> B*1*M*K -> B*M*K
        query = self.fc_query(x).squeeze(dim=1)#B*M*K
        attention = torch.nn.functional.softmax(query, dim=-1)

        value = self.fc_value(x)#B*C*M*K -> B*C*M*K
        #   B*M*1*K multiplies B*M*K*C -> B*M*1*C -> B*M*C
        x = torch.matmul(attention.unsqueeze(-2), value.permute(0,2,3,1)).squeeze(-2)
        x = x.transpose(1,2) #B*C*M
        
        x = self.resnet_block_1D(x) #B*C*M

        return x.transpose(1,2) #B*M*C