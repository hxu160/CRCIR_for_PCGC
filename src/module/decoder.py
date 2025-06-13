import torch.nn as nn
import torch.nn.functional as F
from src.module.layers import ResnetBlockFC

class LocalDecoder(nn.Module):
    ''' Decoder.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
    '''

    def __init__(self, dim=3, c_dim=128, out_dim = 1,
                 hidden_size=256, n_blocks=5, leaky=False):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])


        self.fc_p = nn.Linear(dim, hidden_size)

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        self.fc_out = nn.Linear(hidden_size, out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, c):
        net = self.fc_p(p)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                # fc_c: c_dim -> hidden_size
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))

        return out