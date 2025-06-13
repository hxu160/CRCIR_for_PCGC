import torch
import math
import time
from torch import nn
from compressai.entropy_models import EntropyBottleneck
class FFPCompressor(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        print('Entropy model built with unconditional fully factorized prior.')
        self.entropy_bottleneck = EntropyBottleneck(cfg['model']['encoder_kwargs']['out_channels'])
        
    def forward(self, g_coords, g_feats, points_num):
        '''
            Inputs:
                g_coords: [B,M,3]
                g_feats: [B,M,C]
            Returns:
                (soft) quantized g_feats, bpp
        '''
        g_feats_hat, g_feats_likelihoods = self.entropy_bottleneck(g_feats.transpose(1,2))
        g_bits_size = (torch.log(g_feats_likelihoods).sum()) / (-math.log(2))
        g_bpp = g_bits_size / points_num
        
        return g_feats_hat.transpose(1,2), g_bpp
        
    def compress(self, g_coords, g_feats, points_num):
        '''
            Inputs:
                g_coords: [B,M,3]
                g_feats: [B,M,C]
            Returns:
                (hard) quantized g_feats, bpp, encoding time, decoding time
        '''       
        g_size = g_feats.transpose(1,2).size()[2:]
        #   encoding with factorized entropy model
        t0 = time.time()
        g_latents_str = self.entropy_bottleneck.compress(g_feats.transpose(1,2))
        t_e_eb = time.time() - t0
        
        g_bpp = (sum(len(s) for s in g_latents_str) * 8.0) / points_num
        #   feature transform, means and scales estimation, compress
        enc_time = t_e_eb
        #   means and scales estimation, compress
        return g_latents_str, g_bpp, enc_time, g_size
        
        
    def decompress(self, g_coords, g_latents_str, points_num, eb_size):
        '''
            Inputs:
                g_coords: [B,M,3]
                g_feats: [B,M,C]
            Returns:
                (hard) quantized g_feats, bpp, encoding time, decoding time
        '''       
        g_size = eb_size
        
        t0 = time.time()
        g_feats_hat = self.entropy_bottleneck.decompress(g_latents_str, g_size)
        t_d_eb = time.time() - t0
        
        dec_time = t_d_eb 
        return g_feats_hat.transpose(1,2), dec_time