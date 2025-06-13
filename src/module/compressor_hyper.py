import torch
import math
import time
from torch import nn
from src.module.layers import simple_resnet_graph_conv, mean_scale_estimation
from pytorch3d.ops import sample_farthest_points, knn_points
from src.common import index_points
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
class HyperCompressor(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        print('Entropy model built with conditional hyperprior.')
        self.num_groups = cfg['model']['compressor_kwargs']['num_groups']
        self.group_size = cfg['model']['compressor_kwargs']['group_size']
        self.k_list = cfg['model']['compressor_kwargs']['k_list']
        self.num_points = cfg['model']['compressor_kwargs']['num_points']
        #   downsample and learn feature
        self.ha = nn.ModuleList([
            simple_resnet_graph_conv(
                self.k_list[0], 
                cfg['model']['encoder_kwargs']['out_channels'],
                cfg['model']['compressor_kwargs']['hidden_dim']),
            simple_resnet_graph_conv(
                self.k_list[1], 
                cfg['model']['compressor_kwargs']['hidden_dim'],
                cfg['model']['compressor_kwargs']['hidden_dim']),
            simple_resnet_graph_conv(
                self.k_list[2], 
                cfg['model']['compressor_kwargs']['hidden_dim'],
                cfg['model']['encoder_kwargs']['out_channels'])
        ])
        #   estimate point-wise means and scales
        self.hs = mean_scale_estimation(
            cfg['model']['encoder_kwargs']['out_channels'],
            cfg['model']['compressor_kwargs']['hidden_dim'],
            2*cfg['model']['encoder_kwargs']['out_channels']
        )
        
        self.entropy_bottleneck = EntropyBottleneck(cfg['model']['encoder_kwargs']['out_channels'])
        self.gaussian_conditional = GaussianConditional(None)
        
    def forward(self, g_coords, g_feats, points_num):
        '''
            Inputs:
                g_coords: [B,M,3]
                g_feats: [B,M,C]
            Returns:
                (soft) quantized g_feats, g_bpp, h_bpp
        '''
        input_coords = g_coords
        input_feats = g_feats
        num_g_points = g_coords.shape[1]
        #   progressively downsampling and aggregate features 
        for ds_block_idx in range(len(self.k_list)):
            #sampled_coords, sampled_idx = sample_farthest_points(input_coords, K=self.num_points[ds_block_idx])
            sampled_coords, sampled_idx = sample_farthest_points(input_coords, K = max(self.num_points[ds_block_idx], num_g_points//(6*(ds_block_idx + 1))))
            sampled_feats = index_points(input_feats, sampled_idx)
            _, idxs, _ = knn_points(sampled_coords, input_coords, K = self.k_list[ds_block_idx])
            h_latents = self.ha[ds_block_idx](input_feats, sampled_feats, idxs)
            input_coords = sampled_coords
            input_feats = h_latents
        h_points = sampled_coords#[1,256,3]
        h_latents_hat, h_latents_likelihoods = self.entropy_bottleneck(h_latents.transpose(1,2))#[1,16,1]
        
        h_bits_size = (torch.log(h_latents_likelihoods).sum()) / (-math.log(2))
        h_bpp = h_bits_size / points_num
        g_scales_hat, g_means_hat = self.hs(g_coords, h_points, h_latents_hat.transpose(1,2)).transpose(1,2).chunk(2,1)#[1,8,3000]
        g_feats_hat, g_feats_likelihoods = self.gaussian_conditional(g_feats.transpose(1,2), g_scales_hat, g_means_hat)
        
        g_bits_size = (torch.log(g_feats_likelihoods).sum()) / (-math.log(2))
        g_bpp = g_bits_size / points_num
        return g_feats_hat.transpose(1,2), g_bpp, h_bpp

    def compress(self, g_coords, g_feats, points_num):
        '''
            Inputs:
                g_coords: [B,M,3]
                g_feats: [B,M,C]
            Returns:
                g_latents_str, h_latents_str, g_bpp, h_bpp, enc_time
        '''
        input_coords = g_coords
        input_feats = g_feats
        num_g_points = g_coords.shape[1]
        '''
            transform latents to hyper-latents.
        '''
        t0 = time.time()
        for ds_block_idx in range(len(self.k_list)):
            #sampled_coords, sampled_idx = sample_farthest_points(input_coords, K=self.num_points[ds_block_idx])
            sampled_coords, sampled_idx = sample_farthest_points(input_coords, K = max(self.num_points[ds_block_idx], num_g_points//(6*(ds_block_idx + 1))))
            sampled_feats = index_points(input_feats, sampled_idx)
            _, idxs, _ = knn_points(sampled_coords, input_coords, K = self.k_list[ds_block_idx])
            h_latents = self.ha[ds_block_idx](input_feats, sampled_feats, idxs)
            input_coords = sampled_coords
            input_feats = h_latents
        h_points = sampled_coords#[1,256,3]
        h_size = h_latents.transpose(1,2).size()[2:]
        t_encoder = time.time() - t0 #  the time spent on extracting hyper-latents
        '''
            encoding hyper-latents with factorized entropy model.
        '''
        t0 = time.time()
        h_latents_str = self.entropy_bottleneck.compress(h_latents.transpose(1,2))
        t_e_eb = time.time() - t0 # the time spent on encoding hyper-latents.
        '''
            decoding hyper-latents with factorized entropy model.
        '''
        t0 = time.time()
        h_latents_hat = self.entropy_bottleneck.decompress(h_latents_str, h_size)
        t_d_eb = time.time() - t0 # the time spent on decoding hyper-latents.
        h_bpp = (sum(len(s) for s in h_latents_str) * 8) / points_num

        '''
            estimating entropy parameters
        '''
        t0 = time.time()
        g_scales_hat, g_means_hat = self.hs(g_coords, h_points, h_latents_hat.transpose(1,2)).transpose(1,2).chunk(2,1)
        t_pred_g = time.time() - t0 # the time spent on predicting entropy parameters for g_feat.
        '''
            compress latents with Gaussian entropy model
        '''
        t0 = time.time()
        indexes = self.gaussian_conditional.build_indexes(g_scales_hat)
        g_latents_str = self.gaussian_conditional.compress(g_feats.transpose(1,2), indexes, means=g_means_hat)
        t_e_g = time.time() - t0 # the time spent on encoding latent features.

        g_bpp = (sum(len(s) for s in g_latents_str) * 8.0) / points_num

        enc_time = t_encoder + t_e_eb + t_d_eb + t_pred_g + t_e_g
        
        return g_latents_str, h_latents_str, g_bpp, h_bpp, enc_time, h_size
    
    def decompress(self, g_coords, g_latents_str, h_latents_str, points_num, eb_size):
        '''
            Inputs:
                g_coords: [B,M,3]
                g_latents_str
                h_latents_str
            Returns:
                decoded g_feats, decoding time
        '''
        input_coords = g_coords
        h_size = eb_size
        num_g_points = g_coords.shape[1]
        '''
            transform latents to hyper-latents.
        '''
        t0 = time.time()
        for ds_block_idx in range(len(self.k_list)):
            #sampled_coords, sampled_idx = sample_farthest_points(input_coords, K=self.num_points[ds_block_idx])
            sampled_coords, sampled_idx = sample_farthest_points(input_coords, K = max(self.num_points[ds_block_idx], num_g_points//(6*(ds_block_idx + 1))))
            input_coords = sampled_coords
        h_points = sampled_coords   #[1,256,3]
        
        t_fps = time.time() - t0 #  the time spent on getting h_points

        '''
            decoding hyper-latents with factorized entropy model.
        '''
        t0 = time.time()
        h_latents_hat = self.entropy_bottleneck.decompress(h_latents_str, h_size)
        t_d_eb = time.time() - t0 # the time spent on decoding hyper-latents from string.
        '''
            estimating entropy parameters
        '''
        t0 = time.time()
        g_scales_hat, g_means_hat = self.hs(g_coords, h_points, h_latents_hat.transpose(1,2)).transpose(1,2).chunk(2,1)
        t_pred_g = time.time() - t0 # the time spent on predicting entropy parameters for g_feat.
        '''
            compress latents with Gaussian entropy model
        '''
        t0 = time.time()
        indexes = self.gaussian_conditional.build_indexes(g_scales_hat)
        g_feats_hat = self.gaussian_conditional.decompress(g_latents_str, indexes, means=g_means_hat)
        t_d_g = time.time() - t0 # the time spent on decoding latent features.
        
        dec_time = t_pred_g + t_d_eb + t_d_g + t_fps
        
        return g_feats_hat.transpose(1,2), dec_time