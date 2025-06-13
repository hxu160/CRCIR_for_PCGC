import torch
from torch.nn import L1Loss
from src.common import golden_interpolation
from src.training_AE import BaseAETrainer
from pytorch3d.ops import knn_points
from pytorch3d.loss import chamfer_distance

class AE_Trainer(BaseAETrainer):
    ''' Trainer object for the CRCIR optimized with distortion term only.

    Args:
        encoder (nn.Module): g_a(.)
        decoder (nn.Module): g_s(.)
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
    '''

    def __init__(self, encoder, decoder, optimizer, cfg, device=None):
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.device = device
        self.error_metric = L1Loss()

    
    def eval_step(self, data, upsampling_ratio):
        
        self.encoder.eval()
        self.decoder.eval()
        device = self.device
        gt_pcd = data[0].to(device).unsqueeze(0)
        sparse_pcd = data[1].to(device).unsqueeze(0)
        B = sparse_pcd.shape[0]
        K = sparse_pcd.shape[1]
        C = upsampling_ratio
        eval_dict = {}
        
        with torch.no_grad():
            
            interpolated_pcd = golden_interpolation(sparse_pcd, C)
            reshaped_interpolated_pcd = interpolated_pcd.view(B, -1, 3)
            cd_loss, _ = chamfer_distance(reshaped_interpolated_pcd, gt_pcd)
            eval_dict['cd_loss_ref'] = cd_loss.item()
            _, _, nns = knn_points(reshaped_interpolated_pcd, gt_pcd, K = 1, return_nn=True)
            residuals = nns.squeeze(-2) - reshaped_interpolated_pcd
            residuals_cluster = residuals.view(B, K, C, 3)
            local_centers = torch.mean(interpolated_pcd, dim=2)
            central_diff = interpolated_pcd - local_centers.unsqueeze(2)
            feats = self.encoder(residuals_cluster, central_diff)
            
            D = feats.shape[2]
            pred_residuals = self.decoder(central_diff.view(B, -1, 3), feats.unsqueeze(2).repeat(1, 1, C, 1).view(B, -1, D))
            distortion = self.error_metric(residuals_cluster.view(B, -1, 3), pred_residuals)
            eval_dict['pred_loss'] = distortion.item()
            pred_points = reshaped_interpolated_pcd + pred_residuals
            cd_loss, _ = chamfer_distance(pred_points, gt_pcd)
            eval_dict['cd_loss'] = cd_loss.item()
        return eval_dict
    
    
    
    def train_step(self, data, C = 34):
        ''' Performs a training step.

         Args:
            data (tuple): 
            (
                gt_pcd
                sparse_pcd (precomputed via FPS)
            )
        '''
        self.encoder.train()
        self.decoder.train()
        self.optimizer.zero_grad()
        loss = self.fit_AE(data, C)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    
    def fit_AE(self, batch, C = 34):
        device = self.device
        gt_pcd = batch[0].to(device)
        sparse_pcd = batch[1].to(device)
        B = sparse_pcd.shape[0]
        K = sparse_pcd.shape[1]
        interpolated_pcd = golden_interpolation(sparse_pcd, C)
        reshaped_interpolated_pcd = interpolated_pcd.view(B, -1, 3)
        _, _, nns = knn_points(reshaped_interpolated_pcd, gt_pcd, K = 1, return_nn=True)
        residuals = nns.squeeze(-2) - reshaped_interpolated_pcd
        residuals_cluster = residuals.view(B, K, C, 3)
        local_centers = torch.mean(interpolated_pcd, dim=2)
        central_diff = interpolated_pcd - local_centers.unsqueeze(2)
        feats = self.encoder(residuals_cluster, central_diff)
        
        D = feats.shape[2]
        pred_residuals = self.decoder(central_diff.view(B, -1, 3), feats.unsqueeze(2).repeat(1, 1, C, 1).view(B, -1, D))
        loss = self.error_metric(residuals_cluster.view(B, -1, 3), pred_residuals)
        return loss
    
    

    
    

        
        
    
    
