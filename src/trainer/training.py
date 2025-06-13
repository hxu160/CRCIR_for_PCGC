import os
import torch
import numpy as np
import open3d as o3d
import time
from torch.nn import L1Loss
from src.common import golden_interpolation, denorm_points, norm_01, golden_interpolation2, write_ply_ascii_geo
from src.training import BaseTrainer
from pytorch3d.ops import knn_points, knn_gather
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_farthest_points

class Trainer(BaseTrainer):
    ''' Trainer object for the CRCIR.

    Args:
        model (nn.Module): CRCIR Network [encoder, decoder, compressor]
        optimizer (optimizer): pytorch optimizer object [optimizer, aux_optimizer]
        config dictionary
        device (device): pytorch device
    '''

    def __init__(self, encoder, decoder, compressor, optimizer, aux_optimizer, cfg, device=None):
        self.encoder = encoder
        self.decoder = decoder
        self.compressor = compressor
        self.optimizer = optimizer
        self.aux_optimizer = aux_optimizer
        self.device = device
        self.lambd = cfg['training']['lambda']
        self.codec = cfg['model']['compressor']
    
    def eval_step(self, data, upsampling_ratio):
        
        self.encoder.eval()
        self.decoder.eval()
        self.compressor.eval()
        device = self.device
        gt_pcd = data[0].to(device).unsqueeze(0)
        sparse_pcd = data[1].to(device).unsqueeze(0)
        B = sparse_pcd.shape[0]
        K = sparse_pcd.shape[1]
        C = upsampling_ratio
        eval_dict = {}
        error_metric = L1Loss()
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
            if self.codec == 'ffp':
                feats, g_bpp = self.compressor(g_coords = sparse_pcd, g_feats = feats, points_num = B*100000)
                bpps = g_bpp.mean()
                eval_dict['g_bpp'] = g_bpp.mean().item()
            else:
                feats, g_bpp, h_bpp = self.compressor(g_coords = sparse_pcd, g_feats = feats, points_num = B*100000)
                bpps = g_bpp.mean() + h_bpp.mean()
                eval_dict['g_bpp'] = g_bpp.mean().item()
                eval_dict['h_bpp'] = h_bpp.mean().item()
            D = feats.shape[2]
            pred_residuals = self.decoder(central_diff.view(B, -1, 3), feats.unsqueeze(2).repeat(1, 1, C, 1).view(B, -1, D))
            distortion = error_metric(residuals_cluster.view(B, -1, 3), pred_residuals)
            eval_dict['distortion'] = distortion.item()
            loss = distortion + self.lambd * bpps
            eval_dict['pred_loss'] = loss.item()
            pred_points = reshaped_interpolated_pcd + pred_residuals
            cd_loss, _ = chamfer_distance(pred_points, gt_pcd)
            eval_dict['cd_loss'] = cd_loss.item()
        return eval_dict
    
    def cal_cd_in_original_scale(self, pred_points, gt_pcd, shift, bbox_max, bbox_min):
        '''
            pred_points: [1, N, 3], dynamic range [-1,1]
            gt_pcd: [1, N, 3], dynamic range: the original scale
            
        '''
        
        pred_pcd_o = denorm_points(pred_points.squeeze(0), shift, bbox_max, bbox_min)
        cd_loss, _ = chamfer_distance(pred_pcd_o.unsqueeze(0), gt_pcd)
        return cd_loss.item(), pred_pcd_o
    
    def compress_step(self, gt_pcd_o3d, integer_points_dir, draco_drc_dir, draco_ply_dir, prefix, qp, ds_ratio, draco_path):
        eval_dict = {}
        points = torch.from_numpy(np.asarray(gt_pcd_o3d.points)).to(torch.float32).to(self.device)
        num_points = len(gt_pcd_o3d.points)
        sampled_num_points = num_points // ds_ratio
        xyzs, shift, max_coord, min_coord = norm_01(points)
        cl = 10
        with torch.no_grad():
            points_t = xyzs.unsqueeze(0)
            '''
                step 1: downsampling
            '''
            t0 = time.time()
            points_s, points_idx = sample_farthest_points(points_t, K = sampled_num_points)
            time_fps = time.time() - t0
            eval_dict['time_fps'] = time_fps
            points_s_np = points_s.squeeze(0).cpu().numpy()
            sub_points = np.round((2**qp - 1) * points_s_np).astype('int')
            sub_points, idx = np.unique(sub_points, return_index=True, axis=0)
            draco_input_ply_path = os.path.join(integer_points_dir, prefix + '.ply')
            write_ply_ascii_geo(filedir = draco_input_ply_path, coords=sub_points)
            '''
                step2: use Draco to compress the downsampled point cloud
            '''
            encoder_path = '%s/draco_encoder -point_cloud'%(draco_path)
            draco_bitstream_path = os.path.join(draco_drc_dir, prefix + '.drc')
            t0 = time.time()
            draco_enc_cmd = "{} -i {} -o {} -cl {} -qp {}".format(encoder_path, draco_input_ply_path, draco_bitstream_path, cl, qp)
            os.system(draco_enc_cmd)
            draco_enc_times = time.time() - t0
            eval_dict['time_draco_enc'] = draco_enc_times
            #   add the number of bits to store shift, max_coord, min_coord
            coord_stream_size = 8*os.path.getsize(draco_bitstream_path) + 32*5
            eval_dict['bpp_base_layer'] = coord_stream_size/num_points
            '''
                step 3: use Draco to decompress the downsampled point cloud
            '''
            draco_ply_path = os.path.join(draco_ply_dir, prefix + '.ply')
            decoder_path = '%s/draco_decoder -point_cloud'%(draco_path)
            t0 = time.time()
            draco_dec_cmd = "{} -i {} -o {}".format(decoder_path, draco_bitstream_path, draco_ply_path)
            os.system(draco_dec_cmd)
            draco_dec_times = time.time() - t0
            eval_dict['time_draco_dec'] = draco_dec_times
            '''
                step 4: interpolation
            '''
            fps_pcd = o3d.io.read_point_cloud(draco_ply_path)
            fps_points = 2*(np.asarray(fps_pcd.points)/(2**qp - 1)) - 1 #dynamic range [0, 511 or 255] -> [0,1] -> [-1,1]
            sparse_pcd = torch.from_numpy(fps_points).to(self.device).to(torch.float32).unsqueeze(0)
            gt_pcd = 2*points_t - 1
            
            B = sparse_pcd.shape[0]
            K = sparse_pcd.shape[1]
            
            C = ds_ratio

            t0 = time.time()
            if C<=8:
                interpolated_pcd = golden_interpolation(sparse_pcd, C)
            else:
                interpolated_pcd = golden_interpolation2(sparse_pcd, C)
            eval_dict['time_interpolation'] = time.time() - t0
            reshaped_interpolated_pcd = interpolated_pcd.view(B, -1, 3)
            '''
                step 5: calculate residual and convert residual into feature
            '''
            t0 = time.time()
            _, _, nns = knn_points(reshaped_interpolated_pcd, gt_pcd, K = 1, return_nn=True)
            residuals = nns.squeeze(-2) - reshaped_interpolated_pcd
            residuals_cluster = residuals.view(B, K, C, 3)
            local_centers = torch.mean(interpolated_pcd, dim=2)
            central_diff = interpolated_pcd - local_centers.unsqueeze(2)
            feats = self.encoder(residuals_cluster, central_diff)
            eval_dict['time_ga'] = time.time() - t0
            if self.codec == 'ffp':
                g_latents_str, g_bpp, enc_time, eb_size = self.compressor.compress(g_coords = sparse_pcd, g_feats = feats, points_num = B*num_points)
                eval_dict['bpp_g'] = g_bpp
                bpps = g_bpp
            else:
                g_latents_str, h_latents_str, g_bpp, h_bpp, enc_time, eb_size = self.compressor.compress(g_coords = sparse_pcd, g_feats = feats, points_num = B*num_points)
                eval_dict['bpp_h'] = h_bpp
                eval_dict['bpp_g'] = g_bpp
                bpps = g_bpp + h_bpp
            eval_dict['time_encoding_latent_feature'] = enc_time
            eval_dict['all_time_encoding'] = eval_dict['time_fps'] + eval_dict['time_draco_enc'] + eval_dict['time_draco_dec'] + eval_dict['time_interpolation'] + eval_dict['time_ga'] + eval_dict['time_encoding_latent_feature']
            
            eval_dict['bpp_refinement_layer'] = bpps
            eval_dict['bpp'] = eval_dict['bpp_refinement_layer'] + eval_dict['bpp_base_layer']
            latents_str_dict = {}
            if self.codec == 'ffp':
                latents_str_dict['g_latents_str'] = g_latents_str
            else:
                latents_str_dict['g_latents_str'] = g_latents_str
                latents_str_dict['h_latents_str'] = h_latents_str
            return eval_dict, latents_str_dict, shift, max_coord, min_coord, eb_size
            
    def decompress_step(self, gt_pcd_o3d, latents_str_dict, shift, max_coord, min_coord, eb_size, draco_drc_dir, draco_ply_dir, pred_dir, prefix, qp, upsampling_ratio, draco_path):
        eval_dict = {}
        points = np.asarray(gt_pcd_o3d.points)
        gt_pcd = torch.from_numpy(points).to(torch.float32).unsqueeze(0).to(self.device) # only used for evaluation
        num_points = len(gt_pcd_o3d.points)
        with torch.no_grad():
            '''
                step 1: use Draco to decompress the downsampled point cloud
            '''
            draco_ply_path = os.path.join(draco_ply_dir, prefix + '.ply')
            decoder_path = '%s/draco_decoder -point_cloud'%(draco_path)
            draco_bitstream_path = os.path.join(draco_drc_dir, prefix + '.drc')
            t0 = time.time()
            draco_dec_cmd = "{} -i {} -o {}".format(decoder_path, draco_bitstream_path, draco_ply_path)
            os.system(draco_dec_cmd)
            draco_dec_times = time.time() - t0
            eval_dict['time_draco_dec'] = draco_dec_times
            '''
                step 2: interpolation
            '''
            fps_pcd = o3d.io.read_point_cloud(draco_ply_path)
            fps_points = 2*(np.asarray(fps_pcd.points)/(2**qp - 1)) - 1 #dynamic range [0, 511 or 255] -> [0,1] -> [-1,1]
            sparse_pcd = torch.from_numpy(fps_points).to(self.device).to(torch.float32).unsqueeze(0)
            
            B = sparse_pcd.shape[0]
            K = sparse_pcd.shape[1]
            C = upsampling_ratio
            #print([num_points, K, C])
            t0 = time.time()
            if C<=8:
                interpolated_pcd = golden_interpolation(sparse_pcd, C)
            else:
                interpolated_pcd = golden_interpolation2(sparse_pcd, C)
            local_centers = torch.mean(interpolated_pcd, dim=2)
            central_diff = interpolated_pcd - local_centers.unsqueeze(2)
            eval_dict['time_interpolation'] = time.time() - t0
            reshaped_interpolated_pcd = interpolated_pcd.view(B, -1, 3)
            
            if self.codec == 'ffp':
                feats, dec_time = self.compressor.decompress(g_coords = sparse_pcd, g_latents_str = latents_str_dict['g_latents_str'], points_num = B*num_points, eb_size = eb_size)
            else:
                feats, dec_time = self.compressor.decompress(g_coords = sparse_pcd, g_latents_str = latents_str_dict['g_latents_str'], h_latents_str = latents_str_dict['h_latents_str'], points_num = B*num_points, eb_size = eb_size)
            eval_dict['time_decoding_latent_feature'] = dec_time
            D = feats.shape[2]
            t0d = time.time()
            pred_residuals = self.decoder(central_diff.view(B, -1, 3), feats.unsqueeze(2).repeat(1, 1, C, 1).view(B, -1, D))
            pred_points = reshaped_interpolated_pcd + pred_residuals
            eval_dict['time_gs'] = time.time() - t0d
            eval_dict['all_time_decoding'] = eval_dict['time_draco_dec'] + eval_dict['time_interpolation'] + eval_dict['time_gs'] + eval_dict['time_decoding_latent_feature']
            eval_dict['cd_loss_ref'], _ = self.cal_cd_in_original_scale(reshaped_interpolated_pcd, gt_pcd, shift, max_coord, min_coord)
            eval_dict['cd_loss'], pred_pcd = self.cal_cd_in_original_scale(pred_points, gt_pcd, shift, max_coord, min_coord)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pred_pcd.cpu().numpy())
            save_name = os.path.join(pred_dir, prefix + '.ply')
            o3d.io.write_point_cloud(save_name, pcd)
        return eval_dict
    
    def upsample_after_comp_step(self, fps_time, save_name, decompressed_points, fps_points, shift, bbox_max, bbox_min, upsampling_ratio_1, upsampling_ratio_2):
        '''
            Use CRCIR as an upsampling network that upsamples the decompressed point cloud.
            Use this function when the decompression is done. 
        '''
        self.encoder.eval()
        self.decoder.eval()
        self.compressor.eval()
        device = self.device
        gt_pcd = decompressed_points
        sparse_pcd = fps_points
        num_points = gt_pcd.shape[1]
        B = sparse_pcd.shape[0]
        K = sparse_pcd.shape[1]

        C1 = upsampling_ratio_1
        C2 = upsampling_ratio_2
        eval_dict = {}
        t0 = time.time()
        with torch.no_grad():
            eval_dict['fps_time'] = fps_time
            if C1<=8:
                interpolated_pcd = golden_interpolation(sparse_pcd, C1)
            else:
                interpolated_pcd = golden_interpolation2(sparse_pcd, C1)
            reshaped_interpolated_pcd = interpolated_pcd.view(B, -1, 3)
            _, _, nns = knn_points(reshaped_interpolated_pcd, gt_pcd, K = 1, return_nn=True)
            residuals = nns.squeeze(-2) - reshaped_interpolated_pcd
            residuals_cluster = residuals.view(B, K, C1, 3)
            local_centers = torch.mean(interpolated_pcd, dim=2)
            central_diff = interpolated_pcd - local_centers.unsqueeze(2)
            feats = self.encoder(residuals_cluster, central_diff)

            if self.codec == 'ffp':
                g_latents_str, g_bpp, enc_time, eb_size = self.compressor.compress(g_coords = sparse_pcd, g_feats = feats, points_num = B*num_points)
                
            else:
                g_latents_str, h_latents_str, g_bpp, h_bpp, enc_time, eb_size = self.compressor.compress(g_coords = sparse_pcd, g_feats = feats, points_num = B*num_points)
                
            
            if self.codec == 'ffp':
                feats, dec_time = self.compressor.decompress(g_coords = sparse_pcd, g_latents_str = g_latents_str, points_num = B*num_points, eb_size = eb_size)
            else:
                feats, dec_time = self.compressor.decompress(g_coords = sparse_pcd, g_latents_str = g_latents_str, h_latents_str = h_latents_str, points_num = B*num_points, eb_size = eb_size)
            
            D = feats.shape[2]
            
            if C1*C2 <= 8:
                interpolated_pcd2 = golden_interpolation(sparse_pcd, C1*C2)
            else:
                interpolated_pcd2 = golden_interpolation2(sparse_pcd, C1*C2)
            central_diff = interpolated_pcd2 - local_centers.unsqueeze(2)
            pred_residuals = self.decoder(central_diff.view(B, -1, 3), feats.unsqueeze(2).repeat(1, 1, C1*C2, 1).view(B, -1, D))
            reshaped_interpolated_pcd2 = interpolated_pcd2.view(B, -1, 3)
            pred_points = reshaped_interpolated_pcd2 + pred_residuals
            eval_dict['upsampling_time'] = time.time() - t0 + fps_time
            pcd = (pred_points.squeeze(0).cpu().numpy()/2+0.5)*(bbox_max - bbox_min) + bbox_min + shift
            np.savetxt(save_name, pcd)
        return eval_dict
    
    def train_step(self, data, C = 34):
        ''' Performs a training step.

         Args:
            data (tuple): 
            (
                input
                querys
                gts
            )
        '''
        self.encoder.train()
        self.decoder.train()
        self.compressor.train()
        self.optimizer.zero_grad()
        loss, distortion, bpps = self.fit_crcir(data, C)
        loss.backward()
        self.optimizer.step()
        aux_loss = self.compressor.module.entropy_bottleneck.loss()
        aux_loss.backward()
        self.aux_optimizer.step()
        return loss.item(), distortion.item(), bpps.item(), aux_loss.item()
    
    
    def fit_crcir(self, batch, C = 34):
        ''' Computes the loss.'''
        device = self.device
        gt_pcd = batch[0].to(device)
        sparse_pcd = batch[1].to(device)
        B = sparse_pcd.shape[0]
        K = sparse_pcd.shape[1]
        interpolated_pcd = golden_interpolation(sparse_pcd, C)
        reshaped_interpolated_pcd = interpolated_pcd.view(B, -1, 3)
        _, _, nns = knn_points(reshaped_interpolated_pcd, gt_pcd, K = 1, return_nn=True)
        residuals = nns.squeeze(-2) - reshaped_interpolated_pcd
        #print(residuals.shape)
        residuals_cluster = residuals.view(B, K, C, 3)
        local_centers = torch.mean(interpolated_pcd, dim=2)
        central_diff = interpolated_pcd - local_centers.unsqueeze(2)
        feats = self.encoder(residuals_cluster, central_diff)
        if self.codec == 'ffp':
            feats, g_bpp = self.compressor(sparse_pcd, feats, points_num = B*100000)
            bpps = g_bpp.mean()
        else:
            feats, g_bpp, h_bpp = self.compressor(sparse_pcd, feats, points_num = B*100000)
            bpps = g_bpp.mean() + h_bpp.mean()
        D = feats.shape[2]
        pred_residuals = self.decoder(central_diff.view(B, -1, 3), feats.unsqueeze(2).repeat(1, 1, C, 1).view(B, -1, D))
        error_metric = L1Loss()
        distortion = error_metric(residuals_cluster.view(B, -1, 3), pred_residuals)
        loss = distortion + self.lambd * bpps
        return loss, distortion, bpps
    
    

    
    

        
        
    
    
