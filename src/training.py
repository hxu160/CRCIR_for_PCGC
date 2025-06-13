import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import open3d as o3d
import torch
import time
from pytorch3d.ops import sample_farthest_points

class BaseTrainer(object):
    ''' Base trainer class.
    '''

    def evaluate(self, val_set, upsampling_ratio = 34):
        ''' Performs an evaluation.
        Args:
            val_set (Dataset): pytorch Dataset
        '''
        eval_list = defaultdict(list)
        num_shapes = len(val_set)
        with tqdm(total=num_shapes) as pbar:
            for i in range(num_shapes):
                eval_step_dict = self.eval_step(val_set[i], upsampling_ratio)
                for k, v in eval_step_dict.items():
                    eval_list[k].append(v)
                pbar.update(1)
        eval_dict = {k: np.mean(v) for k, v in eval_list.items()}
        return eval_dict
    
    def test_compression_performance(self, out_dir, args):
        #dense_path = args.gt_path
        dense_path = os.path.expanduser(args.gt_path)
        pcd_names = os.listdir(dense_path)
        num_shapes = len(pcd_names)
        eval_list = defaultdict(list)
        
        integer_points_dir = os.path.join(out_dir, 'integer_points')
        draco_drc_dir = os.path.join(out_dir, 'draco_drc')
        draco_ply_dir = os.path.join(out_dir, 'draco_ply')
        pred_dir = os.path.join(out_dir, 'pred')
        os.makedirs(integer_points_dir, exist_ok=True)
        os.makedirs(draco_drc_dir, exist_ok=True)
        os.makedirs(draco_ply_dir, exist_ok=True)
        os.makedirs(pred_dir, exist_ok=True)
        
        qp = args.QP
        
        ds_ratio = args.K_e
        upsampling_ratio = args.K_d
        draco_path = args.draco_path
        
        with tqdm(total=num_shapes) as pbar:
            for i in range(num_shapes):
                prefix = pcd_names[i][0:-4]
                #   load a dense point cloud
                gt_pcd = o3d.io.read_point_cloud(os.path.join(dense_path, prefix + '.ply'))
                
                enc_dict, latents_str_dict, shift, max_coord, min_coord, eb_size = self.compress_step(gt_pcd, integer_points_dir, draco_drc_dir, draco_ply_dir, prefix, qp, ds_ratio, draco_path)
                for k, v in enc_dict.items():
                    eval_list[k].append(v)
                dec_dict = self.decompress_step(gt_pcd, latents_str_dict, shift, max_coord, min_coord, eb_size, draco_drc_dir, draco_ply_dir, pred_dir, prefix, qp, upsampling_ratio, draco_path)
                for k, v in dec_dict.items():
                    eval_list[k].append(v)
                pbar.update(1)
        eval_dict = {k: np.mean(v) for k, v in eval_list.items()}
        return eval_dict
    
    def test_direct_sr_performance(self, out_dir, args):
        #dense_path = args.gt_path
        dense_path = os.path.expanduser(args.gt_path)
        dense_path_sr = os.path.expanduser(args.gt_path_for_sr)
        pcd_names = os.listdir(dense_path)
        num_shapes = len(pcd_names)
        eval_list = defaultdict(list)
        
        integer_points_dir = os.path.join(out_dir, 'integer_points')
        draco_drc_dir = os.path.join(out_dir, 'draco_drc')
        draco_ply_dir = os.path.join(out_dir, 'draco_ply')
        pred_dir = os.path.join(out_dir, 'pred')
        os.makedirs(integer_points_dir, exist_ok=True)
        os.makedirs(draco_drc_dir, exist_ok=True)
        os.makedirs(draco_ply_dir, exist_ok=True)
        os.makedirs(pred_dir, exist_ok=True)
        
        qp = args.QP
        
        ds_ratio = args.K_e
        upsampling_ratio = args.K_d
        draco_path = args.draco_path
        
        with tqdm(total=num_shapes) as pbar:
            for i in range(num_shapes):
                prefix = pcd_names[i][0:-4]
                #   load a dense point cloud
                gt_pcd = o3d.io.read_point_cloud(os.path.join(dense_path, prefix + '.ply'))
                gt_pcd_sr = o3d.io.read_point_cloud(os.path.join(dense_path_sr, prefix + '.ply'))
                enc_dict, latents_str_dict, shift, max_coord, min_coord, eb_size = self.compress_step(gt_pcd, integer_points_dir, draco_drc_dir, draco_ply_dir, prefix, qp, ds_ratio, draco_path)
                for k, v in enc_dict.items():
                    eval_list[k].append(v)
                
                dec_dict = self.decompress_step(gt_pcd_sr, latents_str_dict, shift, max_coord, min_coord, eb_size, draco_drc_dir, draco_ply_dir, pred_dir, prefix, qp, upsampling_ratio, draco_path)
                for k, v in dec_dict.items():
                    eval_list[k].append(v)
                pbar.update(1)
        eval_dict = {k: np.mean(v) for k, v in eval_list.items()}
        return eval_dict
    
    def test_sr_after_comp(self, out_dir, args):
        pcd_names = os.listdir(args.gt_path)
        num_shapes = len(pcd_names)
        eval_list = defaultdict(list)
        os.makedirs(out_dir, exist_ok=True)
        with tqdm(total=num_shapes) as pbar:
            for i in range(num_shapes):
                #   Only used for getting normalization parameters.
                #   These parameters are sent to the decoder side, which are known parameters. 
                read_pcd = o3d.io.read_point_cloud(os.path.join(args.gt_path, pcd_names[i]))
                data = np.asarray(read_pcd.points)
                shift = np.mean(data, axis=0, keepdims=True)
                c_points = data - shift
                bbox_min = np.min(c_points)
                bbox_max = np.max(c_points)
                #   Load the decompressed point cloud
                pred_pcd = o3d.io.read_point_cloud(os.path.join(args.pred_path, pcd_names[i]))
                pred_points = np.asarray(pred_pcd.points)
                pred_points = pred_points - shift
                pred_points = (pred_points - bbox_min)/(bbox_max - bbox_min)
                pred_points = torch.from_numpy(pred_points).unsqueeze(0).to(torch.float32).cuda()
                num_points = pred_points.shape[1]
                t0 = time.time()
                fps_points, _ = sample_farthest_points(pred_points, K=num_points//args.K_e)
                tfps = time.time() - t0
                pred_points = 2*pred_points - 1
                fps_points = 2*fps_points - 1
                save_name = os.path.join(out_dir, pcd_names[i][0:-4]+'.xyz')
                eval_step_dict = self.upsample_after_comp_step(tfps, save_name, pred_points, fps_points, shift, bbox_max, bbox_min, args.K_e, args.K_d)
                for k, v in eval_step_dict.items():
                    eval_list[k].append(v)
                pbar.update(1)
        eval_dict = {k: np.mean(v) for k, v in eval_list.items()}
        return eval_dict
                
                
    def train_step(self, *args, **kwargs):
        ''' Performs a training step.
        '''
        raise NotImplementedError
                