import os
import torch
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset
class PointCloud_Dataset(Dataset):
    '''Point Cloud Dataset'''
    def __init__(self, cfg, gt_path, path, mode) -> None:
        ''' Returns the dataset instance
        
        Args:
            cfg (dict)
        '''
        super().__init__()
        self.cfg = cfg
        gts, sparse_points = self.load(gt_path, path, mode)
        self.gts = gts
        self.sparse_points = sparse_points

    
    def __len__(self):
        return len(self.gts)
    
    def __getitem__(self, index):
        return (
                self.gts[index],
                self.sparse_points[index])
    
    def load(self, gt_path, path, mode):
        gt_path = os.path.expanduser(gt_path)
        path = os.path.expanduser(path)
        instance_dir = os.listdir(os.path.join(gt_path, mode))
        gts = []
        sparse_points = []
        for instances in instance_dir:
            data =  np.load(os.path.join(gt_path, mode, instances))
            pc_points = data['points']
            fps_pcd = o3d.io.read_point_cloud(os.path.join(path, mode, instances[0:-4]+'.ply'))
            fps_points = 2*np.asarray(fps_pcd.points) - 1
            gts.append(torch.from_numpy(2*pc_points-1).to(torch.float32))
            sparse_points.append(torch.from_numpy(fps_points).to(torch.float32))
        return gts, sparse_points

