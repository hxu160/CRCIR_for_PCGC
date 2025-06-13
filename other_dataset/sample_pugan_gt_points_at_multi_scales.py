import os
from tqdm import tqdm
import time
import argparse
import os
import numpy as np
import open3d as o3d
import trimesh
import torch
from pytorch3d.ops import sample_farthest_points
def parse_dataset_args():
    parser = argparse.ArgumentParser(description='Shapenet Dataset Arguments')
    # data root
    parser.add_argument('--gt_dir', default='pugan/gt_8x', type=str, help='path to shapenet core dataset')
    parser.add_argument('--num_points', default=105000, type=int, help='the number of points sampled in the whole space')
    # save dir for watertight mesh
    parser.add_argument('--output_dir', default='pugan/gt_7x', type=str, help='save dir for sdf and point cloud')
    args = parser.parse_args()
    return args
args = parse_dataset_args()
names = os.listdir(args.gt_dir)
os.makedirs(args.output_dir, exist_ok=True)
for name in names:
    file_name = name[0:-4]
    pcd_name = os.path.join(args.gt_dir, file_name+'.ply')
    cur_pcd = o3d.io.read_point_cloud(pcd_name)
    xyzs = np.asarray(cur_pcd.points)
    
    points_t = torch.from_numpy(xyzs).to(torch.float32).unsqueeze(0).cuda()
    points_s, points_idx = sample_farthest_points(points_t, K = args.num_points)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_s.cpu().squeeze(0).numpy())
    pcd.normals = o3d.utility.Vector3dVector(np.asarray(cur_pcd.normals)[points_idx.squeeze(0).cpu().numpy(), :])
    
    output_pcd_path = os.path.join(args.output_dir, file_name + '.ply')
    o3d.io.write_point_cloud(output_pcd_path, pcd)

