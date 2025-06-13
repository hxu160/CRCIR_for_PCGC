import argparse
import os
import numpy as np
import open3d as o3d
import torch
import time
from tqdm import tqdm
from pytorch3d.ops import sample_farthest_points
def generate_dataset(mode, dataset_args):
    pcd_dir = os.path.expanduser(dataset_args.pcd_dir) 
    fps_path = os.path.join('%s/fps_%d'%(pcd_dir, dataset_args.num_points), mode)
    os.makedirs(fps_path, exist_ok=True)
    meta_path = os.path.join('%s/meta_data'%(pcd_dir), mode)
    os.makedirs(meta_path, exist_ok=True)
    pcd_path = os.path.join(pcd_dir, mode)
    names = os.listdir(pcd_path)
    with tqdm(total = len(names)) as pbar:
        for file_name in names:
            cur_pcd = o3d.io.read_point_cloud(os.path.join(pcd_path, file_name))
            xyzs = np.asarray(cur_pcd.points)
            shift = np.mean(xyzs, axis=0)
            xyzs -= shift
            max_coord, min_coord = np.max(xyzs), np.min(xyzs)
            xyzs = xyzs - min_coord
            xyzs = xyzs / (max_coord - min_coord)
            points_t = torch.from_numpy(xyzs).to(torch.float32).unsqueeze(0).cuda()
            points_s, points_idx = sample_farthest_points(points_t, K = dataset_args.num_points)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_s.cpu().squeeze(0).numpy())
            pcd.normals = o3d.utility.Vector3dVector(np.asarray(cur_pcd.normals)[points_idx.squeeze(0).cpu().numpy(), :])
            o3d.io.write_point_cloud(os.path.join(fps_path, file_name), pcd)
            np.savez_compressed(os.path.join(meta_path, file_name[0:-4]+'.npz'), points = xyzs, shift = shift, bmax = max_coord, bmin = min_coord)
            pbar.update(1)


def parse_dataset_args():
    parser = argparse.ArgumentParser(description='ShapeNet Dataset Arguments')
    parser.add_argument('--pcd_dir', default='~/3D/implicit_surface/data/CRCIR_dataset/Shapenet_points', type=str, help='save dir for sampled point cloud')
    parser.add_argument('--num_points', default=3000, type=int, help='the number of sampled points')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    dataset_args = parse_dataset_args()
    generate_dataset('train', dataset_args)
    #generate_dataset('test', dataset_args)
    generate_dataset('val', dataset_args)