import os
from tqdm import tqdm
import time
import argparse
import os
import numpy as np
import open3d as o3d
import trimesh

def parse_dataset_args():
    parser = argparse.ArgumentParser(description='Shapenet Dataset Arguments')
    # data root
    parser.add_argument('--mesh_dir', default='~/3D/CRCIR_code/data/PU-GAN/test', type=str, help='path to shapenet core dataset')
    parser.add_argument('--number_of_points', default=120000, type=int, help='the number of points sampled in the whole space')
    # save dir for watertight mesh
    parser.add_argument('--output_dir', default='other_dataset/pugan/gt_8x', type=str, help='save dir for sdf and point cloud')
    args = parser.parse_args()
    return args
args = parse_dataset_args()
mesh_dir = os.path.expanduser(args.mesh_dir)
names = os.listdir(mesh_dir)
os.makedirs(args.output_dir, exist_ok=True)
for name in names:
    file_name = name[0:-4]
    mesh_name = os.path.join(mesh_dir, file_name+'.off')
    mesh = o3d.io.read_triangle_mesh(mesh_name)
    mesh.compute_vertex_normals(normalized=True)
    cur_pcd = mesh.sample_points_uniformly(args.number_of_points)
    output_pcd_path = os.path.join(args.output_dir, file_name + '.ply')
    o3d.io.write_point_cloud(output_pcd_path, cur_pcd)

