from pytorch3d.loss import chamfer_distance, point_mesh_face_distance
import numpy as np
import argparse
import open3d as o3d
import os
import torch
from tqdm import tqdm
from pytorch3d.io import IO
from pytorch3d.structures import Pointclouds

mesh_io = IO()
def get_instance_dir_for_each_split(path, mode, postfix):
    txt_name = os.path.join(path, mode+'.txt')
    f = open(txt_name, 'r')
    files = f.readlines()
    #lines = files[0].split('.ply')
    #lines = files[0].split('.off')
    lines = files[0].split(postfix)
    instances_name = lines[0:-1]
    return instances_name

parser = argparse.ArgumentParser(
    description='Train a 3D reconstruction model.'
)
parser.add_argument('--mesh_dir', type=str, default='~/3D/implicit_surface/data/CRCIR_dataset/mesh/test', help='path to ground truth mesh.')
parser.add_argument('--split_dir', type=str, default='dataset/split', help='path to name list of meshes to be tested.')
parser.add_argument('--split_postfix', type=str, default='.ply', help='.ply or .off')
parser.add_argument('--pred_dir', type=str, default='result/ex0_hyper_5e_3/shapenet/shapenet_draco_qp9_48_48_test/pred', help='path to reconstructed point cloud.')
parser.add_argument('--postfix', type=str, default='.obj', help='.obj or .off or .ply')
args = parser.parse_args()

mode = 'test'
mesh_dir = os.path.expanduser(args.mesh_dir)
pcd_item = get_instance_dir_for_each_split(os.path.expanduser(args.split_dir), mode, args.split_postfix)
pred_dir = args.pred_dir
p2ms = []
num_instance = len(pcd_item)
with tqdm(total=num_instance) as pbar:
    for i in range(num_instance):
        item = pcd_item[i]
        #gt_mesh = mesh_io.load_mesh(os.path.join(mesh_dir, mode, item + '.obj')).cuda()
        #gt_mesh = mesh_io.load_mesh(os.path.join(mesh_dir, mode, item+'.off')).cuda()
        #gt_mesh = mesh_io.load_mesh(os.path.join(mesh_dir, mode, item+'.ply')).cuda()
        gt_mesh = mesh_io.load_mesh(os.path.join(mesh_dir, item + args.postfix)).cuda()
        pred_path = os.path.join(pred_dir, item + '.ply')
        pred_pcd = o3d.io.read_point_cloud(pred_path)
        pred_points = np.array(pred_pcd.points)
        pred_points = torch.from_numpy(pred_points).to(torch.float32).to(torch.device('cuda')).unsqueeze(0)
        p2m = point_mesh_face_distance(gt_mesh, Pointclouds(pred_points))
        pbar.update(1)
        p2ms.append(p2m.item())
print(np.mean(np.asarray(p2ms)))
print('In %s, the average p2m is %.8f.'%(pred_dir, np.mean(np.asarray(p2ms)))) 
    
    