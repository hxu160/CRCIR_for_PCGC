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
def get_instance_dir_for_each_split(path, mode):
    txt_name = os.path.join(path, mode+'.txt')
    f = open(txt_name, 'r')
    files = f.readlines()
    lines = files[0].split('.ply')
    instances_name = lines[0:-1]
    names = []
    for instance_dir in instances_name:
        names.append(os.path.split(instance_dir)[-1])
    return names
parser = argparse.ArgumentParser(
    description='Train a 3D reconstruction model.'
)
parser.add_argument('--up_rate', type=int, default=2)
args = parser.parse_args()

mesh_dir = os.path.expanduser('~/3D/CRCIR_code/data/PU-GAN/test')
gt_dir = 'other_dataset/pugan/gt_%dx'%(args.up_rate)
pred_dir = 'result/ex0_hyper_5e_3/pugan_up_after_comp/pugan_2_%d_test/pred'%(args.up_rate)
pcd_item = os.listdir(mesh_dir)
cds = []
p2ms = []
num_instance = len(pcd_item)
print(num_instance)
with tqdm(total=num_instance) as pbar:
    for i in range(num_instance):
        item = pcd_item[i][0:-4]
        path = os.path.join(gt_dir, item + '.ply')
        gt_pcd = o3d.io.read_point_cloud(path)
        gt_points = np.array(gt_pcd.points)
        gt_points = torch.from_numpy(gt_points).to(torch.float32).to(torch.device('cuda')).unsqueeze(0)
        gt_mesh = mesh_io.load_mesh(os.path.join(mesh_dir, item+'.off')).cuda()
        pred_points = np.genfromtxt(os.path.join(pred_dir, item + '.xyz'))
        pred_points = torch.from_numpy(pred_points).to(torch.float32).to(torch.device('cuda')).unsqueeze(0)
        loss, _ = chamfer_distance(pred_points, gt_points)
        p2m = point_mesh_face_distance(gt_mesh, Pointclouds(pred_points))
        pbar.update(1)
        cds.append(loss.item())
        p2ms.append(p2m.item())

print('In %s, the average cd is %.8f, the average p2m is %.8f.'%(pred_dir, np.mean(np.asarray(cds)), np.mean(np.asarray(p2ms)))) 
