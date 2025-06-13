import argparse
import os
import torch
import open3d as o3d
import numpy as np
from tqdm import tqdm
from pytorch3d.ops import knn_points, knn_gather

def sum_d2(p1, p2, normal):
    # x: (n, 3)
    return torch.sum(torch.sum((p1 - p2) * normal, dim=1) ** 2)

def psnr(peak, mse):
    max_energy = 3*peak*peak
    return 10 * torch.log10(max_energy / (mse))

def cal_psnr(peak, gt_xyzs, gt_normals, pred_xyzs):
    #   for every point in gt, find its nearest neighbor in predicted pc.
    _, idx1, gt_nearest_xyzs = knn_points(p1 = gt_xyzs, p2 = pred_xyzs, K=1, return_nn = True)
    #   for every point in predicted pc, find its nearest neighbor in gt.
    _, idx2, pred_nearest_xyzs = knn_points(p1 = pred_xyzs, p2 = gt_xyzs, K=1, return_nn = True)
    gt2pred_idx = idx1.squeeze(0).squeeze(1)
    pred2gt_idx = idx2.squeeze(0).squeeze(1)
    pred_normals = gt_normals[:, pred2gt_idx, :]
    gt_nearest_normals = pred_normals[:, gt2pred_idx, :]
    pred_nearest_normals = gt_normals[:, pred2gt_idx, :]
    gt_xyzs = gt_xyzs.squeeze(0)
    gt_nearest_xyzs = gt_nearest_xyzs.squeeze(0).squeeze(1)
    gt_nearest_normals = gt_nearest_normals.squeeze(0)
    pred_xyzs = pred_xyzs.squeeze(0)
    pred_nearest_xyzs = pred_nearest_xyzs.squeeze(0).squeeze(1)
    pred_nearest_normals = pred_nearest_normals.squeeze(0)
    d2_sum_gt2pred = sum_d2(gt_xyzs, gt_nearest_xyzs, gt_nearest_normals)
    d2_sum_pred2gt = sum_d2(pred_xyzs, pred_nearest_xyzs, pred_nearest_normals)
    d2_mse_gt2pred = d2_sum_gt2pred / gt_xyzs.shape[0]
    d2_mse_pred2gt = d2_sum_pred2gt / pred_xyzs.shape[0]
    d2_max_mse = max(d2_mse_gt2pred, d2_mse_pred2gt)
    
    d2_psnr = psnr(peak, d2_max_mse)
    return d2_psnr.cpu()

def get_instance_dir_for_each_split(path, mode, postfix):
    txt_name = os.path.join(path, mode+'.txt')
    f = open(txt_name, 'r')
    files = f.readlines()
    #lines = files[0].split('.ply')
    #lines = files[0].split('.off')
    lines = files[0].split(postfix)
    instances_name = lines[0:-1]
    return instances_name

def get_psnr_peak(pcd):
    peak = np.max(np.asarray(pcd.compute_nearest_neighbor_distance()))
    return peak


parser = argparse.ArgumentParser(
    description='Train a 3D reconstruction model.'
)
parser.add_argument('--split_dir', type=str, default='dataset/split', help='path to name list of meshes to be tested.')
parser.add_argument('--split_postfix', type=str, default='.ply', help='.ply or .off')
parser.add_argument('--pred_dir', type=str, default='result/ex0_hyper_5e_3/shapenet/shapenet_draco_qp9_48_48_test/pred', help='path to reconstructed point cloud.')
parser.add_argument('--gt_dir', type=str, default='~/3D/implicit_surface/data/CRCIR_dataset/Shapenet_points/test', help='path to gt point cloud.')
parser.add_argument('--from_gt_dir', action='store_true', help='Do not use split.txt to load data.')
parser.add_argument('--check_nan_psnr', action='store_true', help='filter nan values.')
args = parser.parse_args()
pcd_dir = os.path.expanduser(args.gt_dir)
mode = 'test'
if args.from_gt_dir:
    pcd_item = os.listdir(pcd_dir)
else:
    pcd_item = get_instance_dir_for_each_split(os.path.expanduser(args.split_dir), mode, args.split_postfix)

pred_dir = args.pred_dir
psnrs = []
num_instance = len(pcd_item)

with tqdm(total=num_instance) as pbar:
    for i in range(num_instance):
        item = pcd_item[i]
        if args.from_gt_dir:
            path = os.path.join(pcd_dir, item)
            pred_path = os.path.join(pred_dir, item)
        else:    
            path = os.path.join(pcd_dir, item + '.ply')
            pred_path = os.path.join(pred_dir, item + '.ply')
        pcd = o3d.io.read_point_cloud(path)
        xyzs = np.array(pcd.points)
        normals = np.array(pcd.normals)
        pred_pcd = o3d.io.read_point_cloud(pred_path)
        pred_xyzs = np.array(pred_pcd.points)
        #print([xyzs.shape, pred_xyzs.shape])
        #print(get_psnr_peak(pcd))
        cur_psnr = cal_psnr(get_psnr_peak(pcd), 
             torch.from_numpy(xyzs).to(torch.float32).cuda().unsqueeze(0),
             torch.from_numpy(normals).to(torch.float32).cuda().unsqueeze(0),
             torch.from_numpy(pred_xyzs).to(torch.float32).cuda().unsqueeze(0))
        psnrs.append(cur_psnr.item())
        pbar.update(1)


if args.check_nan_psnr:
    non_inf_psnrs = np.asarray(psnrs)[~np.isnan(psnrs)]
else:
    non_inf_psnrs = np.asarray(psnrs)[~np.isinf(psnrs)]
print('In %s, the average PSNR is %f'%(pred_dir, np.mean(non_inf_psnrs)))    