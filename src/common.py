import os
import torch
import math
import numpy as np
from pytorch3d.ops import knn_points

def denorm_points(points, shift, bmax, bmin):
    return (points/2 + 0.5)*(bmax-bmin) + bmin + shift

def norm_01(xyzs):
    shift = torch.mean(xyzs, dim=0)
    xyzs -= shift
    max_coord, min_coord = torch.max(xyzs), torch.min(xyzs)
    xyzs = xyzs - min_coord
    xyzs = xyzs / (max_coord - min_coord)
    return xyzs, shift, max_coord, min_coord
    
def write_ply_ascii_geo(filedir, coords):
    if os.path.exists(filedir): os.system('rm '+filedir)
    f = open(filedir,'a+')
    f.writelines(['ply\n','format ascii 1.0\n'])
    f.write('element vertex '+str(coords.shape[0])+'\n')
    f.writelines(['property float x\n','property float y\n','property float z\n'])
    f.write('end_header\n')
    coords = coords.astype('int')
    for p in coords:
        f.writelines([str(p[0]), ' ', str(p[1]), ' ',str(p[2]), '\n'])
    f.close() 
    return

def get_graph_feature(x, k=20):
    batch_size = x.size(0)
    num_points = x.size(2)
    feat = x.transpose(2, 1)#B*N*C
    _, _, neighbors = knn_points(feat, feat, K=k, return_nn=True)#B*N*K*C
    old_feat = feat.unsqueeze(2).repeat(1, 1, k, 1)#B*N*K*C
    new_feat = torch.cat([old_feat, neighbors-old_feat], dim=3).permute(0, 3, 1, 2).contiguous()
    return new_feat #B*(2*C)*N*K

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def midpoint_interpolation(points, R):
    '''
        points: 1*N*3 array or B*N*3 array
        R: upsampling ratio
    '''
    B = points.shape[0]
    _, _, coords = knn_points(points, points, K = R, return_nn = True)
    neighbors = coords[:, :, 0:R, :]#BxNxRx3
    midpoints = (neighbors + points.unsqueeze(2))/2
    return midpoints

def golden_interpolation(points, R):
    '''
        points: 1*N*3 array or B*N*3 array
        R: upsampling ratio
    '''
    B = points.shape[0]
    _, _, coords = knn_points(points, points, K = R, return_nn = True)
    neighbors = coords[:, :, 0:R, :]#BxNxRx3
    midpoints = points.unsqueeze(2) + (neighbors - points.unsqueeze(2))*((3-math.sqrt(5))/2)
    return midpoints

def golden_interpolation2(points, R):
    '''
        points: 1*N*3 array or B*N*3 array
        R: upsampling ratio
    '''
    B = points.shape[0]
    _, _, coords = knn_points(points, points, K = R, return_nn = True)
    neighbors1 = coords[:, :, 0:8, :]#BxNxRx3
    midpoints1 = points.unsqueeze(2) + (neighbors1 - points.unsqueeze(2))*((3-math.sqrt(5))/2)
    neighbors2 = coords[:, :, 8:R, :]#BxNxRx3
    midpoints2 = points.unsqueeze(2) + (neighbors2 - points.unsqueeze(2))*(0.75*(3-math.sqrt(5))/2)
    midpoints = torch.cat([midpoints1, midpoints2], dim=2)
    return midpoints

def save_model_for_AE(epoch_it, it, metric_val_best, optimizer, encoder, decoder, path):
    if torch.cuda.device_count() > 1:
        state = {
                    "epoch_it": epoch_it,
                    "it": it,
                    "loss_val_best": metric_val_best,
                    "optimizer": optimizer.state_dict(),
                    "encoder": encoder.module.state_dict(),
                    "decoder": decoder.module.state_dict(),
                }
        torch.save(state, path)
    else:
        state = {
                    "epoch_it": epoch_it,
                    "it": it,
                    "loss_val_best": metric_val_best,
                    "optimizer": optimizer.state_dict(),
                    "encoder": encoder.state_dict(),
                    "decoder": decoder.state_dict(),
                }
        torch.save(state, path)

def save_model(epoch_it, it, metric_val_best, optimizer, encoder, decoder, compressor, aux_optimizer, path):
    if torch.cuda.device_count() > 1:
        state = {
                    "epoch_it": epoch_it,
                    "it": it,
                    "loss_val_best": metric_val_best,
                    "optimizer": optimizer.state_dict(),
                    "encoder": encoder.module.state_dict(),
                    "decoder": decoder.module.state_dict(),
                    "compressor": compressor.module.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict()
                }
        torch.save(state, path)
    else:
        state = {
                    "epoch_it": epoch_it,
                    "it": it,
                    "loss_val_best": metric_val_best,
                    "optimizer": optimizer.state_dict(),
                    "encoder": encoder.state_dict(),
                    "decoder": decoder.state_dict(),
                    "compressor": compressor.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict()
                }
        torch.save(state, path)