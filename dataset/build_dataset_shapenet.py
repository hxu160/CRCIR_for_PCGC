import os
from tqdm import tqdm
import argparse
import os
import open3d as o3d

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

def parse_dataset_args():
    parser = argparse.ArgumentParser(description='Shapenet Dataset Arguments')
    # data root
    parser.add_argument('--mesh_dir', default='~/3D/implicit_surface/data/CRCIR_dataset/mesh', type=str, help='path to shapenet core dataset')
    parser.add_argument('--split_dir', default='dataset/split', type=str, help='save dir for dataset split')
    parser.add_argument('--mode', default='test', type=str, help='save dir for dataset split')
    parser.add_argument('--number_of_points', default=120000, type=int, help='the number of points sampled in the whole space')
    parser.add_argument('--output_dir', default='~/3D/implicit_surface/data/CRCIR_dataset/Shapenet_points', type=str, help='save dir for sdf and point cloud')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_dataset_args()
    pcd_item = get_instance_dir_for_each_split(args.split_dir, args.mode)
    print('There are %d items'%(len(pcd_item)))
    mesh_dir = os.path.expanduser(args.mesh_dir)
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(os.path.join(output_dir, args.mode), exist_ok=True)
    with tqdm(total = len(pcd_item)) as pbar:
        for name in pcd_item:
            
            mesh_name = os.path.join(mesh_dir, args.mode, name+'.obj')
            mesh = o3d.io.read_triangle_mesh(mesh_name)
            mesh.compute_vertex_normals(normalized=True)
            
            cur_pcd = mesh.sample_points_uniformly(args.number_of_points)
            output_pcd_path = os.path.join(output_dir, args.mode, name + '.ply')
            o3d.io.write_point_cloud(output_pcd_path, cur_pcd)
            pbar.update(1)
    
    