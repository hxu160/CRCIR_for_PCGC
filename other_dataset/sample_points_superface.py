import os
import argparse
import os
import open3d as o3d
import trimesh
from glob import glob
from tqdm import tqdm

def generate_path_txt(pcd_dir, output_dir, mode):
    with open(os.path.join(output_dir, mode + '.txt'), 'w')as f:
        f.writelines(pcd_dir)
    f.close()
    
def parse_dataset_args():
    parser = argparse.ArgumentParser(description='Shapenet Dataset Arguments')
    # data root
    parser.add_argument('--mesh_dir', default='~/3D/CRCIR_code/data/Others/superface/*.off', type=str, help='path to shapenet core dataset')
    parser.add_argument('--number_of_points', default=120000, type=int, help='the number of points sampled in the whole space')
    parser.add_argument('--mode', default='test', type=str, help='test')
    parser.add_argument('--split_dir', default='other_dataset/superface_points/split', type=str, help='save dir for the list of file names')
    parser.add_argument('--output_dir', default='other_dataset/superface_points', type=str, help='save dir for the sampled points')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_dataset_args()
    names = glob(os.path.expanduser(args.mesh_dir), recursive=True)
    ply_names = []
    os.makedirs(args.split_dir, exist_ok=True)
    save_dir = os.path.join(args.output_dir, args.mode)
    os.makedirs(save_dir, exist_ok=True)
    with tqdm(total = len(names)) as pbar:
        for file_name in names:
            cur_name = os.path.split(file_name)[-1]
            ply_names.append(cur_name)
            read_mesh = trimesh.load(file_name)
            v,f = read_mesh.vertices, read_mesh.faces
            mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f))
            mesh.compute_vertex_normals(normalized=True)
            cur_pcd = mesh.sample_points_uniformly(args.number_of_points)
            output_pcd_path = os.path.join(save_dir, cur_name[0:-4]+'.ply')
            o3d.io.write_point_cloud(output_pcd_path, cur_pcd)
            pbar.update(1)
    generate_path_txt(ply_names, args.split_dir, 'test')
    