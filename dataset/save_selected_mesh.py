from tqdm import tqdm
import trimesh
import argparse
import os

def get_instance_dir(path, mode):
    txt_name = os.path.join(path, mode+'.txt')
    f = open(txt_name, 'r')
    files = f.readlines()
    lines = files[0].split('.ply')
    instances_name = lines[0:-1]
    return instances_name

def parse_dataset_args():
    parser = argparse.ArgumentParser(description='Shapenet Dataset Arguments')
    # data root
    parser.add_argument('--data_root', default='~/3D/implicit_surface/data', type=str, help='path to shapenet core dataset')
    parser.add_argument('--mesh_dir', default='~/3D/implicit_surface/data/CRCIR_dataset/mesh', type=str, help='save dir for selected mesh')
    parser.add_argument('--split_dir', default='dataset/split', type=str, help='save dir for dataset split')
    parser.add_argument('--mode', default='train', type=str, help='save dir for dataset split')
    args = parser.parse_args()
    return args

#   python dataset/save_selected_mesh.py --mode train
#   python dataset/save_selected_mesh.py --mode test
#   python dataset/save_selected_mesh.py --mode val
if __name__ == '__main__':
    dataset_args = parse_dataset_args()
    dataset_args.data_root = os.path.join(dataset_args.data_root, 'ShapeNetCore.v1')
    mesh_dir = os.path.expanduser(dataset_args.mesh_dir)
    
    os.makedirs(os.path.join(mesh_dir, dataset_args.mode), exist_ok=True)
    instance_dir_path = 'dataset/split/instance_dir.txt'
    f = open(instance_dir_path, "r")
    lines = f.readlines()
    instance_dir = []
    for line in lines:
        line = line.strip('\n')
        instance_dir.append(line) 
    train_dir = get_instance_dir(dataset_args.split_dir, dataset_args.mode)
    print(len(train_dir))
    num_instance = len(instance_dir)
    total_test = 0
    total_val = 0
    with tqdm(total=num_instance) as pbar:
        for instance in instance_dir:
            prefix_name = instance.split('/')[-1]
            instance_path = os.path.join(instance, 'model.obj')
            
            mesh = trimesh.load(instance_path)
            if prefix_name in train_dir:
                output_path = os.path.join(mesh_dir, dataset_args.mode, prefix_name + '.obj')
                mesh.export(output_path)
            pbar.update(1)

    
    