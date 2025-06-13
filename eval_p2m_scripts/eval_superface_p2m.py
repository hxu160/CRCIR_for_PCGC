import os

K_list = [48, 30, 24, 20, 15, 12]

for k_value in K_list:
    cur_cmd = "python eval_p2m.py --mesh_dir ~/3D/CRCIR_code/data/Others/superface --split_dir other_dataset/superface_points/split --pred_dir result/ex0_hyper_5e_3/superface/superface_draco_qp9_{}_{}_test/pred --split_postfix .off --postfix .off".format(k_value, k_value)
    os.system(cur_cmd)