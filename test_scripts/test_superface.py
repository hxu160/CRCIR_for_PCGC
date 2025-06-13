import os

K_list = [48, 30, 24, 20, 15, 12]

for k_value in K_list:
    cur_cmd = "python test.py result/ex0_hyper_5e_3/config.yaml --gt_path other_dataset/superface_points/test --save_dir_name superface/superface_draco_qp9 --K_e {} --K_d {}".format(k_value, k_value)
    os.system(cur_cmd)