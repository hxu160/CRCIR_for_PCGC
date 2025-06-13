import os

K_e_list = [10]
K_d_list = [10]

for k_e_value in K_e_list:
    for k_d_value in K_d_list:
        cur_cmd = "python test.py result/ex0_hyper_5e_3/config.yaml --gt_path other_dataset/pugan/test --save_dir_name pugan/pugan_qp9 --K_e {} --K_d {}".format(k_e_value, k_d_value)
        os.system(cur_cmd)