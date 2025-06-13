import os

K_list = [48, 30, 24, 20, 15, 12]

for k_value in K_list:
    cur_cmd = "python eval_psnr.py --split_dir other_dataset/superface_points/split --gt_dir other_dataset/superface_points/test --pred_dir result/ex0_hyper_5e_3/superface/superface_draco_qp9_{}_{}_test/pred --split_postfix .off".format(k_value, k_value)
    os.system(cur_cmd)