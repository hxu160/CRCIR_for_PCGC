import os

#K_list = [48, 30, 24, 20, 15, 12, 10, 8, 6]
K_list = [2,3,4,5,6,7,8]
for k_value in K_list:
    cur_cmd = "python test_sr_after_decompression.py result/ex0_hyper_5e_3/config.yaml --K_d {}".format(k_value)
    os.system(cur_cmd)