import os

K_list = [2,3,4,5,6,7,8]

for k_value in K_list:
    cur_cmd = "python eval_cd_p2m_for_pugan.py --up_rate {}".format(k_value)
    os.system(cur_cmd)