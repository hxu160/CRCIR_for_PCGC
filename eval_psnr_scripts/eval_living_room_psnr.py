import os

K_list = [48, 30, 24, 20, 15, 12]

for k_value in K_list:
    cur_cmd = "python eval_psnr.py --pred_dir result/ex0_hyper_5e_3/living_room/living_room_draco_qp9_{}_{}_test/pred --gt_dir ~/3D/CRCIR_code/data/LivingRoom_points/test --from_gt_dir --check_nan_psnr".format(k_value, k_value)
    os.system(cur_cmd)