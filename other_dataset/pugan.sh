#   create the point cloud dataset for 8x upsampling
python other_dataset/sample_pugan_mesh.py
#   create the point cloud dataset for 7x upsampling
python other_dataset/sample_pugan_gt_points_at_multi_scales.py --gt_dir other_dataset/pugan/gt_8x --num_points 105000 --output_dir other_dataset/pugan/gt_7x
#   create the point cloud dataset for 6x upsampling
python other_dataset/sample_pugan_gt_points_at_multi_scales.py --gt_dir other_dataset/pugan/gt_7x --num_points 90000 --output_dir other_dataset/pugan/gt_6x
#   create the point cloud dataset for 5x upsampling
python other_dataset/sample_pugan_gt_points_at_multi_scales.py --gt_dir other_dataset/pugan/gt_6x --num_points 75000 --output_dir other_dataset/pugan/gt_5x
#   create the point cloud dataset for 4x upsampling
python other_dataset/sample_pugan_gt_points_at_multi_scales.py --gt_dir other_dataset/pugan/gt_5x --num_points 60000 --output_dir other_dataset/pugan/gt_4x
#   create the point cloud dataset for 5x upsampling
python other_dataset/sample_pugan_gt_points_at_multi_scales.py --gt_dir other_dataset/pugan/gt_4x --num_points 45000 --output_dir other_dataset/pugan/gt_3x
#   create the point cloud dataset for 2x upsampling
python other_dataset/sample_pugan_gt_points_at_multi_scales.py --gt_dir other_dataset/pugan/gt_3x --num_points 30000 --output_dir other_dataset/pugan/gt_2x
#   create the point cloud dataset to be compressed
python other_dataset/sample_pugan_gt_points_at_multi_scales.py --gt_dir other_dataset/pugan/gt_2x --num_points 15000 --output_dir other_dataset/pugan/test
