# [ECCV 2024] Fast Point Cloud Geometry Compression with Context-based Residual Coding and INR-based Refinement
This is the official PyTorch implementation of our paper "Fast Point Cloud Geometry Compression with Context-based Residual Coding and INR-based Refinement" (ECCV 2024). [arXiv](https://arxiv.org/pdf/2408.02966)
# Abstract
Compressing a set of unordered points is far more challenging than compressing images/videos of regular sample grids, because of the
difficulties in characterizing neighboring relations in an irregular layout of points. Many researchers resort to voxelization to introduce regularity, but this approach suffers from quantization loss. In this research, we use the KNN method to determine the neighborhoods of raw surface points. This gives us a means to determine the spatial context in which the latent features of 3D points are compressed by arithmetic coding. As such, the conditional probability model is adaptive to local geometry, leading to significant rate reduction. Additionally, we propose a dual-layer architecture where a non-learning base layer reconstructs the main structures
of the point cloud at low complexity, while a learned refinement layer focuses on preserving fine details. This design leads to reductions in model complexity and coding latency by two orders of magnitude compared to SOTA methods. Moreover, we incorporate an implicit neural representation (INR) into the refinement layer, allowing the decoder to sample points on the underlying surface at arbitrary densities. This work is the first to effectively exploit content-aware local contexts for compressing irregular raw point clouds, achieving high rate-distortion performance, low complexity, and the ability to function as an arbitrary-scale upsampling network simultaneously.
# Installation
* Install pytorch3d. Please the official installation [instructions](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)

* Then install the following packages
  ```bash
  pip install open3d
  pip install compressai
  ````
* Install Draco. Please follow the official [instructions](https://github.com/google/draco) to install Draco.

> **Note:** The code has been tested on `Ubuntu 22.04.1 LTS` with `Python 3.9`, `PyTorch 1.12.1`, `CUDA 11.6`, `CompressAI 1.2.6`, `PyTorch3D 0.7.0`, and `Open3D 0.17.0`.
# Data Preparation
* Download the ShapeNetCore v1 dataset.
  
* A subset of shapes is sampled from each category. The selected shape instance directories and their splits are listed in the following files:

  - `dataset/split/instance_dir.txt`: List of selected shape instance directories.
  - `dataset/split/train.txt`: Training split.
  - `dataset/split/test.txt`: Test split.
  - `dataset/split/val.txt`: Validation split.
* The selected meshes can be exported to a dedicated directory by running `python dataset/save_selected_mesh.py --mode train`, `--mode test`, or `--mode val` depending on the desired split. 
* Sample a dense set of points from mesh with `bash dataset/build_dataset_shapenet.sh`.
* To avoid applying farthest point sampling (FPS) at each training step, precompute and save the downsampled sparse point clouds in advance by running `python dataset/fps_sampling_points.py`.
  
#  Train
A pretrained model is provided at `result/ex0_hyper_5e_3/checkpoint_best.pth`.  
To reproduce the training of CRCIR, run:
```bash
  python train.py configs/shapenet.yaml --exit-after 30000
````
> **Note:** To avoid overwriting existing results, please update the `out_dir` field in `configs/shapenet.yaml` before starting training.

The training process of CRCIR consists of two stages. In the first stage, the encoder and decoder are trained without a rate constraint. We have provided a pretrained model from this stage in `configs/8D_lr3`. 
If you are interested, you can reproduce the first-stage training using the following two commands. 
```bash
python train_AE.py configs/shapenet_AE.yaml --exit-after 70000
python train_AE.py configs/shapenet_AE2.yaml --exit-after 30000
````
> **Note:** After completing this step, make sure to set the `--AE_path` argument to the path of your newly trained model when running `train.py`.

#  Test Compression Performance
## ShapeNet
```bash
python test_scripts/test_shapenet.py
python eval_p2m_scripts/eval_shapenet_p2m.py
python eval_psnr_scripts/eval_shapenet_psnr.py
````
## Superface
  
* Download the [Florence Superface dataset](https://www.micc.unifi.it/resources/datasets/florence-superface/), which includes 20 `.off` 3D face meshes.
* Sample points from the mesh:
  ```bash
  python other_dataset/sample_points_superface.py
  ````
* Run the test and evaluation scripts:
  ```bash
  python test_scripts/test_superface.py
  python eval_p2m_scripts/eval_superface_p2m.py
  python eval_psnr_scripts/eval_superface_psnr.py
  ````
## DFAUST
* Download the [MPI Dynamic FAUST dataset](https://dfaust.is.tue.mpg.de/index.html). We use a subset of 200 meshes named from `test_scan_000.ply` to `test_scan_199.ply`. 
* Sample points from the mesh:
  ```bash
  python other_dataset/sample_points_dfaust.py
  ````
* Run the test and evaluation scripts:
  ```bash
  python test_scripts/test_dfaust.py
  python eval_p2m_scripts/eval_dfaust_p2m.py
  python eval_psnr_scripts/eval_dfaust_psnr.py
  ````
## LivingRoom
* This test set can be accessed using the [`open3d.data.LivingRoomPointClouds`](https://www.open3d.org/docs/release/python_api/open3d.data.html) module, and contains 57 point clouds.
* Run the test and evaluation scripts:
  ```bash
  python test_scripts/test_living_rooms.py
  python eval_psnr_scripts/eval_living_room_psnr.py
  ````
## Ford
* Download the [Ford Campus Vision and Lidar Data Set](https://robots.engin.umich.edu/SoftwareData/Ford). We use a subset of 200 scans, starting from scan index 1000 to 1200.
* Run the test and evaluation scripts:
  ```bash
  python test_scripts/test_ford.py
  ````
#  Test Upsampling Performance
The proposed CRCIR architecture, although trained solely as a compression network, can also function effectively as a point cloud upsampling network. We evaluate its performance using the PU-GAN dataset.

* Download the [PU-GAN dataset](https://drive.google.com/file/d/1BNqjidBVWP0_MUdMTeGy1wZiR6fqyGmC/view)
* Sample points from the test set using:
  ```bash
  bash other_dataset/pugan.sh
  ````
* Our network supports point cloud upsampling in two modes:

  - **Direct Upsampling during decompression.** During the decompression stage, directly set the decoder-side upsampling rate $u^{\prime}$ to a higher value than the encoder-side downsampling rate $u$. We suggest that $u^{\prime} / u \leq 5$, as overly high ratios may lead to degraded reconstruction quality. 
    ```bash
    python test_scripts/test_pugan_direct_sr.py
    ````
  - **Post-decompression Cascaded Upsampling.** After decompressing the sparse point cloud, we can treat the compression network as a regular cascaded upsampling model. It takes the decompressed data as input and generates a denser point cloud. This helps preserve reconstruction quality at high upsampling ratios. Use the script below to try it out:
    ```bash
    python test_scripts/test_pugan_up_after_comp.py
    python eval_p2m_scripts/eval_pugan_up_after_comp.py
    ````
#  Acknowledgments
Our code is built upon the following repositories: [D-PCC](https://github.com/yunhe20/D-PCC), [Convolutional Occupancy Networks](https://github.com/autonomousvision/convolutional_occupancy_networks), [PyTorch3D](https://github.com/facebookresearch/pytorch3d) and [CompressAI](https://github.com/InterDigitalInc/CompressAI). Thanks for their great work.
#  Citation
If you find our project is useful, please consider citing:
```bibtex
@inproceedings{xu2024fast,
  title={Fast Point Cloud Geometry Compression with Context-based Residual Coding and INR-based Refinement},
  author={Xu, Hao and Zhang, Xi and Wu, Xiaolin},
  booktitle={European Conference on Computer Vision},
  pages={270--288},
  year={2024},
  organization={Springer}
}
