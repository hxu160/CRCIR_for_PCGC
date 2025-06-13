import argparse
import torch
import os
import time
import math
from src import config
import shutil
#   python test.py result/ex0_hyper_5e_3/config.yaml

from torch.utils.tensorboard import SummaryWriter
parser = argparse.ArgumentParser(
    description='Train a 3D reconstruction model.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--test_val', type=str, default='test', help='test or val.')
parser.add_argument('--gt_path', type=str, default='other_dataset/pugan/test', help='test or val.')
parser.add_argument('--gt_path_for_sr', type=str, default='other_dataset/pugan/gt_2x', help='test or val.')
parser.add_argument('--draco_path', type=str, default='draco/build_dir', help='path to the draco codec.')
parser.add_argument('--save_dir_name', type=str, default='shapenet/shapenet_draco_qp9', help='path to draco decompressed point clouds.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--K_e', default=48, type=int, help='downsampling ratio')
parser.add_argument('--K_d', default=48, type=int, help='upsampling ratio')
parser.add_argument('--QP', default=9, type=int, help='the quantization precision of base layer, 8 or 9')
args = parser.parse_args()
cfg = config.load_config(args.config)
'''
    Whether use GPU
'''
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
print('GPU is available: %s'%(str(is_cuda)))
device = torch.device("cuda" if is_cuda else "cpu")

encoder = config.get_encoder(cfg)
nparameters_encoder = sum(p.numel() for p in encoder.parameters())
print('Total number of parameters (encoder): %d' % nparameters_encoder)

decoder = config.get_decoder(cfg)
nparameters_decoder = sum(p.numel() for p in decoder.parameters())
print('Total number of parameters (decoder): %d' % nparameters_decoder)

compressor = config.get_compressor(cfg)
nparameters_compressor = sum(p.numel() for p in compressor.parameters())
print('Total number of parameters (compressor): %d' % nparameters_compressor)
checkpoint = torch.load(os.path.join(cfg['training']['out_dir'], 'checkpoint_best.pth'))
encoder.load_state_dict(checkpoint['encoder'])
decoder.load_state_dict(checkpoint['decoder'])
compressor.load_state_dict(checkpoint['compressor'])

encoder = encoder.to(device)
decoder = decoder.to(device)
compressor = compressor.to(device)
compressor.entropy_bottleneck.update(force=True)
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64
#https://github.com/InterDigitalInc/CompressAI/blob/master/compressai/models/base.py
def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    """Returns table of logarithmically scales."""
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))
scale_table = get_scale_table()
if cfg['model']['compressor'] == 'hyper':
    compressor.gaussian_conditional.update_scale_table(scale_table)
    compressor.gaussian_conditional.update()

out_dir = os.path.join(cfg['training']['out_dir'], '%s_%d_%d_%s'%(args.save_dir_name, args.K_e, args.K_d, args.test_val))
os.makedirs(out_dir, exist_ok = True)
print('The output dir %s has been created'%(out_dir))
if not cfg['resume']:
    shutil.copyfile(args.config, os.path.join(out_dir, 'config.yaml'))
logger = SummaryWriter(out_dir)

it = 0

trainer = config.get_trainer(encoder, decoder, compressor, None, None, cfg, device)

eval_dict = trainer.test_direct_sr_performance(out_dir, args)
for k, v in eval_dict.items():
    logger.add_scalar('val/%s' % k, v, it)
logger.close()
        
