import argparse
import torch
import numpy as np
import os
import time
import datetime
from src import config
from src.common import save_model
from torch.utils.data import DataLoader
import shutil

#   python train.py configs/shapenet.yaml --exit-after 30000

from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(
    description='Train a 3D reconstruction model.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--test_val', type=str, default='val', help='test or val.')
parser.add_argument('--AE_path', type=str, default='configs/8D_lr3/checkpoint_best.pth', help='test or val.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--exit-after', type=int, default=-1,
                    help='Checkpoint and exit after specified number of seconds'
                         'with exit code 2.')

args = parser.parse_args()
cfg = config.load_config(args.config)
#   whether use GPU.
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
print('GPU is available: %s'%(str(is_cuda)))
device = torch.device("cuda" if is_cuda else "cpu")
model_selection_metric = cfg['training']['model_selection_metric']
if cfg['training']['model_selection_mode'] == 'maximize':
    model_selection_sign = 1
elif cfg['training']['model_selection_mode'] == 'minimize':
    model_selection_sign = -1
else:
    raise ValueError('model_selection_mode must be '
                     'either maximize or minimize.')
metric_val_best = -model_selection_sign * np.inf
print('Current model selection mode: %s'%(cfg['training']['model_selection_mode']))
print('Current best validation metric (%s): %.8f' % (model_selection_metric, metric_val_best))

#   config encoder
encoder = config.get_encoder(cfg)
nparameters_encoder = sum(p.numel() for p in encoder.parameters())
print('Total number of parameters (encoder): %d' % nparameters_encoder)
#   config decoder
decoder = config.get_decoder(cfg)
nparameters_decoder = sum(p.numel() for p in decoder.parameters())
print('Total number of parameters (decoder): %d' % nparameters_decoder)
#   config entropy model
compressor = config.get_compressor(cfg)
nparameters_compressor = sum(p.numel() for p in compressor.parameters())
print('Total number of parameters (compressor): %d' % nparameters_compressor)

out_dir = cfg['training']['out_dir']
os.makedirs(out_dir, exist_ok = True)
print('The output dir %s has been created'%(out_dir))
if not cfg['resume']:
    shutil.copyfile(args.config, os.path.join(out_dir, 'config.yaml'))
summaries_dir = os.path.join(cfg['training']['out_dir'], 'log')
os.makedirs(summaries_dir, exist_ok = True)
logger = SummaryWriter(summaries_dir)
if cfg['training']['from_pretrained']:
    pretrained = torch.load(cfg['training']['pretrained_path'])
else:
    pretrained = None
    
if cfg['resume'] and os.path.exists(out_dir):
    checkpoint = torch.load(os.path.join(out_dir, "checkpoint.pth"))
else:
    checkpoint = None
    
encoder.load_state_dict(torch.load(args.AE_path)['encoder'])
decoder.load_state_dict(torch.load(args.AE_path)['decoder'])

encoder, decoder, compressor = config.load_model(cfg, pretrained, checkpoint, encoder, decoder, compressor)
encoder, decoder, compressor = config.config_model_device(cfg, is_cuda, encoder, decoder, compressor)
optimizer, aux_optimizer = config.config_optim(cfg, encoder, decoder, compressor)

epoch_it = 0
it = 0

if cfg['resume'] and os.path.exists(out_dir):
    optimizer.load_state_dict(checkpoint["optimizer"])
    epoch_it = checkpoint["epoch_it"]
    it = checkpoint["it"]
    metric_val_best = checkpoint["loss_val_best"]

print('The learning rate is %f' %(cfg['training']['lr']))
batch_size = cfg['training']['batch_size']
backup_every = cfg['training']['backup_every']
print_every = cfg['training']['print_every']
checkpoint_every = cfg['training']['checkpoint_every']
validate_every = cfg['training']['validate_every']
exit_after = args.exit_after
print('The code will be stoped after %ds'%(exit_after))


gt_path = cfg['data']['gt_path']
data_path = cfg['data']['path']
t0 = time.time()
train_set = config.get_dataset(cfg, gt_path, data_path, 'train')
val_set = config.get_dataset(cfg, gt_path, data_path, args.test_val)
print('There are %d and %d instances in the training and validation set respectively'%(len(train_set), len(val_set)))
train_loader = DataLoader(train_set, batch_size, shuffle=True, pin_memory=True, num_workers=8)



trainer = config.get_trainer(encoder, decoder, compressor, optimizer, aux_optimizer, cfg, device=device)
t0 = time.time()

while True:
    epoch_it += 1
    for batch in train_loader:
        it += 1
        
        loss, distortion, bpp, aux_loss = trainer.train_step(batch)
        logger.add_scalar('train/loss', loss, it)
        logger.add_scalar('train/distortion', distortion, it)
        logger.add_scalar('train/bpp', bpp, it)
        logger.add_scalar('train/aux_loss', aux_loss, it)
        if print_every > 0 and (it % print_every) == 0:
            t = datetime.datetime.now()
            print('[Epoch %02d] it=%03d, loss=%.5f, time: %.2fs, %02d:%02d'
                     % (epoch_it, it, loss, time.time() - t0, t.hour, t.minute))
        
        # Save checkpoint
        if (checkpoint_every > 0 and (it % checkpoint_every) == 0):
            print('Saving checkpoint')
            save_model(epoch_it, it, metric_val_best, optimizer, encoder, decoder, compressor, aux_optimizer, os.path.join(out_dir, "checkpoint.pth"))

        # Backup if necessary
        if (backup_every > 0 and (it % backup_every) == 0):
            print('Backup checkpoint')
            save_model(epoch_it, it, metric_val_best, optimizer, encoder, decoder, compressor, aux_optimizer, os.path.join(out_dir, "checkpoint_%d.pth" % it))
        
        if validate_every > 0 and (it % validate_every) == 0:
            eval_dict = trainer.evaluate(val_set)
            metric_val = eval_dict[model_selection_metric]
            print('Validation metric (%s): %.5f'
                  % (model_selection_metric, metric_val))

            for k, v in eval_dict.items():
                logger.add_scalar('val/%s' % k, v, it)
            
            if model_selection_sign * (metric_val - metric_val_best) > 0:
                metric_val_best = metric_val
                print('New best model (loss %.5f)' % metric_val_best)
                save_model(epoch_it, it, metric_val_best, optimizer, encoder, decoder, compressor, aux_optimizer, os.path.join(out_dir, "checkpoint_best.pth"))


        if exit_after > 0 and (time.time() - t0) >= exit_after:
            print('Time limit reached. Exiting.')
            logger.close()
            save_model(epoch_it, it, metric_val_best, optimizer, encoder, decoder, compressor, aux_optimizer, os.path.join(out_dir, "checkpoint.pth"))
            exit(3)
    
        
