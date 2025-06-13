import yaml
import torch
from torch.nn import DataParallel
from dataset.point_clouds import PointCloud_Dataset
from src.module.decoder import LocalDecoder
from src.module.interp_attention import InterpAttentionNet
from src.trainer import training
from src.trainer import training_AE
from src.module.compressor_hyper import HyperCompressor
from src.module.compressor_ffp import FFPCompressor
import torch.optim as optim

encoder_dict = {
    'attention': InterpAttentionNet
}
decoder_dict = {
    'onet': LocalDecoder
}
compressor_dict = {
    'hyper': HyperCompressor,
    'ffp': FFPCompressor
}
def load_config(path):
    ''' Loads config file.

    Args:  
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg
    
def get_encoder(cfg):
    ''' Returns the encoder instance
    
    Args:
        cfg (dict): config dictionary
    '''
    encoder = encoder_dict[cfg['model']['encoder']](**cfg['model']['encoder_kwargs'])
    return encoder

def get_compressor(cfg):
    return compressor_dict[cfg['model']['compressor']](cfg)
    #return Compressor(cfg)

def get_decoder(cfg):
    ''' Returns the decoder instance
    
    Args:
        cfg (dict): config dictionary
    '''
    decoder = cfg['model']['decoder']
    dim = cfg['data']['dim']
    c_dim = cfg['model']['encoder_kwargs']['out_channels']
    decoder_kwargs = cfg['model']['decoder_kwargs']
    decoder = decoder_dict[decoder](
        dim=dim, c_dim=c_dim,
        **decoder_kwargs
    )
    return decoder

def load_model_AE(cfg, pretrained, checkpoint, encoder, decoder):
    if checkpoint is not None:
        print('loading models from previous checkpoint')
        encoder.load_state_dict(checkpoint["encoder"])
        decoder.load_state_dict(checkpoint["decoder"])
    elif pretrained is not None:
        print('loading from pretrained models')
        encoder.load_state_dict(pretrained["encoder"])
        decoder.load_state_dict(pretrained["decoder"])
    else:
        encoder = encoder
        decoder = decoder
    return encoder, decoder

def load_model(cfg, pretrained, checkpoint, encoder, decoder, compressor):
    if checkpoint is not None:
        print('loading models from previous checkpoint')
        encoder.load_state_dict(checkpoint["encoder"])
        decoder.load_state_dict(checkpoint["decoder"])
        compressor.load_state_dict(checkpoint["compressor"])
    elif pretrained is not None:
        print('loading from pretrained models')
        encoder.load_state_dict(pretrained["encoder"])
        decoder.load_state_dict(pretrained["decoder"])
        compressor.load_state_dict(pretrained["compressor"])
    else:
        encoder = encoder
        decoder = decoder
        compressor = compressor
    return encoder, decoder, compressor

def config_model_AE_device(cfg, is_cuda, encoder, decoder):
    if is_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        if torch.cuda.device_count() > 1:
            encoder = DataParallel(encoder)
            decoder = DataParallel(decoder)
    return encoder, decoder

def config_model_device(cfg, is_cuda, encoder, decoder, compressor):
    if is_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        compressor = compressor.cuda()
        if torch.cuda.device_count() > 1:
            encoder = DataParallel(encoder)
            decoder = DataParallel(decoder)
            compressor = DataParallel(compressor)
    return encoder, decoder, compressor

def config_optim(cfg, encoder, decoder, compressor):
    learning_rate = cfg['training']['lr']
    print('The learning rate is %f' %(learning_rate))
    parameters = list(p for n, p in compressor.named_parameters() if not n.endswith(".quantiles"))
    aux_parameters = set(p for n, p in compressor.named_parameters() if n.endswith(".quantiles"))
    optimizer = optim.Adam(
        [{'params': encoder.parameters()}, 
        {'params': decoder.parameters()},
        {'params': parameters}], 
        lr = learning_rate)
    aux_optimizer = optim.Adam(aux_parameters, learning_rate)
    return optimizer, aux_optimizer

def config_AE_optim(cfg, encoder, decoder):
    learning_rate = cfg['training']['lr']
    print('The learning rate is %f' %(learning_rate))
    optimizer = optim.Adam(
        [{'params': encoder.parameters()}, 
        {'params': decoder.parameters()}], 
        lr = learning_rate)
    return optimizer

def get_dataset(cfg, gt_path, path, mode):
    return PointCloud_Dataset(cfg, gt_path, path, mode)

def get_AE_trainer(encoder, decoder, optimizer, cfg, device):
    return training_AE.AE_Trainer(encoder, decoder, optimizer, cfg, device=device)

def get_trainer(encoder, decoder, compressor, optimizer, aux_optimizer, cfg, device):
    return training.Trainer(encoder, decoder, compressor, optimizer, aux_optimizer, cfg, device=device)