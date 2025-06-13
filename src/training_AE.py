import numpy as np
from collections import defaultdict
from tqdm import tqdm

class BaseAETrainer(object):
    ''' 
        Base trainer class.
    '''

    def evaluate(self, val_set, upsampling_ratio = 34):
        ''' Performs an evaluation.
        Args:
            val_set (Dataset): pytorch Dataset
        '''
        eval_list = defaultdict(list)
        num_shapes = len(val_set)
        with tqdm(total=num_shapes) as pbar:
            for i in range(num_shapes):
                eval_step_dict = self.eval_step(val_set[i], upsampling_ratio)
                for k, v in eval_step_dict.items():
                    eval_list[k].append(v)
                pbar.update(1)
        eval_dict = {k: np.mean(v) for k, v in eval_list.items()}
        return eval_dict
    
    
    def train_step(self, *args, **kwargs):
        ''' Performs a training step.
        '''
        raise NotImplementedError
                