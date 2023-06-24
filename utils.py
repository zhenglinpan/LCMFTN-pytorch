import torch
import torch.nn as nn

def weights_init_normal(model):
    classname = model.__class__.__name__
    if classname.lower().find('conv') != -1:
        nn.init.normal(model.weight.data, 0.0, 0.02)
    elif classname.lower().find('batchnorm2d') != -1:
        nn.init.normal(model.weight.data, 1.0, 0.02)
        nn.init.constant(model.weight.data, 0.0)

class LambdaLR():
    def __init__(self, epoch_end, offset, decay_start):
        assert((epoch_end - decay_start) > 0)
        self.epoch_end = epoch_end
        self.offset = offset
        self.decay_start = decay_start
        
    def step(self, epoch):
        return 1.0 - max(0.0, epoch + self.offset - self.decay_start) / (self.epoch_end - self.decay_start)