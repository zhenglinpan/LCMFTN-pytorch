import os, sys
import numpy as np

import torch
import torchvision.models as models
from torchvision.utils import save_image
from torch.autograd import Variable


from module import LCMFTNGenerator, EI
from utils import weights_init_normal
from utils import LambdaLR
from args import args


DEVICE = 0

generator = LCMFTNGenerator().to(DEVICE)
feature_extractor = EI().to(DEVICE)
discriminator_vgg = models.vgg19(pretrained=True)

feature_extractor.eval()
discriminator_vgg.eval()

generator.apply(weights_init_normal)

criterion_color = torch.nn.L1Loss()
criterion_perceptual = torch.nn.MSELoss()

optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(args.end_epochs, args.start_epoch, args.decay_epoch).step)

Tensor = torch.cuda.FloatTensor
Input = Variable(args.batch_size, args.in_channel, args.input_size, args.input_size)


