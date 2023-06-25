import os, sys
import numpy as np

import torch

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable

from args import args
from dataset import AnimeDataset
from torch.utils.data import DataLoader

from module import LCMFTNGenerator, EI, VGGDiscriminator
from utils import weights_init_normal
from utils import LambdaLR

from tqdm import tqdm

DEVICE = 0

generator = LCMFTNGenerator().to(DEVICE)
feature_extractor = EI().to(DEVICE)
discriminator = VGGDiscriminator().to(DEVICE)

feature_extractor.eval()
discriminator.eval()

generator.apply(weights_init_normal)

criterion_color = torch.nn.L1Loss()
criterion_MSE = torch.nn.MSELoss()

optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(args.end_epochs, args.start_epoch, args.decay_epoch).step)

Tensor = torch.cuda.FloatTensor
Input = Tensor(args.batch_size, args.in_channel, args.input_height, args.input_width)

trans = [transforms.Resize(256, 256),
         transforms.ToTensor()]
train_dataset = AnimeDataset(args, trans=trans)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu)
print("Data Loaded====================>")

###### Training ######
for epoch in tqdm(range(args.start_epoch, args.end_epoch)):
    for i, batch in enumerate(train_loader):
        Sn = Variable(Input.copy_(batch['Sn'])).to(DEVICE)
        Sp = Variable(Input.copy_(batch['Sp'])).to(DEVICE)
        Cn = Variable(Input.copy_(batch['Cn'])).to(DEVICE)
        Cp = Variable(Input.copy_(batch['Cp'])).to(DEVICE)
        
        optimizer_G.zero_grad()
        
        pred_Cn = LCMFTNGenerator(args, Sn, Sp, Cp)
        
        color_loss = criterion_color(Cn, pred_Cn)
        
        fms_real = discriminator(Cn)
        fms_fake = discriminator(pred_Cn)
        perceptual_loss = criterion_MSE(fms_real[0], fms_fake[0])
        for i in range(1, len(fms_real)):
            perceptual_loss += criterion_MSE(fms_real[i], fms_fake[i])
        
        loss = color_loss + perceptual_loss
        
        loss.backward()
        optimizer_G.step()
        
    lr_scheduler_G.step()
    
    if epoch % 5 == 0:
        save_image(Sn, os.path.join('./results/train', str(epoch) + 'Sn.jpg'))
        save_image(Sp, os.path.join('./results/train', str(epoch) + 'Sp.jpg'))
        save_image(Cn, os.path.join('./results/train', str(epoch) + 'Cn.jpg'))
        save_image(Cp, os.path.join('./results/train', str(epoch) + 'Cp.jpg'))
        save_image(pred_Cn, os.path.join('./results/train', str(epoch) + 'pred_Cn.jpg'))

    if epoch % 10 == 0:
        torch.save(LCMFTNGenerator.state_dict(), args.models_root + '/' + str(epoch) + 'generator.pth')
