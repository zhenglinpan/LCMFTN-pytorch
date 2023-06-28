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

# import wandb
# wandb.init(project="LCMFTN")

DEVICE = 0

generator = LCMFTNGenerator(args).to(DEVICE)
feature_extractor = EI(nc_in=1).to(DEVICE)
discriminator = VGGDiscriminator().to(DEVICE)

feature_extractor.eval()
discriminator.eval()

generator.apply(weights_init_normal)

criterion_color = torch.nn.L1Loss()
criterion_MSE = torch.nn.MSELoss()

optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(args.end_epoch, args.start_epoch, args.decay_epoch).step)

Tensor = torch.cuda.FloatTensor
Input_C = Tensor(args.batch_size, args.c_channel, args.input_height, args.input_width)
Input_S = Tensor(args.batch_size, args.s_channel, args.input_height, args.input_width)

trans = [transforms.ToPILImage(),
         transforms.Resize(size=(args.input_height, args.input_width), interpolation=transforms.InterpolationMode.BICUBIC),
         transforms.ToTensor()]
train_dataset = AnimeDataset(args, trans=trans)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu)
print("Data Loaded====================>")

###### Training ######
for epoch in tqdm(range(args.start_epoch, args.end_epoch + 1)):
    for i, batch in enumerate(train_loader):
        
        Sn = Variable(Input_S.copy_(batch['Sn'])).to(DEVICE)
        Sp = Variable(Input_S.copy_(batch['Sp'])).to(DEVICE)
        Cn = Variable(Input_C.copy_(batch['Cn'])).to(DEVICE)
        Cp = Variable(Input_C.copy_(batch['Cp'])).to(DEVICE)
        
        optimizer_G.zero_grad()
        
        EIp = feature_extractor(Sp)
        EIn = feature_extractor(Sn)
        
        pred_Cn = generator(Sn, Sp, Cp, EIp.detach(), EIn.detach())
        
        color_loss = criterion_color(Cn, pred_Cn)
        
        fms_real = discriminator(Cn)
        fms_fake = discriminator(pred_Cn)
        perceptual_loss = criterion_MSE(fms_real[0], fms_fake[0])
        for i in range(1, len(fms_real)):
            perceptual_loss += criterion_MSE(fms_real[i], fms_fake[i])
        
        loss = color_loss + perceptual_loss
        
        # wandb.log({"color_loss": color_loss.item(), "perceptual_loss": loss.item()})
        
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
        torch.save(generator.state_dict(), args.models_root + '/' + str(epoch) + '_generator.pth')
