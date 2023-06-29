import torch
import torch.nn as nn
import torch.autograd.variable as Variable
from torchvision.models import vgg19

class LCMFTNGenerator(nn.Module):
    def __init__(self, args):
        super(LCMFTNGenerator, self).__init__()
        self.Eh = Eh(args.s_channel)
        self.Ec = Ec(args.c_channel)
        self.Es = Es(args.s_channel)
        self.RU0 = RU(512, 256, 20)
        self.RU1 = RU(256, 128, 10)
        self.RU2 = RU(128, 64, 10)
        self.CMFT0 = CMFT()  # TODO: CMFT don't need grad
        self.CMFT1 = CMFT()
        self.CMFT2 = CMFT()
        # self.CMFT3 = CMFT()   # skipped as per paper
        self.cat = cat()
        
        self.conv_out = nn.Sequential(nn.Conv2d(64, args.c_channel, 1))
        
    def forward(self, Sn, Sp, Cp, EIp, EIn):
        Eh3, Eh2, Eh1, Eh0 = self.Eh(Sn)
        # print(Eh3.shape, Eh2.shape, Eh1.shape, Eh0.shape)
        
        Ec3, Ec2, Ec1, Ec0 = self.Ec(Cp)
        # print(Ec3.shape, Ec2.shape, Ec1.shape, Ec0.shape)
        
        Es3p, Es2p, Es1p, Es0p = self.Es(Sp)
        # print(Es3p.shape, Es2p.shape, Es1p.shape, Es0p.shape)
        
        Es3n, Es2n, Es1n, Es0n = self.Es(Sn)
        # print(Es3n.shape, Es2n.shape, Es1n.shape, Es0n.shape)

        # print("0")
        x = self.CMFT0(self.cat(EIn, Es0n), self.cat(EIp, Es0p), Ec0)
        # print("1")
        x = self.RU0(self.cat(x, Eh0))
        # print('x', x.shape)
        # print("2")
        x = self.CMFT1(self.cat(x, Es1n), self.cat(Ec1, Es1p), Ec1)
        # print("3")
        x = self.RU1(self.cat(x, Eh1))
        # print("4")
        x = self.CMFT2(self.cat(x, Es2n), self.cat(Ec2, Es2p), Ec2)
        # print("5")
        x = self.RU2(self.cat(x, Eh2))
        # print("6")
        # x = self.CMFT3(self.cat(x, Es3n), self.cat(Ec3, Es3p), Ec3)
        # print("7")
        x = self.conv_out(self.cat(x, Eh3))
        # print("8")
        return x

class cat():
    def __init__(self,):
        pass
        
    def __call__(self, a, b):
        # print('a', a.shape, 'b', b.shape)
        # x = torch.cat((a, b), dim=1)
        # x = a + b     # create tuple, wrong
        x = torch.add(a, b)
        return x

class RU(nn.Module):
    def __init__(self, nc_in, nc_out, res_n):
        super(RU, self).__init__()
        self.nc_in = nc_in
        self.nc_out = nc_out
        
        model = [ResNeXtBlock(nc_in, nc_in) for _ in range(res_n)] + [ESPCN(nc_in, nc_out)]
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)


class ResNeXtBlock(nn.Module):
    """
    Alternative: https://github.com/prlz77/ResNeXt.pytorch/blob/master/models/model.py
    """
    def __init__(self, nc_in, nc_out, cardinality=32):
        super(ResNeXtBlock, self).__init__()
        
        nc_hidden = 32
        # nc_hidden = nc_in // 2
        self.conv_in = nn.Sequential(nn.Conv2d(nc_in, nc_hidden, 1),
                                     nn.BatchNorm2d(nc_hidden),
                                     nn.ReLU(1))
        self.bottleneck = nn.Sequential(nn.Conv2d(nc_hidden, nc_hidden, 3, 1, 1, groups=cardinality))
        self.conv_out = nn.Sequential(nn.Conv2d(nc_hidden, nc_out, 1),
                                      nn.BatchNorm2d(nc_out))
        
    def forward(self, x):
        x1 = self.conv_in(x)
        x2 = self.bottleneck(x1)
        x3 = self.conv_out(x2)
        return x + x3


class ESPCN(nn.Module):
    """
    Alternative: https://github.com/Lornatang/ESPCN-PyTorch/blob/master/model.py
    """
    def __init__(self, nc_in, nc_out, upscale_factor=2):
        super(ESPCN, self).__init__()
        self.nc_in = nc_in
        self.nc_out = nc_out
        self.upscale_factor = upscale_factor
        
        conv = [nn.Conv2d(self.nc_in, 64, 5, 1, 2),
                nn.ReLU(1),
                nn.Conv2d(64, 32, 3, 1, 1),
                nn.ReLU(1)]
        
        subpixel_conv = [nn.Conv2d(32, int(self.nc_out * (self.upscale_factor ** 2)), 3, 1, 1),
                         nn.ReLU(1),
                         nn.PixelShuffle(upscale_factor)]
        
        self.conv = nn.Sequential(*conv)
        self.subpixel_conv = nn.Sequential(*subpixel_conv)

    def forward(self, x):
        x = self.conv(x)
        x = self.subpixel_conv(x)
        # x = torch.clamp_(x, 0.0, 1.0)
        return x


class CMFT(nn.Module):
    def __init__(self):
        super(CMFT, self).__init__()
        
    def forward(self, fxA, fyA, fyB):
        assert fxA.shape == fyA.shape == fyB.shape
        N, C, H, W = fxA.shape[:]
        CA = torch.exp(fxA.view(N, C, H*W, 1) @ fyA.view(N, C, 1, H*W))
        X = CA @ fyB.view(N, C, H*W, 1)
        X = X.view(N, C, H, W)
        return X


class Eh(nn.Module):
    def __init__(self, nc_in):
        super(Eh, self).__init__()
        self.nc_in = nc_in
        
        self.conv1 = nn.Sequential(nn.Conv2d(self.nc_in, 64, 7, 1, 3), 
                                   nn.BatchNorm2d(64), 
                                   nn.ReLU(1))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), 
                                   nn.BatchNorm2d(128), 
                                   nn.ReLU(1))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1), 
                                   nn.BatchNorm2d(256), 
                                   nn.ReLU(1))
        self.conv4 = nn.Sequential(nn.Conv2d(256, 512, 4, 2, 1), 
                                   nn.BatchNorm2d(512), 
                                   nn.ReLU(1))
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        return x1, x2, x3, x4
        

class Ec(nn.Module):
    def __init__(self, nc_in):
        super(Ec, self).__init__()
        self.nc_in = nc_in
        
        self.conv1 = nn.Sequential(nn.Conv2d(self.nc_in, 64, 7, 1, 3), 
                                   nn.BatchNorm2d(64), 
                                   nn.ReLU(1))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), 
                                   nn.BatchNorm2d(128), 
                                   nn.ReLU(1))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1), 
                                   nn.BatchNorm2d(256), 
                                   nn.ReLU(1))
        self.conv4 = nn.Sequential(nn.Conv2d(256, 512, 4, 2, 1), 
                                   nn.BatchNorm2d(512), 
                                   nn.ReLU(1))
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        return x1, x2, x3, x4


class Es(nn.Module):
    def __init__(self, nc_in):
        super(Es, self).__init__()
        self.nc_in = nc_in
        
        self.conv1 = nn.Sequential(nn.Conv2d(self.nc_in, 64, 7, 1, 3), 
                                   nn.BatchNorm2d(64), 
                                   nn.ReLU(1))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 6, dilation=4), # dilation: ⌊(W-F'+2*P)/S⌋+1  F'=F+(D-1)(F-1)
                                   nn.BatchNorm2d(128), 
                                   nn.ReLU(1))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 12, dilation=8), 
                                   nn.BatchNorm2d(256), 
                                   nn.ReLU(1))
        self.conv4 = nn.Sequential(nn.Conv2d(256, 512, 4, 2, 12, dilation=8), 
                                   nn.BatchNorm2d(512), 
                                   nn.ReLU(1))
        self.conv5 = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1), 
                                   nn.BatchNorm2d(512), 
                                   nn.ReLU(1))
        self.conv6 = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU(1))
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        return x1, x2, x3, x6

class EI(nn.Module):
    ### should be Illustration2Vet Network
    ### Use Conv for substitue
    def __init__(self, nc_in):
        super(EI, self).__init__()
        self.nc_in = nc_in
        
        model = [nn.Conv2d(self.nc_in, 64, 7, 1, 3),
                 nn.BatchNorm2d(64),
                 nn.ReLU(1)]
        
        in_features = 64
        out_features = in_features * 2
        for _ in range(3):
            model += [nn.Conv2d(in_features, out_features, 4, 2, 1),
                      nn.BatchNorm2d(out_features),
                      nn.LeakyReLU(0.2, 1)]
            in_features = out_features
            out_features = in_features * 2

        self.model = nn.Sequential(*model)
        
    def forward(self, x):
        return self.model(x)


class VGGDiscriminator(nn.Module):
    def __init__(self) :
        super(VGGDiscriminator, self).__init__()
        self.index = [0, 2, 5, 10, 14]
        features = list(vgg19(weights='DEFAULT').features)[:15]
        self.features = nn.ModuleList(features).eval()
        
    def forward(self, x):
        xs = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.index:
                xs.append(x)
        return xs
