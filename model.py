import torch.nn as nn
import trainer
import torch.nn.functional as F

nClass = trainer.nClass
nNoise = trainer.nNoise
magnitudeSize = trainer.magnitudeSize
nc = trainer.nc


class FrontEnd(nn.Module):
    ''' front end part of discriminator and Q'''

    def __init__(self):
        super(FrontEnd, self).__init__()
        self.nc = nc
        self.main = nn.Sequential(
            # nc*64*64
            nn.Conv2d(self.nc, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # 64*32*32
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 128*16*16
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 256*8*8
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # 512*4*4
            # nn.Conv2d(512, 1024, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(1024),
            # nn.LeakyReLU(0.2, inplace=True),
            # 1024*1*1
        )

    def forward(self, x):
        output = self.main(x)
        return output


class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(512, nClass + 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()
        )

    def forward(self, x):
        output = self.main(x).view(-1, nClass + 1).squeeze()
        return output

class D_Mag(nn.Module):
    def __init__(self):
        super(D_Mag, self).__init__()

        self.main = nn.Sequential(
            # 1*32*32
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # 64*16*16
            nn.Conv2d(64, 128, 4, 2, 1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 128*8*8
            nn.Conv2d(128, nClass + 1 , 4, 2, 0, bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            #256*4*4
            nn.Conv2d(256, nClass + 1, 4, 1, 0, bias=True),
            # nClass+1 * 1*1
        )

    def forward(self, x):
        output = self.main(x).view(-1, nClass + 1).squeeze()
        return output

class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.nc = nc
        self.main = nn.Sequential(
            # input is #class+noise vector.
            nn.ConvTranspose2d(nClass + nNoise, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 512*4*4
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 256*8*8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 128*16*16
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 64*32*32
            nn.ConvTranspose2d(64, self.nc, 4, 2, 1, bias=False),
            # 3*64*64
            nn.Tanh()  # dcgan used tanh here
            # nn.Sigmoid()
        )

    def forward(self, x):
        output = self.main(x)
        return output


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
