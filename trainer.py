import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import numpy as np
# for loss plot
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# set the dataroot, which is a folder
dataset_type = 'lsun'
dataroot = '/home/zeyuan/lsun/lmdb'
lsun_classes=['conference_room_train', 'bridge_train']
# dataset_type = 'folder'
# dataroot = '/home/zeyuan/dataset/places'
# dataset_type = 'mnist'
# dataroot = '/home/zeyuan/dataset/mnist'
outf = './checkpoints/lsun/bridge_conference_room.0002'

workers = 2
imageSize = 64
nDisC = 2
nConC = 2
nNoise = 100
nEpoch = 100
nCPerRow = 10
lr_D = 0.0002
lr_G = 0.0002
normalizeImage = True

try:
    os.makedirs(outf)
except OSError:
    pass


class log_gaussian:
    def __call__(self, x, mu, var):
        logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - \
                (x - mu).pow(2).div(var.mul(2.0) + 1e-6)

        return logli.sum(1).mean().mul(-1)


class Trainer:
    def __init__(self, G, FE, D, Q):

        self.G = G
        self.FE = FE
        self.D = D
        self.Q = Q

        self.batch_size = 100

    def _noise_sample(self, dis_c, con_c, noise, bs):

        idx = np.random.randint(nDisC, size=bs)
        c = np.zeros((bs, nDisC))
        c[range(bs), idx] = 1.0

        dis_c.data.copy_(torch.Tensor(c))
        con_c.data.uniform_(-1.0, 1.0)
        noise.data.uniform_(-1.0, 1.0)
        z = torch.cat([noise, dis_c, con_c], 1).view(-1, nDisC + nConC + nNoise, 1, 1)

        return z, idx

    def draw(self, figTitle, fileName, data1, data2):
        plt.figure(figsize=(30, 10))
        plt.title(figTitle)
        plt.xlabel('iterations')
        plt.ylabel('loss')
        plt.plot(np.arange(len(data1)), data1, 'r')
        plt.plot(np.arange(len(data2)), data2, 'b')
        plt.savefig(fileName)
        plt.close()

    def train(self):
        real_x = torch.FloatTensor(self.batch_size, 1, imageSize, imageSize).cuda()
        label = torch.FloatTensor(self.batch_size).cuda()
        dis_c = torch.FloatTensor(self.batch_size, nDisC).cuda()
        con_c = torch.FloatTensor(self.batch_size, nConC).cuda()
        noise = torch.FloatTensor(self.batch_size, nNoise).cuda()

        real_x = Variable(real_x)
        label = Variable(label, requires_grad=False)
        dis_c = Variable(dis_c)
        con_c = Variable(con_c)
        noise = Variable(noise)

        criterionD = nn.BCELoss().cuda()
        criterionQ_dis = nn.CrossEntropyLoss().cuda()
        criterionQ_con = log_gaussian()

        optimD = optim.Adam([{'params': self.FE.parameters()}, {'params': self.D.parameters()}], lr=lr_D,
                            betas=(0.5, 0.999))
        optimG = optim.Adam([{'params': self.G.parameters()}, {'params': self.Q.parameters()}], lr=lr_G,
                            betas=(0.5, 0.999))

        if dataset_type == 'lsun':
            dataset = dset.LSUN(db_path=dataroot, classes=lsun_classes,
                                transform=transforms.Compose([
                                    transforms.Scale(imageSize),
                                    transforms.CenterCrop(imageSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))
        elif dataset_type == 'folder':
            dataset = dset.ImageFolder(root=dataroot,
                                       transform=transforms.Compose([
                                           transforms.Scale(imageSize),
                                           transforms.CenterCrop(imageSize),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                       ]))
        elif dataset_type == 'mnist':
            dataset = dset.MNIST(root=dataroot,
                                 download=True,
                                 transform=transforms.Compose([
                                     transforms.Scale(imageSize),
                                     transforms.CenterCrop(imageSize),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                 ]))
            # dataset = dset.MNIST(dataroot, transform=transforms.ToTensor())

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=workers)

        # fixed random variables
        c = np.linspace(-1, 1, nCPerRow).reshape(1, -1)
        c = np.repeat(c, 10, 0).reshape(-1, 1)

        c1 = np.hstack([c, np.zeros_like(c)])
        c2 = np.hstack([np.zeros_like(c), c])

        idx = np.arange(nDisC).repeat(100 / nDisC)
        one_hot = np.zeros((100, nDisC))
        one_hot[range(100), idx] = 1
        fix_noise = torch.Tensor(100, nNoise).uniform_(-1, 1)

        D_loss_overall = np.zeros(nEpoch)
        G_loss_overall = np.zeros(nEpoch)
        for epoch in range(nEpoch):
            start_time = time.time()
            D_loss_epoch = np.zeros(len(dataloader))
            G_loss_epoch = np.zeros(len(dataloader))
            for num_iters, batch_data in enumerate(dataloader, 0):
                # real part
                optimD.zero_grad()
                x, _ = batch_data
                bs = x.size(0)
                real_x.data.resize_(x.size())
                label.data.resize_(bs)
                dis_c.data.resize_(bs, nDisC)
                con_c.data.resize_(bs, nConC)
                noise.data.resize_(bs, nNoise)

                real_x.data.copy_(x)
                fe_out1 = self.FE(real_x)
                probs_real = self.D(fe_out1)
                label.data.fill_(1)
                loss_real = criterionD(probs_real, label)
                loss_real.backward()

                # fake part
                z, idx = self._noise_sample(dis_c, con_c, noise, bs)
                fake_x = self.G(z)
                fe_out2 = self.FE(fake_x.detach())
                probs_fake = self.D(fe_out2)
                label.data.fill_(0)
                loss_fake = criterionD(probs_fake, label)
                loss_fake.backward()

                D_loss = loss_real + loss_fake

                optimD.step()

                # G and Q part
                optimG.zero_grad()

                fe_out = self.FE(fake_x)
                probs_fake = self.D(fe_out)
                label.data.fill_(0.9)

                reconstruct_loss = criterionD(probs_fake, label)

                q_logits, q_mu, q_var = self.Q(fe_out)
                class_ = torch.LongTensor(idx).cuda()
                target = Variable(class_)
                dis_loss = criterionQ_dis(q_logits, target)
                con_loss = criterionQ_con(con_c, q_mu, q_var) * 0.1

                G_loss = reconstruct_loss + dis_loss + con_loss
                G_loss.backward()
                optimG.step()

                D_loss_epoch[num_iters] = D_loss.data[0]
                G_loss_epoch[num_iters] = G_loss.data[0]
                if num_iters % 10 == 0:
                    # print('Epoch/Iter:{0}/{1}, Dloss: {2}, Gloss: {3}'.format(
                    #     epoch, num_iters, D_loss.data.cpu().numpy(),
                    #     G_loss.data.cpu().numpy())
                    # )
                    print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
                          % (epoch, nEpoch, num_iters, len(dataloader),
                             D_loss.data[0], G_loss.data[0]))

                    # save the randomly generated images
                    save_image(fake_x.data,
                               '%s/fake_samples_latest.png' % outf,
                               normalize=normalizeImage, nrow=10)
                    save_image(fake_x.data,
                               '%s/fake_samples_epoch_%d.png' % (outf, epoch),
                               normalize=normalizeImage, nrow=10)

                    noise.data.copy_(fix_noise)
                    dis_c.data.copy_(torch.Tensor(one_hot))

                    con_c.data.copy_(torch.from_numpy(c1))
                    z = torch.cat([noise, dis_c, con_c], 1).view(-1, nDisC + nConC + nNoise, 1, 1)
                    x_save = self.G(z)
                    save_image(x_save.data,
                               '%s/c1_fake_samples_epoch_%d.png' % (outf, epoch),
                               normalize=normalizeImage, nrow=nCPerRow)
                    save_image(x_save.data,
                               '%s/c1_fake_latest.png' % (outf),
                               normalize=normalizeImage, nrow=nCPerRow)

                    con_c.data.copy_(torch.from_numpy(c2))
                    z = torch.cat([noise, dis_c, con_c], 1).view(-1, nDisC + nConC + nNoise, 1, 1)
                    x_save = self.G(z)
                    save_image(x_save.data,
                               '%s/c2_fake_samples_epoch_%d.png' % (outf, epoch),
                               normalize=normalizeImage, nrow=nCPerRow)
                    save_image(x_save.data,
                               '%s/c2_fake_latest.png' % (outf),
                               normalize=normalizeImage, nrow=nCPerRow)
                    self.draw('Loss of epoch %d' % epoch,
                              '%s/loss_epoch_%d.pdf' % (outf, epoch),
                              D_loss_epoch, G_loss_epoch)

            D_loss_overall[epoch] = D_loss_epoch[-1]
            G_loss_overall[epoch] = G_loss_epoch[-1]
            self.draw('Overall Loss',
                      '%s/loss_overall.pdf' % (outf),
                      D_loss_overall, G_loss_overall)
            torch.save(self.G.state_dict(), '%s/netG_epoch_%d.pth' % (outf, epoch))
            torch.save(self.D.state_dict(), '%s/netD_epoch_%d.pth' % (outf, epoch))
            elapsed_time = time.time() - start_time
            print("Elapsed Time for epoch %d is:%4f" % (epoch, elapsed_time))
