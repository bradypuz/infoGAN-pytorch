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
#
# dataset_type = 'folder'
# dataroot = '/home/zeyuan/dataset/places'

# dataset_type = 'mnist'
# dataroot = '/home/zeyuan/dataset/mnist'

outf = './checkpoints_forget/lsun/conference_room_bridge_L0_5.0_D_1.0'
# outf = './checkpoints_forget/mnist/test_L1_5.0_D_0.7'

nClass = 10

workers = 2
imageSize = 64
nNoise = 100
nc = 3
if dataset_type == 'mnist':
    nc = 1
    nClass = 10
nEpoch = 45
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
    def __init__(self, G, FE, D):

        self.G = G
        self.FE = FE
        self.D = D

        self.batch_size = 100

    def _noise_sample(self, dis_c, noise, bs):

        idx = np.random.randint(nClass, size=bs)
        c = np.zeros((bs, nClass))
        c[range(bs), idx] = 1.0

        dis_c.data.copy_(torch.Tensor(c))
        # con_c.data.uniform_(-1.0, 1.0)
        noise.data.uniform_(-1.0, 1.0)
        z = torch.cat([noise, dis_c], 1).view(-1, nClass + nNoise, 1, 1)

        return z, idx

    def L0_loss(self, images):
        tmp = images.data + 1.0
        tmp[tmp < 0.05] = 0 #threshold
        output = torch.nonzero(tmp)
        num_non_zeros = output.size(0)
        images_size = images.data.size()
        num_pixels = 1
        for i in range(len(images_size)):
            num_pixels *= images_size[i]

        return num_non_zeros/num_pixels #the percentage of non-zero elements


    def draw(self, figTitle, fileName, data1, data2, data3):
        plt.figure(figsize=(30, 10))
        plt.title(figTitle)
        plt.xlabel('iterations')
        plt.ylabel('loss')
        plt.plot(np.arange(len(data1)), data1, 'r')
        plt.plot(np.arange(len(data2)), data2, 'b')
        plt.plot(np.arange(len(data3)), data3, 'g')
        plt.savefig(fileName)
        plt.close()

    def train(self):
        real_x = torch.FloatTensor(self.batch_size, nc, imageSize, imageSize).cuda()
        dis_c = torch.FloatTensor(self.batch_size, nClass).cuda()
        label = torch.LongTensor(self.batch_size).cuda()
        noise = torch.FloatTensor(self.batch_size, nNoise).cuda()
        empty_image = torch.FloatTensor(self.batch_size, nc, imageSize, imageSize).cuda()

        real_x = Variable(real_x)
        dis_c = Variable(dis_c)
        label = Variable(label, requires_grad=False)
        noise = Variable(noise)
        empty_image = Variable(empty_image)

        empty_image.data.fill_(-1.0)

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

        #calculate weights of each class
        if dataset_type == 'lsun':
            #double check if the classes number is set correctly
            classes = dataloader.dataset.classes
            if nClass != len(classes):
                print('The user-defined number of classes is different from the number read from dataset! %d : %d' % (nClass, len(classes)))
                exit(-1)
        class_weights = np.ones(nClass + 1)
        if dataset_type == 'lsun':
            #assign weights to different classes based on their data size
            indices = dataloader.dataset.indices
            n_images = sum(indices)
            for i in range(nClass):
                class_weights[i] = n_images / indices[i] # number of images/number in this class; assign 1 to fake class

        criterionD = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights)).cuda()
        criterion_L1 = nn.L1Loss().cuda()
        # criterionD = nn.CrossEntropyLoss().cuda()
        # criterionD = nn.BCELoss().cuda()

        optimD = optim.Adam([{'params': self.FE.parameters()}, {'params': self.D.parameters()}], lr=lr_D,
                            betas=(0.5, 0.999))
        optimG = optim.Adam([{'params': self.G.parameters()}], lr=lr_G,
                            betas=(0.5, 0.999))

        # fixed random variables
        idx = np.arange(nClass).repeat(100 / nClass)
        one_hot = np.zeros((100, nClass))
        one_hot[range(100), idx] = 1
        fix_noise = torch.Tensor(100, nNoise).uniform_(-1, 1)

        D_loss_overall = np.zeros(nEpoch)
        G_loss_overall = np.zeros(nEpoch)
        L1_loss_overall = np.zeros(nEpoch)
        for epoch in range(nEpoch):
            start_time = time.time()
            D_loss_epoch = np.zeros(len(dataloader))
            G_loss_epoch = np.zeros(len(dataloader))
            L1_loss_epoch = np.zeros(len(dataloader))
            for num_iters, batch_data in enumerate(dataloader, 0):
                # D loss
                # real part
                optimD.zero_grad()
                x, batch_labels = batch_data
                # x, _ = batch_data
                bs = x.size(0)
                real_x.data.resize_(x.size())
                label.data.resize_(bs)
                dis_c.data.resize_(bs, nClass)
                noise.data.resize_(bs, nNoise)

                real_x.data.copy_(x)
                fe_out1 = self.FE(real_x)
                probs_real = self.D(fe_out1)
                # label.data.fill_(1)
                label.data.copy_(batch_labels)
                # target = Variable(torch.LongTensor(batch_labels).cuda())
                loss_real = criterionD(probs_real, label)
                loss_real.backward()

                # fake part
                z, idx = self._noise_sample(dis_c, noise, bs)
                fake_x = self.G(z)
                fe_out2 = self.FE(fake_x.detach())
                probs_fake = self.D(fe_out2)
                label.data.fill_(nClass)    #the generated images are all fake to the D. So they should be classified to the "fake" class
                loss_fake = criterionD(probs_fake, label)
                loss_fake.backward()

                D_loss = loss_real + loss_fake

                optimD.step()

                # G lossimage
                optimG.zero_grad()

                fe_out = self.FE(fake_x)
                probs_fake = self.D(fe_out)
                label.data.copy_(torch.LongTensor(idx))
                L1_loss = criterion_L1(fake_x, empty_image) * 5.0
                L0_loss = self.L0_loss(fake_x) * 5.0
                G_loss = criterionD(probs_fake, label)  + L1_loss    #G tries to make D "believe" that the generated image should be classified to a real class
                G_loss.backward()
                optimG.step()

                D_loss_epoch[num_iters] = D_loss.data[0]
                G_loss_epoch[num_iters] = G_loss.data[0]
                L1_loss_epoch[num_iters] = L1_loss.data[0]
                if num_iters % 10 == 0:
                    print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f Loss_L1: %.4f'
                          % (epoch, nEpoch, num_iters, len(dataloader),
                             D_loss.data[0], G_loss.data[0], L1_loss.data[0]))

                    # save the randomly generated images
                    save_image(fake_x.data,
                               '%s/random_fake_samples_latest.png' % outf,
                               normalize=normalizeImage, nrow=10)
                    save_image(fake_x.data,
                               '%s/random_fake_samples_epoch_%d.png' % (outf, epoch),
                               normalize=normalizeImage, nrow=10)

                    #reproduce images using saved noise and class settings
                    noise.data.copy_(fix_noise)
                    dis_c.data.copy_(torch.Tensor(one_hot))

                    z = torch.cat([noise, dis_c], 1).view(-1, nClass + nNoise, 1, 1)
                    x_save = self.G(z)
                    save_image(x_save.data,
                               '%s/fixed_fake_samples_epoch_%d.png' % (outf, epoch),
                               normalize=normalizeImage, nrow=nCPerRow)
                    save_image(x_save.data,
                               '%s/fixed_fake_latest.png' % (outf),
                               normalize=normalizeImage, nrow=nCPerRow)

                    self.draw('Loss of epoch %d' % epoch,
                              '%s/loss_epoch_%d.pdf' % (outf, epoch),
                              D_loss_epoch, G_loss_epoch, L1_loss_epoch)

            D_loss_overall[epoch] = D_loss_epoch[-1]
            G_loss_overall[epoch] = G_loss_epoch[-1]
            L1_loss_overall[epoch] = L1_loss_overall[-1]
            self.draw('Overall Loss',
                      '%s/loss_overall.pdf' % (outf),
                      D_loss_overall, G_loss_overall, L1_loss_overall)
            torch.save(self.G.state_dict(), '%s/netG_epoch_%d.pth' % (outf, epoch))
            torch.save(self.D.state_dict(), '%s/netD_epoch_%d.pth' % (outf, epoch))
            elapsed_time = time.time() - start_time
            print("Elapsed Time for epoch %d is:%4f" % (epoch, elapsed_time))
