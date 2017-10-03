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
import pytorch_fft.fft.autograd as fft_autograd
import torch.nn.functional as F

import numpy as np
# for loss plot
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# set the dataroot, which is a folder
# dataset_type = 'lsun'
# dataroot = '/home/zeyuan/lsun/lmdb'
lsun_classes=['conference_room_train', 'bridge_train']

dataset_type = 'folder'
dataroot = '/home/zeyuan/dataset/places'

# dataset_type = 'mnist'
# dataroot = '/home/zeyuan/dataset/mnist'

# outf = './checkpoints/mnist/D_Mag_takeTurn_0.2'

outf = './checkpoints/places/D_Mag_size20_w0.2_lrMag_0.00001_lrD_0.0002_skycraper_bridge_fc_v1'

nClass = 2

workers = 2
imageSize = 64
magnitudeSize = 32
nNoise = 100
nc = 3
if dataset_type == 'mnist':
    nc = 1
nEpoch = 1000
nCPerRow = 10
lr_D = 0.0002
lr_G = 0.0002
lr_mag = 0.00001
weigth_D_mag = 0.2
normalizeImage = True

load_z_fixed = True
use_fc = True


try:
    os.makedirs(outf)
except OSError:
    pass


class log_gaussian:
    def __call__(self, x, mu, var):
        logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - \
                (x - mu).pow(2).div(var.mul(2.0) + 1e-6)

        return logli.sum(1).mean().mul(-1)

class myresize(torch.autograd.Function):
    def forward(self, input):
        output = input.clone()

        return output.resize_(input.size(0), 1, input.size(1), input.size(2))
    def backward(self, grad_outputs):
        grad_inputs = grad_outputs.clone()

        return grad_inputs.resize_(grad_outputs.size(0), grad_outputs.size(2), grad_outputs.size(3))

class mylowpass(torch.autograd.Function):
    def forward(self, input):
        bs,nc,rows, cols = input.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        frequency_limit = int(magnitudeSize/2)
        # output = torch.zeros(bs, nc, magnitudeSize, magnitudeSize).cuda()
        output = input[:, :, crow - frequency_limit:crow + frequency_limit, ccol - frequency_limit:ccol + frequency_limit]
        return output

    def backward(self, grad_outputs):
        bs, nc, rows, cols = grad_outputs.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        frequency_limit = int(magnitudeSize / 2)
        grad_inputs = torch.zeros(bs, nc, imageSize, imageSize).cuda()
        grad_inputs[:, :, crow - frequency_limit:crow + frequency_limit, ccol - frequency_limit:ccol + frequency_limit] = grad_outputs
        # grad_inputs[:, :, 0 : crow - 16, :] = 0
        # grad_inputs[:, :, crow+16: rows, :] = 0
        # grad_inputs[:, :, :, 0 : ccol - 16] = 0
        # grad_inputs[:, :, :, ccol+16: cols] = 0

        return grad_inputs


class myfftshift(torch.autograd.Function):
    def forward(self, input):
        #switch the first quadrant of input with the third, and the second quadrant with the fourth
        _,_, rows, cols = input.shape
        half_rows = int(rows/2)
        half_cols = int(cols/2)
        output = input.clone()
        output[:, :, 0: half_rows, 0: half_cols] = input[:, :, half_rows: rows, half_cols : cols]
        output[:, :, half_rows : rows, half_cols : cols] = input[:, :, 0: half_rows, 0: half_cols]
        output[:, :, 0: half_rows, half_cols: cols] = input[:, :, half_rows: rows, 0: half_cols]
        output[:, :, half_rows: rows, 0: half_cols] = input[:, :, 0: half_rows, half_cols : cols]

        return output

    def backward(self, grad_outputs):
        _, _, rows, cols = grad_outputs.shape
        half_rows = int(rows / 2)
        half_cols = int(cols / 2)
        grad_inputs = grad_outputs.clone()
        grad_inputs[:, :, 0: half_rows, 0: half_cols] = grad_outputs[:, :, half_rows : rows, half_cols : cols]
        grad_inputs[:, :, half_rows: rows, half_cols: cols] = grad_outputs[:, :, 0: half_rows, 0: half_cols]
        grad_inputs[:, :, 0: half_rows, half_cols: cols] = grad_outputs[:, :, half_rows: rows, 0: half_cols]
        grad_inputs[:, :, half_rows: rows, 0: half_cols] = grad_outputs[:, :, 0: half_rows, half_cols: cols]
        return grad_inputs



class Trainer:
    def __init__(self, G, FE, D, D_Mag, D_Mag_FC):

        self.G = G
        self.FE = FE
        self.D = D
        if use_fc:
            self.D_Mag = D_Mag_FC
        else:
            self.D_Mag = D_Mag

        self.batch_size = 100

    def _noise_sample(self, batch_labels, dis_c, noise, bs):

        idx = batch_labels.int().cpu().numpy().astype(int)
        # idx = np.random.randint(nClass, size=bs)
        c = np.zeros((bs, nClass))
        c[range(bs), idx] = 1.0

        dis_c.data.copy_(torch.Tensor(c))
        noise.data.uniform_(-1.0, 1.0)
        z = torch.cat([noise, dis_c], 1).view(-1, nClass + nNoise, 1, 1)

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

    def saveImgMagnitude(self, real, real_mag, fake, fake_mag):
        plt.figure()
        plt.subplot(221), plt.imshow(real.data[0][0].cpu().numpy(), cmap='gray')
        plt.title('Real Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(222), plt.imshow(real_mag.data[0][0].cpu().numpy(), cmap='gray')
        plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
        plt.subplot(223), plt.imshow(fake.data[0][0].cpu().numpy(), cmap='gray')
        plt.title('Fake Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(224), plt.imshow(fake_mag.data[0][0].cpu().numpy(), cmap='gray')
        plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
        plt.savefig('%s/magnitude_spectrum_random_real.pdf' % outf)
        plt.close()

    def image2magnitude(self, img):
        # convert RGB to grayscale and then to magnitude
        # img_gray = np.zeros((img.data.size(0), img.data.size(2), img.data.size(3)))
        # img_gray = img.clone()
        if img.data.size(1) == 3:
            img_gray = img[:, 0, :, :].mul(0.3).add(img[:, 1, :, :].mul(0.59)).add(img[:, 2, :, :].mul(0.11))
            img_gray = myresize()(img_gray)
        else:
            img_gray = img

        re, im = fft_autograd.Fft2d()(img_gray, Variable(torch.zeros(img_gray.size()).cuda()))
        spectrum = (re.pow(2)+ im.pow(2))
        spectrum_shift = myfftshift()(spectrum)
        spectrum_shift_low = mylowpass()(spectrum_shift)

        return spectrum_shift_low


    def train(self):
        real_x = torch.FloatTensor(self.batch_size, nc, imageSize, imageSize).cuda()
        dis_c = torch.FloatTensor(self.batch_size, nClass).cuda()
        label = torch.LongTensor(self.batch_size).cuda()
        noise = torch.FloatTensor(self.batch_size, nNoise).cuda()
        magnitude_real = torch.FloatTensor(self.batch_size, 1, magnitudeSize, magnitudeSize).cuda()
        magnitude_fake = torch.FloatTensor(self.batch_size, 1, magnitudeSize, magnitudeSize).cuda()



        real_x = Variable(real_x)
        magnitude_real = Variable(magnitude_real, requires_grad=False)
        magnitude_fake = Variable(magnitude_fake)
        dis_c = Variable(dis_c)
        label = Variable(label, requires_grad=False)
        noise = Variable(noise)

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
            class_weights /= max(class_weights)

        criterionD = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights)).cuda()
        criterion_L1 = nn.L1Loss().cuda()
        criterion_MSE = nn.MSELoss().cuda()
        criterionD_Mag = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights)).cuda()

        optimD = optim.Adam([{'params': self.FE.parameters()}, {'params': self.D.parameters()}], lr=lr_D,
                            betas=(0.5, 0.999))
        optimG = optim.Adam([{'params': self.G.parameters()}], lr=lr_G,
                            betas=(0.5, 0.999))
        optimG_mag = optim.Adam([{'params': self.G.parameters()}], lr=lr_mag,
                                betas=(0.5, 0.999))
        optimD_Mag = optim.Adam([{'params': self.D_Mag.parameters()}], lr=lr_mag,
                                betas=(0.5, 0.999))

        # fixed random variables
        idx = np.arange(nClass).repeat(100 / nClass)
        one_hot = np.zeros((100, nClass))
        one_hot[range(100), idx] = 1
        fix_noise = torch.Tensor(100, nNoise).uniform_(-1, 1)

        noise.data.copy_(fix_noise)
        dis_c.data.copy_(torch.Tensor(one_hot))

        if not load_z_fixed:
            z_fixed = torch.cat([noise, dis_c], 1).view(-1, nClass + nNoise, 1, 1)
            torch.save(z_fixed, '/home/zeyuan/infoGAN-pytorch/checkpoints/z_fixed.pt')
        else:
            z_fixed = torch.load('/home/zeyuan/infoGAN-pytorch/checkpoints/z_fixed.pt')
        D_loss_overall = np.zeros(nEpoch)
        G_loss_overall = np.zeros(nEpoch)

        for epoch in range(nEpoch):
            start_time = time.time()
            D_loss_epoch = np.zeros(len(dataloader))
            G_loss_epoch = np.zeros(len(dataloader))
            for num_iters, batch_data in enumerate(dataloader, 0):
                x, batch_labels = batch_data
                bs = x.size(0)

                real_x.data.resize_(x.size())
                label.data.resize_(bs)
                dis_c.data.resize_(bs, nClass)
                noise.data.resize_(bs, nNoise)
                magnitude_real.data.resize_(bs, 1, magnitudeSize, magnitudeSize) #the size of the last batch might be smaller than the default batch size
                magnitude_fake.data.resize_(bs, 1, magnitudeSize, magnitudeSize)
                z_current_fixed= z_fixed[ :bs, : ,: ,:]

                # D loss
                # real part
                optimD.zero_grad()
                real_x.data.copy_(x)
                fe_out1 = self.FE(real_x)
                probs_real = self.D(fe_out1)
                label.data.copy_(batch_labels)
                loss_real = criterionD(probs_real, label)
                loss_real.backward()

                # fake part
                z, idx = self._noise_sample(batch_labels, dis_c, noise, bs)
                fake_x = self.G(z)
                fe_out2 = self.FE(fake_x.detach())
                probs_fake = self.D(fe_out2)
                label.data.fill_(nClass)    #the generated images are all fake to the D. So they should be classified to the "fake" class
                loss_fake = criterionD(probs_fake, label)
                loss_fake.backward()

                D_loss = loss_real + loss_fake
                optimD.step()


                #D_Mag loss
                optimD_Mag.zero_grad()
                magnitude_real.data.copy_(self.image2magnitude(real_x).data)
                magnitude_fake= self.image2magnitude(fake_x.detach())
                #real part
                probs_mag_real = self.D_Mag(magnitude_real)
                label.data.copy_(batch_labels)
                loss_mag_real = criterionD_Mag(probs_mag_real, label) * weigth_D_mag
                loss_mag_real.backward()

                #fake part
                probs_mag_fake = self.D_Mag(magnitude_fake.detach())
                label.data.fill_(nClass)
                loss_mag_fake = criterionD_Mag(probs_mag_fake,label) * weigth_D_mag
                loss_mag_fake.backward()

                D_mag_loss = loss_mag_real + loss_mag_fake
                optimD_Mag.step()

                # G loss
                optimG_mag.zero_grad()
                #mag part
                magnitude_fake = self.image2magnitude(fake_x)
                probs_mag_fake = self.D_Mag(magnitude_fake)
                label.data.copy_(torch.LongTensor(idx))
                # L1_loss = criterion_L1(fake_x, real_x)
                G_loss_1 = criterionD_Mag(probs_mag_fake, label)
                G_loss_1.backward()
                optimG_mag.step()

                #origin part
                optimG.zero_grad()
                fake_x_2 = self.G(z) # regenerate the fake image with the updated G
                fe_out = self.FE(fake_x_2)
                probs_fake = self.D(fe_out)
                G_loss_2 =  criterionD(probs_fake, label)
                G_loss_2.backward()
                optimG.step()
                G_loss = G_loss_1 + G_loss_2

                D_loss_epoch[num_iters] = D_loss.data[0]
                G_loss_epoch[num_iters] = G_loss.data[0]
                if num_iters % 10 == 0:
                    print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G_2: %.4f, Loss_D_Mag: %.4f, LossG_1: %.4f'
                          % (epoch, nEpoch, num_iters, len(dataloader),
                             D_loss.data[0], G_loss_2.data[0], D_mag_loss.data[0], G_loss_1.data[0]))

                if num_iters % 100 == 0:
                    # save the randomly generated images
                    save_image(fake_x.data,
                               '%s/random_fake_samples_latest.png' % outf,
                               normalize=normalizeImage, nrow=10)
                    save_image(fake_x.data,
                               '%s/random_fake_samples_epoch_%d.png' % (outf, epoch),
                               normalize=normalizeImage, nrow=10)

                    #reproduce images using saved noise and class settings

                    x_save = self.G(z_current_fixed)
                    save_image(x_save.data,
                               '%s/fixed_fake_samples_epoch_%d.png' % (outf, epoch),
                               normalize=normalizeImage, nrow=nCPerRow)
                    save_image(x_save.data,
                               '%s/fixed_fake_latest.png' % (outf),
                               normalize=normalizeImage, nrow=nCPerRow)

                    self.draw('Loss of epoch %d' % epoch,
                              '%s/loss_epoch_%d.pdf' % (outf, epoch),
                              D_loss_epoch, G_loss_epoch)
                    self.saveImgMagnitude(real_x, magnitude_real, fake_x, magnitude_fake)

            D_loss_overall[epoch] = D_loss_epoch[-1]
            G_loss_overall[epoch] = G_loss_epoch[-1]
            self.draw('Overall Loss',
                      '%s/loss_overall.pdf' % (outf),
                      D_loss_overall, G_loss_overall)
            torch.save(self.G.state_dict(), '%s/netG_epoch_%d.pth' % (outf, epoch))
            torch.save(self.D.state_dict(), '%s/netD_epoch_%d.pth' % (outf, epoch))
            elapsed_time = time.time() - start_time
            print("Elapsed Time for epoch %d is:%4f" % (epoch, elapsed_time))
