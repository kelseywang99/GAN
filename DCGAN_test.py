import numpy as np
import random
import matplotlib.pyplot as plt
import os
import cv2
import sys

import torch
from torch import optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

# 初始化权重
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# DCGAN使用卷积神经网络
class Generator(nn.Module):
    # input (batch size, in_dim)
    # output (batch size, 3, 64, 64)

    def __init__(self, in_dim, dim=64):
        super(Generator, self).__init__()

        def dconv_bn_relu(in_dim, out_dim):
            block =  nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 5, 2,
                                   padding=2, output_padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU())
            return block
          
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.ReLU())
        
        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim * 8, dim * 4),
            dconv_bn_relu(dim * 4, dim * 2),
            dconv_bn_relu(dim * 2, dim),
            nn.ConvTranspose2d(dim, 3, 5, 2, padding=2, output_padding=1),
            nn.Tanh())
        
        self.apply(weights_init)

    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2_5(y)
        return y

class Discriminator(nn.Module):
    # input (batch size, 3, 64, 64)
    # output (batch size, )

    def __init__(self, in_dim, dim=64):
        super(Discriminator, self).__init__()

        def conv_bn_lrelu(in_dim, out_dim):
            block = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2))
            return block

        self.ls = nn.Sequential(
            nn.Conv2d(in_dim, dim, 5, 2, 2), 
            nn.LeakyReLU(0.2),

            conv_bn_lrelu(dim, dim * 2),
            conv_bn_lrelu(dim * 2, dim * 4),
            conv_bn_lrelu(dim * 4, dim * 8),

            nn.Conv2d(dim * 8, 1, 4),
            nn.Sigmoid())
        
        self.apply(weights_init)        

   
    def forward(self, x):
        y = self.ls(x)
        y = y.view(-1)
        return y

# 固定 random seed
def same_seeds(seed):
    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    # Numpy
    np.random.seed(seed)  
    # Python
    random.seed(seed)
    
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':    
    param_dir = sys.argv[1]
    # dcgan_g.param
    save_name  = sys.argv[2]
    # dcgan.jpg

    latent_dim = 100

    G = Generator(latent_dim)
    G.load_state_dict(torch.load(param_dir))
    G.eval()
    G.cuda()

    # generate images and save the result
    n_output = 20
    z_sample = Variable(torch.randn(n_output, latent_dim)).cuda()
    imgs_sample = (G(z_sample).data + 1) / 2.0
    torchvision.utils.save_image(imgs_sample, save_name, nrow=10)
    # show image
    grid_img = torchvision.utils.make_grid(imgs_sample.cpu(), nrow=10)
    plt.figure(figsize=(10,10))
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()
