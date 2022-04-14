import numpy as np
import random
import matplotlib.pyplot as plt
import os
import glob
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

# linear 全连接网络
class Generator(nn.Module):
    # input (batch size, in_dim)
    # output (batch size, 3, 64, 64)

    def __init__(self, in_dim, dim=64):
      super(Generator, self).__init__()

      self.model = nn.Sequential(
          nn.Linear(in_dim, 128),
          nn.LeakyReLU(0.2, inplace=True),

          nn.Linear(128, 256),
          nn.BatchNorm1d(256, 0.8),
          nn.LeakyReLU(0.2, inplace=True),

          nn.Linear(256, 512),
          nn.BatchNorm1d(512, 0.8),
          nn.LeakyReLU(0.2, inplace=True),

          nn.Linear(512, 1024),
          nn.BatchNorm1d(1024, 0.8),
          nn.LeakyReLU(0.2, inplace=True),

          nn.Linear(1024, 3*dim*dim),
          nn.Tanh()
      )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], 3, 64, 64)
        return img

class Discriminator(nn.Module):
    # input (batch size, 3, 64, 64)
    # output (batch size, )

    def __init__(self, dim=64):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(3*dim*dim, 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_flat = x.view(x.shape[0], -1)
        y = self.model(x_flat).squeeze()
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
    # gan_g.param
    save_name  = sys.argv[2]
    # gan.jpg

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
