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


"""# 1、Data
预处理：
变为RGB图片，resize到64*64，从[0, 1] 映射到 [-1, 1]
"""

class FaceDataset(Dataset):
    def __init__(self, fnames, transform):
        self.transform = transform
        self.fnames = fnames
        self.num_samples = len(self.fnames)
    def __getitem__(self,idx):
        fname = self.fnames[idx]
        img = cv2.imread(fname)
        # 将cv2的BGR图片转化为torchvision的RGB图片
        img = self.BGR2RGB(img)
        img = self.transform(img)
        return img

    def __len__(self):
        return self.num_samples


    def BGR2RGB(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

def get_dataset(root):
    # 读取root文件下的所有data
    fnames = glob.glob(os.path.join(root, '*'))
    transform = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.Resize((64, 64)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3) ] )
    dataset = FaceDataset(fnames, transform)
    return dataset

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

"""# 2、train方法
WGAN方法
卷积网络结构
"""
# 初始化权重
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train_wgan(generator, discriminator, optimizer_G, optimizer_D, clip_value, n_critic, save_dir, dataloader, latent_dim, n_epoch):
    # 用于evaluate的sample，随机生成100个input vector
    z_sample = Variable(torch.randn(100, latent_dim)).cuda()

    Tensor = torch.cuda.FloatTensor

    # train
    generator.train()
    discriminator.train()

    for epoch in range(n_epoch):
        for i, data in enumerate(dataloader):
            imgs = data
            imgs = imgs.cuda()
            # batch size
            bs = imgs.size(0)

            # ---------------------
            #  Train Discriminator
            # ---------------------
    
            optimizer_D.zero_grad()
    
            # 随机生成batch size个input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))))
    
            # generator生成的图片
            fake_imgs = generator(z).detach()
            # 真实图片
            real_imgs = Variable(imgs).cuda()

            # 计算loss
            loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))

            # 更新模型
            loss_D.backward()
            optimizer_D.step()
    
            # Discriminator需要Clip weights
            for p in discriminator.parameters():
                p.data.clamp_(-clip_value, clip_value)

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # 生成图片
            gen_imgs = generator(z)
            # 计算loss
            loss_G = -torch.mean(discriminator(gen_imgs))

            # 更新模型
            loss_G.backward()
            optimizer_G.step()

            print(f'\rEpoch [{epoch+1}/{n_epoch}] {i+1}/{len(dataloader)} Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}', end='')
    
        # 每10个epoch计算、存一下生成的照片
        if (epoch+1) % 10 == 0:
            # generator eval模式
            generator.eval()
            #  从[-1, 1]重新映射到[0, 1]
            f_imgs_sample = (generator(z_sample).data + 1) / 2.0
    
            # filename = os.path.join(save_dir, f'Epoch_{epoch+1:03d}.jpg')
            # torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
            # print(f' | Save some samples to {filename}.')
    
            # 绘制生成图片
            grid_img = torchvision.utils.make_grid(f_imgs_sample.cpu(), nrow=10)
            plt.figure(figsize=(10,10))
            plt.imshow(grid_img.permute(1, 2, 0))
            plt.show()

            # generator train模式
            generator.train()

    # 存储参数
    torch.save(generator.state_dict(), os.path.join(save_dir, f'wgan_g.param'))
    torch.save(discriminator.state_dict(), os.path.join(save_dir, f'wgan_d.param'))

# 卷积神经网络 + WGAN
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

            nn.Conv2d(dim * 8, 1, 4))
        
        self.apply(weights_init)        

    def forward(self, x):
        y = self.ls(x)
        y = y.view(-1)
        return y


if __name__ == '__main__':
    data_dir = sys.argv[1]
    # faces文件夹地址
    save_dir =  sys.argv[2]
    # g.param、d.param要保存在save_dir文件夹下，例如save_dir = path/log,则参数保存为 path/log/wagan_g.param, path/log/wgan_d.param
    n_epoch =  sys.argv[3]

    latent_dim = 100

    lr=5e-5
    clip_value=0.01

    # load data
    same_seeds(0)
    batch_size = 64
    dataset = get_dataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 初始化model
    generator = Generator(latent_dim).cuda()
    discriminator = Discriminator(3).cuda()

    # optimizer
    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=lr)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=lr)

    train_wgan(generator, discriminator, optimizer_G, optimizer_D,  clip_value, n_critic, save_dir, dataloader, latent_dim, n_epoch)