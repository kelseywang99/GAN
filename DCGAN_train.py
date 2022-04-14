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
经典GAN方法
Linear网络结构
"""

# 初始化权重
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# 经典gan算法
def train_gan(G, D, opt_D, opt_G, criterion, save_dir, dataloader, latent_dim, n_epoch):
    # 用于evaluate的sample，随机生成100个input vector
    z_sample = Variable(torch.randn(100, latent_dim)).cuda()
    # train
    G.train()
    D.train()
    for e, epoch in enumerate(range(n_epoch)):
        for i, data in enumerate(dataloader):
            imgs = data
            imgs = imgs.cuda()
            # batch size
            bs = imgs.size(0)
    
            # ---------------------
            #  Train Discriminator
            # ---------------------
            
            # 随机生成bs个input
            z = Variable(torch.randn(bs, latent_dim)).cuda()
            # 真实图片
            r_imgs = Variable(imgs).cuda()
            # generator生成的图片
            f_imgs = G(z)
    
            # 真实图片label=1，生成图片label=0
            r_label = torch.ones((bs)).cuda()
            f_label = torch.zeros((bs)).cuda()
    
            # 图片经过discriminator，并将generator的部分detach
            r_logit = D(r_imgs.detach())
            f_logit = D(f_imgs.detach())
            
            # 计算loss
            r_loss = criterion(r_logit, r_label)
            f_loss = criterion(f_logit, f_label)
            loss_D = (r_loss + f_loss) / 2
    
            # 更新模型
            D.zero_grad()
            loss_D.backward()
            opt_D.step()
    
            # ---------------------
            #  Train Generator
            # ---------------------

            # bs个input，generator生成图片
            z = Variable(torch.randn(bs, latent_dim)).cuda()
            f_imgs = G(z)
    
            # 经过discriminator
            f_logit = D(f_imgs)
            
            # 计算loss
            loss_G = criterion(f_logit, r_label)
    
            # 更新模型
            G.zero_grad()
            loss_G.backward()
            opt_G.step()
    
        print(f'\rEpoch [{epoch+1}/{n_epoch}]  Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}', end='')
        

        # 每10个epoch计算、存一下生成的照片
        if (e+1) % 10 == 0:
            # generator evaluate模式
            G.eval()
            # 从[-1, 1]重新映射到[0, 1]
            f_imgs_sample = (G(z_sample).data + 1) / 2.0
            # filename = os.path.join(save_dir, f'Epoch_{epoch+1:03d}.jpg')
            # torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
            # print(f' | Save some samples to {filename}.')
    
            # 绘制生成图片
            grid_img = torchvision.utils.make_grid(f_imgs_sample.cpu(), nrow=10)
            plt.figure(figsize=(10,10))
            plt.imshow(grid_img.permute(1, 2, 0))
            plt.show()

            # generator训练模式
            G.train()

    # 存储参数
    torch.save(G.state_dict(), os.path.join(save_dir, f'dcgan_g.param'))
    torch.save(D.state_dict(), os.path.join(save_dir, f'dcgan_d.param'))


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


if __name__ == '__main__':
    data_dir = sys.argv[1]
    # faces文件夹地址
    save_dir =  sys.argv[2]
    # g.param、d.param要保存在save_dir文件夹下，例如save_dir = path/log,则参数保存为 path/log/dcgan_g.param, path/log/dcgan_d.param
    n_epoch =  sys.argv[3]

    latent_dim = 100
    lr = 1e-4

    # load data
    same_seeds(0)
    batch_size = 64
    dataset = get_dataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 初始化model
    G = Generator(in_dim=latent_dim).cuda()
    D = Discriminator(3).cuda()

    # loss
    criterion = nn.BCELoss()

    # optimizer
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

    train_gan(G, D, opt_D, opt_G, criterion, save_dir, dataloader, latent_dim, n_epoch)