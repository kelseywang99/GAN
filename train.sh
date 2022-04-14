#!/usr/bin/env bash

python3 GAN_train.py $1 $2 $3 & python3 DCGAN_train.py $1 $2 $3 & python3 WGAN_train.py $1 $2 $3

# $1: faces data 文件夹的path
# $2: generator、discriminator的参数保存的文件夹path，例如$2 = path/log,则wgan的参数保存为 path/log/wgan_g.param, path/log/wgan_d.param
# $3: 训练的epoch数量