#!/usr/bin/env bash

python3 GAN_test.py $1 $2 & DCGAN_test.py $3 $4 & WGAN_test.py $5 $6

# $1 = gan generator的参数文件，例如path/log/gan_g.param
# $2 = gan test生成图片的保存名，例如path/result/gan.jpg
# $3 = dcgan generator的参数文件，例如path/log/dcgan_g.param
# $4 = dcgan test生成图片的保存名，例如path/result/dcgan.jpg
# $5 = wgan generator的参数文件，例如path/log/wgan_g.param
# $6 = wgan test生成图片的保存名，例如path/result/wgan.jpg