

# README
实验采用了GAN、WGAN两种模型进行二次元小姐姐头像生成。其中GAN又分别采用了全连接网络、卷积神经网络（DCGAN）两种方式；WGAN在卷积神经网络（DCGAN）基础上进行修改得到。

## train.sh
依次训练GAN、DCGAN、WGAN
$1: faces data文件夹的path,例如path/faces
$2: generator、discriminator的参数保存的文件夹path，例如$2 = path/log,则训练结束后wgan的参数自动保存到 path/log/wgan_g.param、path/log/wgan_d.param
$3: 训练的epoch数量

需要说明的用到的包：
- glob、cv2：用于读入图片


## test.sh 
$1 = 需要读入的gan generator的参数文件，例如path/log/gan_g.param
$2 = gan test生成图片的保存名，例如path/result/gan.jpg
$3 = 需要读入的dcgan generator的参数文件，例如path/log/dcgan_g.param
$4 = dcgan test生成图片的保存名，例如path/result/dcgan.jpg
$5 = 需要读入的wgan generator的参数文件，例如path/log/wgan_g.param
$6 = wgan test生成图片的保存名，例如path/result/wgan.jpg