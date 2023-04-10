import argparse
# argparse 模块可以让人轻松编写用户友好的命令行接口。程序定义它需要的参数，然后 argparse 将弄清如何从 sys.argv 解析出那些参数。
import os
rank = int(os.environ["rank"])
# 在python环境下对文件，文件夹执行操作的一个模块
import numpy as np
import math

from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
# pytorch中的图像预处理包，包含了很多种对图像数据进行变换的函数
from torchvision.utils import save_image

from torch.utils.data import DataLoader, Dataset

from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from torch.utils.data.distributed import DistributedSampler  # 分布式训练多GPU


"""多GPU训练配置"""
# 1) 初始化
torch.distributed.init_process_group(backend="nccl")

# 2） 配置每个进程的gpu
rank = torch.distributed.get_rank()
print('rank', rank)
torch.cuda.set_device(rank)
device = torch.device("cuda", rank)


"""参数初始化"""
os.makedirs("images_test", exist_ok=True)  # 创建images文件夹用来存放生成结果


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")  # 训练epoch次数
parser.add_argument("--batch_size", type=int, default=125, help="number of batch_size")  # 设置batch_size
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")  # 学习率
parser.add_argument("--b1", type=float, default=0.5,
                    help="adam: decay of first order momentum of gradient")  # betas: 用于计算梯度以及梯度平方的运行平均值的系数
parser.add_argument("--b2", type=float, default=0.999,
                    help="adam: decay of first order momentum of gradient")  # betas: 用于计算梯度以及梯度平方的运行平均值的系数
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")  # 隐空间的维度  噪声向量大小
parser.add_argument("--img_size", type=int, default=96, help="size of each image dimension")  # 训练图片大小 96*96
parser.add_argument("--channels", type=int, default=3, help="number of image channels")  # 输入通道 RGB三通道
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
#parser.add_argument("--rank", default=-1, type=int, help="node rank for distributed training")  # 分布式训练GPU，不加会报错
opt = parser.parse_args()

img_shape = (opt.channels, opt.img_size, opt.img_size)  # 输入图片数据shape=（3，96，96）




"""读取数据集"""
train_augs = transforms.Compose([
    transforms.Resize(opt.img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])  # 对输入图片从PIL转成numpy 并resize

data_images = ImageFolder(root='./facesdata', transform=train_augs)


# ImageFolder读取上file 这里 file/label(用label作为文件名)/image.jpg
# data_images[0]为图片数据  data_images[1]为根据图片所在文件夹名称设置的label 因为本次数据label在csv文件中所以这个label都为上一层文件夹名称


# 创建数据集
class faces_dataset(Dataset):  # 继承自torch.utils.data.Dataset
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        label = self.labels[item][1]
        data = self.images[item][0]  # 届时传入一个ImageFolder对象，需要取[0]获取数据，不要标签
        return data, label


faces_dataset = faces_dataset(images=data_images, labels=data_images)  # 实例化faces_dataset
# train_iter = DataLoader(dataset=faces_dataset, batch_size=opt.batch_size,
#                         drop_last=True,
#                         sampler=DistributedSampler(faces_dataset))  # drop_last=True舍弃掉最后无法凑齐batch_size的数据

train_iter = DataLoader(dataset=faces_dataset, batch_size=opt.batch_size,
                        drop_last=True
                        ) 

# X, y = next(iter(train_iter))  # next(iter())迭代器 取出一条batch
# print(X.shape, y[0])

"""网络结构"""


class Generator(nn.Module):  # 生成网络
    def __init__(self):
        super(Generator, self).__init__()  # 超类继承
        self.model = nn.Sequential(
            # input is Z, going into a convolution  输入(batch_size=125,channel=100,1,1)  输入100个1x1的随机矩阵
            nn.ConvTranspose2d(opt.latent_dim, opt.img_size * 8, 4, 1, 0, bias=False),
            # ConvTranspose2d(input_chanel,output_chanel,kernel_size=4 ,stride=1, padding=0, output_padding=0)
            # output_size=(input_size-1) * stride - 2padding + kernel_size
            # 输入通道为100，输出通道96*8,核函数是4*4

            nn.BatchNorm2d(opt.img_size * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (1-1)*1-2*0+4=4   输入(125, opt.img_size * 8, 4, 4)
            nn.ConvTranspose2d(opt.img_size * 8, opt.img_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.img_size * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # state size.   (4-1)*2-2*1+4=8  输入(125, opt.img_size * 4, 8, 8)
            nn.ConvTranspose2d(opt.img_size * 4, opt.img_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.img_size * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (8-1)*2-2*1+4=16  输入(125, opt.img_size * 2, 16, 16)
            nn.ConvTranspose2d(opt.img_size * 2, opt.img_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.img_size),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (16-1)*2-2*1+4=132  输入(125, opt.img_size , 32, 32)
            nn.ConvTranspose2d(opt.img_size, 3, 3, 3, 0, bias=False),
            # (32 - 1)*3-2*0+3=96 输出(125, 3, 96, 96)
            nn.Tanh()
                    )  # 快速搭建网络， np.prod 用来计算所有元素的乘积

    def forward(self, z):  # 前向传播 z代表输入 z.size=(batch_size=125,channel=100, 1 , 1 )
        img = self.model(z)  # img.shape=(125, 3, 96, 96)
        return img


'''
这段定义了一个判别网络，也可以是先拿一个套路来看
'''


def vgg_block(conv_number, in_channels, out_channels):
    """
    构造一个vgg块
    :param conv_number: 卷积层数
    :param in_channels: 输入通道
    :param out_channels: 输出通道
    :return: 一个vgg模块
    """
    block = []
    for i in range(conv_number):
        block.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))  # 每一层卷积使用相同卷积核
        block.append(nn.LeakyReLU(0.2, inplace=True))
        in_channels = out_channels  # 第一层将输出通道改为设定out_channels此后每一层的通道不变
    block.append(nn.MaxPool2d(kernel_size=2, stride=2))  # 卷积层结束后加上最大池化层矩阵size减半
    return nn.Sequential(*block)


conv_arch = ((1, 3, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512))
fc_features = 512 * 3 * 3  # c * w * h 根据最终flatten的大小决定


class Discriminator(nn.Module):  # 判别网络
    def __init__(self):
        super(Discriminator, self).__init__()

        vgg = []
        for conv_number, in_channels, out_channels in conv_arch:
            vgg.append(vgg_block(conv_number, in_channels, out_channels))  # 五个卷积块

        self.model = nn.Sequential(
            *vgg, nn.Flatten(),
            nn.Linear(fc_features, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        # img_flat = img.view(img.size(0), -1)  # img_flat.shape=(125,1*28*28)
        validity = self.model(img)

        return validity


# 定义了一个损失函数nn.BCELoss()，输入（X，Y）, X 需要经过sigmoid, Y元素的值只能是0或1的float值，依据这个损失函数来计算损失。
# Loss function
adversarial_loss = torch.nn.BCELoss()

# 初始化生成器和判别器
# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    # 也可以判断是否为conv2d，使用相应的初始化方式 
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


generator.apply(weight_init)
discriminator.apply(weight_init)

# 4) 封装之前要把模型移到对应的gpu
generator.to(device)
discriminator.to(device)
adversarial_loss.to(device)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # 5) 封装
    generator = torch.nn.parallel.DistributedDataParallel(generator,
                                                          device_ids=[rank],
                                                          output_device=rank)
    discriminator = torch.nn.parallel.DistributedDataParallel(discriminator,
                                                              device_ids=[rank],
                                                              output_device=rank)

# 定义了神经网络的优化器，Adam就是一种优化器
# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr,
                               betas=(opt.b1, opt.b2))  # betas: 用于计算梯度以及梯度平方的运行平均值的系数
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
Tensor = torch.cuda.FloatTensor  # 将tensor数据放到GPU

for epoch in range(opt.n_epochs):  # 训练的次数就是opt.n_epochs，epoch：1个epoch表示过了1遍训练集中的所有样本。
    for i, (imgs, _) in enumerate(train_iter):
        '''
        dataloader中的数据是一张图片对应一个标签，所以imgs对应的是图片，_对应的是标签，而i是enumerate输出的功能，
        enumerate用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用
        在 for 循环当中，所以i就是相当于1,2,3…..的数据下标。
        '''
        # Adversarial ground truths
        # vaild可以想象成是64行1列的向量，就是为了在后面计算损失时，和1比较；fake也是一样是全为0的向量，用法和1的用法相同
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        # 将真实的图片转化为神经网络可以处理的变量
        real_imgs = Variable(imgs.type(Tensor))

        '''
        训练生成网络
        '''
        # -----------------
        #  Train Generator
        # -----------------

        # 在每次的训练之前都将上一次的梯度置为零，以避免上一次的梯度的干扰
        optimizer_G.zero_grad()

        # Sample noise as generator input
        # 输入从0到1之间，形状为imgs.shape[0], opt.latent_dim的随机高斯数据。
        z = Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim, 1, 1)))

        # Generate a batch of images
        # 开始得到一个批次的图片，上面说了这些数据是分批进行训练，每一批是64张，所以，这这一批图片为64张。
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        # 计算生成器的损失
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        # 进行反向传播和模型更新
        g_loss.backward()
        optimizer_G.step()

        '''
        训练判别网络
        '''
        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()
        # 在每次的训练之前都将上一次的梯度置为零，以避免上一次的梯度的干扰

        # Measure discriminator's ability to classify real from generated samples
        # 衡量判别器分类能力(论文公式)
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        # 进行反向传播和模型更新
        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(train_iter), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(train_iter) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images_test/%d.png" % batches_done, nrow=5, normalize=True)
