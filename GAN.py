
# coding: utf-8

# In[ ]:


import argparse

parser = argparse.ArgumentParser("cDCGAN")

parser.add_argument('--dataset_dir', type=str, default='../celeba')
parser.add_argument('--result_dir', type=str, default='./celeba_result')
parser.add_argument('--condition_file', type=str, default='./list_attr_celeba.txt')

parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--nepoch', type=int, default=20)
parser.add_argument('--nz', type=int, default=100) # number of noise dimension
parser.add_argument('--nc', type=int, default=3) # number of result channel
parser.add_argument('--nfeature', type=int, default=40)
parser.add_argument('--lr', type=float, default=0.0002)
betas = (0.0, 0.99) # adam optimizer beta1, beta2

config, _ = parser.parse_known_args()


# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torch.autograd import Variable


# In[ ]:


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(config.nz + config.nfeature, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, config.nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
    
    def forward(self, x, attr):
        attr = attr.view(-1, config.nfeature, 1, 1)
        x = torch.cat([x, attr], 1)
        return self.main(x)


# In[ ]:


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.feature_input = nn.Linear(config.nfeature, 64 * 64)
        self.main = nn.Sequential(
            nn.Conv2d(config.nc + 1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
        )
    
    def forward(self, x, attr):
        attr = self.feature_input(attr).view(-1, 1, 64, 64)
        x = torch.cat([x, attr], 1)
        return self.main(x).view(-1, 1)


# In[ ]:


class Trainer:
    def __init__(self):
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.loss = nn.MSELoss()
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=config.lr, betas=betas)
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=config.lr, betas=betas)
        
        self.generator.cuda()
        self.discriminator.cuda()
        self.loss.cuda()        
        
    def train(self, dataloader):
        noise = Variable(torch.FloatTensor(config.batch_size, config.nz, 1, 1).cuda())
        label_real = Variable(torch.FloatTensor(config.batch_size, 1).fill_(1).cuda())
        label_fake = Variable(torch.FloatTensor(config.batch_size, 1).fill_(0).cuda())
        for epoch in range(config.nepoch):
            for i, (data, attr) in enumerate(dataloader, 0):
                # train discriminator
                self.discriminator.zero_grad()

                batch_size = data.size(0)
                label_real.data.resize(batch_size, 1).fill_(1)
                label_fake.data.resize(batch_size, 1).fill_(0)
                noise.data.resize_(batch_size, config.nz, 1, 1).normal_(0, 1)
                
                attr = Variable(attr.cuda())
                real = Variable(data.cuda())
                d_real = self.discriminator(real, attr)

                fake = self.generator(noise, attr)
                d_fake = self.discriminator(fake.detach(), attr) # not update generator
                
                d_loss = self.loss(d_real, label_real) + self.loss(d_fake, label_fake) # real label
                d_loss.backward()
                self.optimizer_d.step()

                # train generator
                self.generator.zero_grad()
                d_fake = self.discriminator(fake, attr)
                g_loss = self.loss(d_fake, label_real) # trick the fake into being real
                g_loss.backward()
                self.optimizer_g.step()
            print("epoch{:03d} d_real: {}, d_fake: {}".format(epoch, d_real.mean(), d_fake.mean()))
            vutils.save_image(fake.data, '{}/result_epoch_{:03d}.png'.format(config.result_dir, epoch), normalize=True)


# In[ ]:


import torch.utils.data
from ImageFeatureFolder import ImageFeatureFolder

dataset = ImageFeatureFolder(config.dataset_dir, config.condition_file, transform=transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
                           
trainer = Trainer()
trainer.train(dataloader)

