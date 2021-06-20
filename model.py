"""
Discriminator and Generator implementation from DCGAN paper
"""

import torch
import torch.nn as nn
from torch.nn.modules.activation import Sigmoid
import my_config

class Discriminator(nn.Module):
    def __init__(self, Model):
        super(Discriminator, self).__init__()
        # base number of layers at the start 
        num_of_layers=Model['features_multiplyer']

        self.net = nn.Sequential( 
            nn.Conv2d(Model['num_of_output_channels'], num_of_layers, kernel_size=4, stride=2, padding=1),#64 16*16
            nn.LeakyReLU(0.2),
            self.conv_block(num_of_layers, num_of_layers * 2, 4, 2, 1,Model),#128 8*8
            self.conv_block(num_of_layers * 2, num_of_layers * 4, 4, 2, 1,Model),#256 4*4
            # After all conv_block  output is 4x4 (Conv2d below makes into 1x1)
            nn.Conv2d(num_of_layers * 4, 1, kernel_size=4, stride=2, padding=0)
        )

        self.model=Model
        if self.model['name']=='DCgan':
              self.criterion = nn.BCELoss()
              self.sigmod=nn.Sigmoid()

    def conv_block(self, in_channels, out_channels, kernel_size, stride, padding,Model):
            if Model['name'] == 'Wgan-GP': 
              return   nn.Sequential( nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.InstanceNorm2d(out_channels, affine=True),
                nn.LeakyReLU(0.2))
            else:
             return    nn.Sequential( nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2))


             
                
  
    def forward(self, x):
        output= self.net(x)
        if self.model['name']== 'Wgan-GP':
            return output
        else : 
            return self.sigmod(output)
      

    ## regular function -> not belong to class neighter to instance 
    @staticmethod
    def calculate_disc_loss(disc, real, fake):
        if disc.model['name']=="DCgan":
             disc_real = disc(real).reshape(-1)
             # we use the binary cross entropy loss for calculation when we set the labales zero s and once accordingly and those get 
             #- ( 1*log(d(x)) +(1-0) log(1-d(x)) -> - ( log(d(x)) +log(1-d(x))
             #  when the negative  sign is fitted because we take min of - instad maximaztion 
             loss_disc_real = disc.criterion(disc_real, torch.ones_like(disc_real))
             disc_fake = disc(fake).reshape(-1)
             loss_disc_fake = disc.criterion(disc_fake, torch.zeros_like(disc_fake))
             loss_disc = (loss_disc_real + loss_disc_fake) / 2
             return loss_disc

        else:
                # Train disc: max E[disc(real)] - E[disc(fake)]
                 # equivalent to minimizing the negative of that
            def gradient_penalty(disc, real, fake):
                disc.to(disc.model['device'])
                BATCH_SIZE, C, H, W = real.shape
                alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(disc.model['device'])
                interpolated_images = real * alpha + fake * (1 - alpha)

                # Calculate disc scores
                mixed_scores = disc(interpolated_images)

                # Take the gradient of the scores with respect to the images
                gradient = torch.autograd.grad(
                    inputs=interpolated_images,
                    outputs=mixed_scores,
                    grad_outputs=torch.ones_like(mixed_scores),
                    create_graph=True,
                    retain_graph=True,
                )[0]
                gradient = gradient.view(gradient.shape[0], -1)
                gradient_norm = gradient.norm(2, dim=1)
                gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
                return gradient_penalty


            disc_real = disc(real).reshape(-1)
            disc_fake = disc(fake).reshape(-1)
            gp = gradient_penalty(disc, real, fake)
            ## minimize the mean of the diatrbution od fake and real image + gp  -> loss function 
            loss_disc = (-(  torch.mean(disc_real) - torch.mean(disc_fake) ) + disc.model['lambada_gp'] * gp)
            return loss_disc


class Generator(nn.Module):
    def __init__(self,Model):
        super(Generator, self).__init__()
        self.model=Model
        features_multiplyer=self.model['features_multiplyer']
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self.conv_block(Model['z_dim'], features_multiplyer * 8, 4, 1, 0,Model),  # img: 4x4
            self.conv_block(features_multiplyer * 8, features_multiplyer * 4, 4, 2, 1,Model),  # img: 8x8
            self.conv_block(features_multiplyer * 4, features_multiplyer * 2, 4, 2, 1,Model),  # img: 16x16
            nn.ConvTranspose2d(
                features_multiplyer * 2, Model['num_of_input_channels'], kernel_size=4, stride=2, padding=1# img: 32x32
            ),
            # Output: N x channels_img x 64 x 64
            nn.Tanh(),
        )
    def conv_block(self, in_channels, out_channels, kernel_size, stride, padding,Model):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        return self.net(x)

    @staticmethod
    def calculate_gen_loss(disc,fake):
    
        gen_fake = disc(fake).reshape(-1)
        if disc.model['name']=="DCgan":
            ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))<->min -log(D(G(z))
            loss_gen = disc.criterion(gen_fake, torch.ones_like(gen_fake))
        else:
            # Train Generator: max E[disc(gen_fake)] <-> min -E[disc(gen_fake)]
            gen_fake = disc(fake).reshape(-1)
            loss_gen = -torch.mean(gen_fake)
        return loss_gen


def test():
    Model=my_config.WGAN
    N, in_channels, H, W = 8, 1, 32,32
    noise_dim = 128
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(Model)
    assert disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"
    gen = Generator(Model)
    z = torch.randn((N, noise_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"

