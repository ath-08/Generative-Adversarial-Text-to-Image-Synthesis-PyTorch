import torch
import torch.nn as nn


class Discriminator(nn.Module):
  
    class ConcatEmbeds(nn.Module):
        def __init__(self, input_dim=1024, projected_dim=128):
            super().__init__()
            self.input_dim = input_dim
            self.projected_dim = projected_dim
            self.projection_layer = nn.Sequential(
                nn.Linear(in_features=self.input_dim, out_features=self.projected_dim),
                nn.LeakyReLU(0.2)
            )
        
        def forward(self, img_embeds, txt_embeds):
            txt_embeds = self.projection_layer(txt_embeds)
            txt_embeds = txt_embeds.repeat(4, 4, 1, 1).permute(2, 3, 0, 1)
            return torch.cat([img_embeds, txt_embeds], dim=1)


    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.ndf = 64
        self.inp_txt_dim = 1024
        self.projected_txt_dim = 128
        self.conv_blocks = nn.Sequential(
            self.convolution_block(3, self.ndf, first_layer=True),
            self.convolution_block(self.ndf, self.ndf*2),
            self.convolution_block(self.ndf*2, self.ndf*4),
            self.convolution_block(self.ndf*4, self.ndf*8)
        )
        self.concat_embeds = self.ConcatEmbeds(self.inp_txt_dim, self.projected_txt_dim)
        self.final_block = nn.Sequential(
            nn.Conv2d(in_channels=self.ndf*8+self.projected_txt_dim, out_channels=self.ndf*8, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=self.ndf*8, out_channels=1, kernel_size=4),
            nn.Sigmoid()
        )

    def forward(self, img, txt_embeds):
        img = self.conv_blocks(img)
        latent_embeds = self.concat_embeds(img, txt_embeds)
        score = self.final_block(latent_embeds)
        score = score.view(-1, 1)
        return score.squeeze(1), img
  
  
    def convolution_block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, first_layer=False):
        if not first_layer:
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.LeakyReLU(0.2)
            )

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.img_size = 64
        self.num_img_channels = 3
        self.inp_txt_dim = 1024
        self.projected_txt_dim = 128
        self.noise_dim = 100
        self.ngf = 64

        self.projection_layer = nn.Sequential(
            nn.Linear(in_features=self.inp_txt_dim, out_features=self.projected_txt_dim),
            nn.LeakyReLU(0.2)
        )

        self.conv_transpose_blocks = nn.Sequential(
            self.conv_transpose_block(self.noise_dim + self.projected_txt_dim, self.ngf*8, stride=1, padding=0),
            self.conv_transpose_block(self.ngf*8, self.ngf*4),
            self.conv_transpose_block(self.ngf*4, self.ngf*2),
            self.conv_transpose_block(self.ngf*2, self.ngf),
            self.conv_transpose_block(self.ngf, self.num_img_channels, last_layer=True)
        )

    def forward(self, noise, txt_embeds):
        txt_embeds = self.projection_layer(txt_embeds)
        txt_embeds = txt_embeds.view(-1, self.projected_txt_dim, 1, 1)
        combined_embeds = torch.cat([noise, txt_embeds], dim=1)
        generated_img = self.conv_transpose_blocks(combined_embeds)
        return generated_img

    def conv_transpose_block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, last_layer=False):
        if not last_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.Tanh()
            )