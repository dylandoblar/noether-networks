import torch
import torch.nn as nn
from models.cn import CNLayer


class vgg_layer(nn.Module):
    def __init__(self, nin, nout, use_cn_layers=False, batch_size=32):
        super(vgg_layer, self).__init__()
        if use_cn_layers:
            self.main = nn.Sequential(
                nn.Conv2d(nin, nout, 3, 1, 1),
                nn.BatchNorm2d(nout),
                CNLayer(shape=(batch_size, nout, 1, 1)),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            self.main = nn.Sequential(
                nn.Conv2d(nin, nout, 3, 1, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True)
            )


    def forward(self, input):
        return self.main(input)


class encoder(nn.Module):
    def __init__(self, dim, nc=1, use_cn_layers=True, batch_size=16):
        super(encoder, self).__init__()
        self.dim = dim
        # 64 x 64
        self.c1 = nn.Sequential(
                vgg_layer(nc, 64, use_cn_layers, batch_size=batch_size),
                vgg_layer(64, 64, use_cn_layers, batch_size=batch_size),
                )
        # 32 x 32
        self.c2 = nn.Sequential(
                vgg_layer(64, 128, use_cn_layers, batch_size=batch_size),
                vgg_layer(128, 128, use_cn_layers, batch_size=batch_size),
                )
        # 16 x 16 
        self.c3 = nn.Sequential(
                vgg_layer(128, 256, use_cn_layers, batch_size=batch_size),
                vgg_layer(256, 256, use_cn_layers, batch_size=batch_size),
                vgg_layer(256, 256, use_cn_layers, batch_size=batch_size),
                )
        # 8 x 8
        self.c4 = nn.Sequential(
                vgg_layer(256, 512, use_cn_layers, batch_size=batch_size),
                vgg_layer(512, 512, use_cn_layers, batch_size=batch_size),
                vgg_layer(512, 512, use_cn_layers, batch_size=batch_size),
                )
        # 4 x 4
        self.c5 = nn.Sequential(
                nn.Conv2d(512, dim, 4, 1, 0),
                nn.BatchNorm2d(dim),
                CNLayer(shape=(batch_size, dim, 1, 1)),
                nn.Tanh()
                ) if use_cn_layers else nn.Sequential(
                nn.Conv2d(512, dim, 4, 1, 0),
                nn.BatchNorm2d(dim),
                nn.Tanh()
                )
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)


    def forward(self, input):
        h1 = self.c1(input) # 64 -> 32
        h2 = self.c2(self.mp(h1)) # 32 -> 16
        h3 = self.c3(self.mp(h2)) # 16 -> 8
        h4 = self.c4(self.mp(h3)) # 8 -> 4
        h5 = self.c5(self.mp(h4)) # 4 -> 1
        return h5.view(-1, self.dim), [h1, h2, h3, h4]


class decoder(nn.Module):
    def __init__(self, dim, nc=1, use_cn_layers=True, batch_size=16):
        super(decoder, self).__init__()
        self.dim = dim
        # 1 x 1 -> 4 x 4
        self.upc1 = nn.Sequential(
                nn.ConvTranspose2d(dim, 512, 4, 1, 0),
                nn.BatchNorm2d(512),
                CNLayer(shape=(batch_size, 512, 1, 1)),
                nn.LeakyReLU(0.2, inplace=True)
                ) if use_cn_layers else nn.Sequential(
                nn.ConvTranspose2d(dim, 512, 4, 1, 0),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True)
                )
        # 8 x 8
        self.upc2 = nn.Sequential(
                vgg_layer(512*2, 512, use_cn_layers, batch_size=batch_size),
                vgg_layer(512, 512, use_cn_layers, batch_size=batch_size),
                vgg_layer(512, 256, use_cn_layers, batch_size=batch_size)
                )
        # 16 x 16
        self.upc3 = nn.Sequential(
                vgg_layer(256*2, 256, use_cn_layers, batch_size=batch_size),
                vgg_layer(256, 256, use_cn_layers, batch_size=batch_size),
                vgg_layer(256, 128, use_cn_layers, batch_size=batch_size)
                )
        # 32 x 32
        self.upc4 = nn.Sequential(
                vgg_layer(128*2, 128, use_cn_layers, batch_size=batch_size),
                vgg_layer(128, 64, use_cn_layers, batch_size=batch_size)
                )
        # 64 x 64
        self.upc5 = nn.Sequential(
                vgg_layer(64*2, 64, use_cn_layers, batch_size=batch_size),
                nn.ConvTranspose2d(64, nc, 3, 1, 1),
                nn.Sigmoid()
                )
        self.up = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, input):
        vec, skip = input
        d1 = self.upc1(vec.view(-1, self.dim, 1, 1)) # 1 -> 4
        up1 = self.up(d1) # 4 -> 8
        d2 = self.upc2(torch.cat([up1, skip[3]], 1)) # 8 x 8
        up2 = self.up(d2) # 8 -> 16 
        d3 = self.upc3(torch.cat([up2, skip[2]], 1)) # 16 x 16 
        up3 = self.up(d3) # 8 -> 32 
        d4 = self.upc4(torch.cat([up3, skip[1]], 1)) # 32 x 32
        up4 = self.up(d4) # 32 -> 64
        output = self.upc5(torch.cat([up4, skip[0]], 1)) # 64 x 64
        return output
