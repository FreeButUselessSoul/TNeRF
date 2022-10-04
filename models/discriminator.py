import torch
import torch.nn as nn
import random


class discriminator(nn.Module):
    def __init__(self, ndf=16, imsize=64):
        super(discriminator, self).__init__()
        assert (imsize == 16 or imsize == 32 or imsize == 64 or imsize == 128)

        nc = 3
        self.imsize = imsize

        SN = torch.nn.utils.spectral_norm
        IN = lambda x: nn.InstanceNorm2d(x)
        BN = torch.nn.BatchNorm2d

        final_dim = 1

        blocks = []
        if self.imsize == 128:
            blocks += [
                # input is (nc) x 128 x 128
                SN(nn.Conv2d(nc, ndf // 2, (4, 4), (2, 2), (1, 1), bias=False)),
                nn.LeakyReLU(0.2, inplace=True),
                # input is (ndf//2) x 64 x 64
                SN(nn.Conv2d(ndf // 2, ndf, (4, 4), (2, 2), (1, 1), bias=False)),
                IN(ndf),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                SN(nn.Conv2d(ndf, ndf * 2, (4, 4), (2, 2), (1, 1), bias=False)),
                # nn.BatchNorm2d(ndf * 2),
                IN(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                SN(nn.Conv2d(ndf * 2, ndf * 4, (4, 4), (2, 2), (1, 1), bias=False)),
                # nn.BatchNorm2d(ndf * 4),
                IN(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        elif self.imsize == 64:
            blocks += [
                # BN(nc),
                # input is (nc) x 64 x 64
                SN(nn.Conv2d(nc, ndf, (3, 3), (1, 1), (1, 1), bias=False)),
                # BN(ndf),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                SN(nn.Conv2d(ndf, ndf * 2, (3, 3), (1, 1), (1, 1), bias=False)),
                # BN(ndf * 2),
                # nn.BatchNorm2d(ndf * 2),
                IN(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                SN(nn.Conv2d(ndf * 2, ndf * 4, (3, 3), (1, 1), (1, 1), bias=False)),
                # BN(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        elif self.imsize == 32:
            blocks += [
                # input is (nc) x 32 x 32
                SN(nn.Conv2d(nc, ndf * 2, (4, 4), (2, 2), (1, 1), bias=False)),
                # nn.BatchNorm2d(ndf * 2),
                IN(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                SN(nn.Conv2d(ndf * 2, ndf * 4, (4, 4), (2, 2), (1, 1), bias=False)),
                # nn.BatchNorm2d(ndf * 4),
                IN(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        else:
            blocks += [
                # state size. (ndf*2) x 16 x 16
                SN(nn.Conv2d(nc, ndf * 4, (4, 4), (2, 2), (1, 1), bias=False)),
                # nn.BatchNorm2d(ndf * 4),
                IN(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        blocks += [
            # state size. (ndf*4) x 8 x 8
            SN(nn.Conv2d(ndf * 4, ndf * 8, (3, 3), (1, 1), (1, 1), bias=False)),
            # nn.BatchNorm2d(ndf * 8),
            IN(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            SN(nn.Conv2d(ndf * 8, final_dim, (1, 1), (1, 1), (0, 0), bias=False)),
            nn.Sigmoid()
        ]
        blocks = [x for x in blocks if x]
        self.main = nn.Sequential(*blocks)

    def forward(self, input, y=None):
        input = input.contiguous()

        input = self.main(input)  # [N, c1, 1, 1]

        return input.squeeze(-1).squeeze(-1)
