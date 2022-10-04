import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from kornia.utils import create_meshgrid

def extract_model_state_dict(ckpt_path, model_name='model', prefixes_to_ignore=[]):
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    checkpoint_ = {}
    if 'state_dict' in checkpoint: # if it's a pytorch-lightning checkpoint
        checkpoint = checkpoint['state_dict']
    for k, v in checkpoint.items():
        if not k.startswith(model_name):
            continue
        k = k[len(model_name)+1:]
        for prefix in prefixes_to_ignore:
            if k.startswith(prefix):
                print('ignore', k)
                break
        else:
            checkpoint_[k] = v
    return checkpoint_

def load_ckpt(model, ckpt_path, model_name='model', prefixes_to_ignore=[]):
    model_dict = model.state_dict()
    checkpoint_ = extract_model_state_dict(ckpt_path, model_name, prefixes_to_ignore)
    model_dict.update(checkpoint_)
    model.load_state_dict(model_dict)

class Enc0(nn.Module):
    def __init__(self,n_channels,bilinear):
        super(Enc0, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        def down(in_channels, out_channels):
            return nn.Sequential(
                double_conv(in_channels, out_channels),
                nn.MaxPool2d(2) # MaxPool first
            )

        # self.inc = double_conv(self.n_channels, 64)
        # self.down1 = down(64, 128)
        # self.down2 = down(128, 256)
        # self.down3 = down(256, 512)
        # self.down4 = down(512, 512)
        self.inc = double_conv(self.n_channels, 32)
        self.down1 = down(32, 64)
        self.down2 = down(64, 96)
        self.down3 = down(96, 128)
        self.down4 = down(128, 192)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x5,x4,x3,x2,x1

#ERRNET
class PyramidPooling(nn.Module):
    def __init__(self, in_channels, out_channels, scales=(4, 8, 16, 32), ct_channels=1):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_channels, scale, ct_channels) for scale in scales])
        self.bottleneck = nn.Conv2d(in_channels + len(scales) * ct_channels, out_channels, kernel_size=1, stride=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def _make_stage(self, in_channels, scale, ct_channels):
        # prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        prior = nn.AvgPool2d(kernel_size=(scale, scale))
        conv = nn.Conv2d(in_channels, ct_channels, kernel_size=1, bias=False)
        relu = nn.LeakyReLU(0.2, inplace=True)
        return nn.Sequential(prior, conv, relu)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = torch.cat([F.interpolate(input=stage(feats), size=(h, w), mode='nearest') for stage in self.stages] + [feats], dim=1)
        return self.relu(self.bottleneck(priors))


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        
        return x * y        
     

class DRNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, n_feats, n_resblocks, norm=nn.BatchNorm2d, 
    se_reduction=None, res_scale=1, bottom_kernel_size=3, pyramid=False):
        super(DRNet, self).__init__()
        # Initial convolution layers
        conv = nn.Conv2d
        deconv = nn.ConvTranspose2d
        act = nn.ReLU(True)
        
        self.pyramid_module = None
        self.conv1 = ConvLayer(conv, in_channels, n_feats, kernel_size=bottom_kernel_size, stride=1, norm=None, act=act)
        self.conv2 = ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=1, norm=norm, act=act)
        self.conv3 = ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=2, norm=norm, act=act)

        # Residual layers
        dilation_config = [1] * n_resblocks

        self.res_module = nn.Sequential(*[ResidualBlock(
            n_feats, dilation=dilation_config[i], norm=norm, act=act, 
            se_reduction=se_reduction, res_scale=res_scale) for i in range(n_resblocks)])

        # Upsampling Layers
        self.deconv1 = ConvLayer(deconv, n_feats, n_feats, kernel_size=4, stride=2, padding=1, norm=norm, act=act)

        if not pyramid:
            self.deconv2 = ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=1, norm=norm, act=act)
            self.deconv3 = ConvLayer(conv, n_feats, out_channels, kernel_size=1, stride=1, norm=None, act=act)
        else:
            self.deconv2 = ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=1, norm=norm, act=act)
            self.pyramid_module = PyramidPooling(n_feats, n_feats, scales=(4,8,16,32), ct_channels=n_feats//4)
            self.deconv3 = ConvLayer(conv, n_feats, out_channels, kernel_size=1, stride=1, norm=None, act=act)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.res_module(x)

        x2 = self.deconv1(x)
        x2 = self.deconv2(x2)
        if self.pyramid_module is not None:
            x2 = self.pyramid_module(x2)
        code_B = x2
        x2 = self.deconv3(code_B)

        return x,code_B, x2


class ConvLayer(torch.nn.Sequential):
    def __init__(self, conv, in_channels, out_channels, kernel_size, stride, padding=None, dilation=1, norm=None, act=None):
        super(ConvLayer, self).__init__()
        # padding = padding or kernel_size // 2
        padding = padding or dilation * (kernel_size - 1) // 2
        self.add_module('conv2d', conv(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation))
        if norm is not None:
            self.add_module('norm', norm(out_channels))
            # self.add_module('norm', norm(out_channels, track_running_stats=True))
        if act is not None:
            self.add_module('act', act)


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels, dilation=1, norm=nn.BatchNorm2d, act=nn.ReLU(True), se_reduction=None, res_scale=1):
        super(ResidualBlock, self).__init__()
        conv = nn.Conv2d
        self.conv1 = ConvLayer(conv, channels, channels, kernel_size=3, stride=1, dilation=dilation, norm=norm, act=act)
        self.conv2 = ConvLayer(conv, channels, channels, kernel_size=3, stride=1, dilation=dilation, norm=norm, act=None)
        self.se_layer = None
        self.res_scale = res_scale
        if se_reduction is not None:
            self.se_layer = SELayer(channels, se_reduction)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.se_layer:
            out = self.se_layer(out)
        out = out * self.res_scale
        out = out + residual
        return out

    def extra_repr(self):
        return 'res_scale={}'.format(self.res_scale)



class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        self.vgg_pretrained_features = models.vgg19(pretrained=True).features
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
    def forward(self, X, indices=None):
        if indices is None:
            indices = [2, 7, 12, 21, 30]
        out = []
        for i in range(indices[-1]):
            X = self.vgg_pretrained_features[i](X)
            if (i+1) in indices:
                out.append(X)
        
        return out

class self_attn(nn.Module):
    def __init__(self,n_channels):
        super(self_attn, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(n_channels,n_channels*2,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels*2,n_channels,kernel_size=1,padding=0),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        output = self.convs(x) * x
        return output

class double_conv(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(double_conv,self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ELU(inplace=True)
        )
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    def forward(self,x):
        x = self.conv1(x)
        return x+self.conv2(x)


class Enc(nn.Module):
    def __init__(self,n_channels,bilinear):
        super(Enc, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        def down(in_channels, out_channels):
            return nn.Sequential(
                double_conv(in_channels, out_channels),
                nn.MaxPool2d(2) # MaxPool first
            )

        # self.inc = double_conv(self.n_channels, 64)
        # self.down1 = down(64, 128)
        # self.down2 = down(128, 256)
        # self.down3 = down(256, 512)
        # self.down4 = down(512, 512)
        self.inc = double_conv(self.n_channels, 32)
        self.down1 = down(32, 64)
        self.down2 = down(64, 96)
        self.down3 = down(96, 128)
        self.down4 = down(128, 192)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x5,x4,x3,x2,x1

class Dec(nn.Module):
    def __init__(self,n_classes,bilinear):
        super(Dec, self).__init__()
        self.n_classes = n_classes
        self.bilinear = bilinear
        # def double_conv(in_channels, out_channels):
        #     return nn.Sequential(
        #         nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        #         nn.BatchNorm2d(out_channels),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        #         nn.BatchNorm2d(out_channels),
        #         nn.ReLU(inplace=True),
        #     )

        class up(nn.Module):
            def __init__(self, in_channels, out_channels, bilinear=True):
                super().__init__()

                if bilinear:
                    self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                else:
                    self.up = nn.ConvTranpose2d(in_channels // 2, in_channels // 2,
                                                kernel_size=2, stride=2)

                self.conv = double_conv(in_channels, out_channels)

            def forward(self, x1, x2):
                x1 = self.up(x1)
                # [?, C, H, W]
                diffY = x2.size()[2] - x1.size()[2]
                diffX = x2.size()[3] - x1.size()[3]

                x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2])
                x = torch.cat([x2, x1], dim=1) ## why 1?
                return self.conv(x)

        # self.up1 = up(1024, 256)
        # self.up2 = up(512, 128)
        # self.up3 = up(256, 64)
        # self.up4 = up(128, 64)
        # self.out = nn.Conv2d(64, self.n_classes, kernel_size=1)
        self.up1 = up(192+128, 128)
        self.up2 = up(128+96, 96)
        self.up3 = up(96+64, 64)
        self.up4 = up(96, 32)
        self.out = nn.Conv2d(32, self.n_classes, kernel_size=1)

    def forward(self, x5,x4,x3,x2,x1):
        x4_ = self.up1(x5, x4)
        x3_ = self.up2(x4_, x3)
        x2_ = self.up3(x3_, x2)
        x1_ = self.up4(x2_, x1) # Use this!
        return self.out(x1_),x1_,x2_,x3_,x4_

class UNet(nn.Module):
    def __init__(self,n_channels,n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        # self.n_channels += 1472
        self.bilinear = True
        self.enc = Enc(self.n_channels,self.bilinear)
        self.dec = Dec(self.n_classes,self.bilinear)
        # self.enc = ResNetEncoder(EncoderBlock, [2, 2, 2, 2], False, False, 3, 256)
        # self.dec = ResNetDecoder(DecoderBlock, [2, 2, 2, 2], 256, self.hparams.img_wh[1])

    def forward(self, x):
        latent = self.enc(x)
        decodeds = self.dec(*latent)
        return latent[0],decodeds[1]# 224,decodeds[2] #192+32=224+64=288

class UNet_warping(nn.Module):
    def __init__(self,n_channels,n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        # self.n_channels += 1472
        self.bilinear = True
        self.enc = Enc(self.n_channels,self.bilinear)
        self.dec = Dec(self.n_classes,self.bilinear)
        # self.enc = ResNetEncoder(EncoderBlock, [2, 2, 2, 2], False, False, 3, 256)
        # self.dec = ResNetDecoder(DecoderBlock, [2, 2, 2, 2], 256, self.hparams.img_wh[1])

    def forward(self, x):
        latent = self.enc(x)
        decodeds = self.dec(*latent)
        return latent[0],decodeds[1]# 224,decodeds[2] #192+32=224+64=288

def homo_warp(src_feat, proj_mat, depth_values, src_grid=None, pad=0):
    """
    src_feat: (C, H, W)
    proj_mat: (3, 4) equal to "src_proj @ ref_proj_inv"
    depth_values: (H, W)
    out: (C, H, W)
    """
    if pad>0:
        src_feat = F.pad(src_feat,(pad,pad),mode='reflect')
    if src_grid==None:
        C, H, W = src_feat.shape
        device = src_feat.device
        if pad>0:
            H_pad, W_pad = H + pad*2, W + pad*2
        else:
            H_pad, W_pad = H, W

        # depth_values = depth_values[...,None,None].repeat(1, 1, H_pad, W_pad)
        # D = depth_values.shape[1]

        R = proj_mat[:, :3]  # (3, 3)
        T = proj_mat[:, 3:]  # (3, 1)
        # create grid from the ref frame
        ref_grid = create_meshgrid(H_pad, W_pad, normalized_coordinates=False, device=device)  # (1, H, W, 2)
        if pad>0:
            ref_grid -= pad

        ref_grid = ref_grid.permute(0, 3, 1, 2)  # (1, 2, H, W)
        ref_grid = ref_grid.reshape(1, 2, W_pad * H_pad)  # (1, 2, H*W)
        ref_grid = torch.cat((ref_grid, torch.ones_like(ref_grid[:, :1])), 1)  # (1, 3, H*W)
        # ref_grid_d = ref_grid.repeat(1, D)  # (3, D*H*W)
        
        src_grid_d = R @ ref_grid + T / depth_values.view(1, W_pad * H_pad)
        del ref_grid, proj_mat, R, T, depth_values  # release (GPU) memory

        src_grid = src_grid_d[0,:2] / src_grid_d[0,2:]  # divide by depth (., H*W)
        del src_grid_d
        src_grid[0] = src_grid[0] / ((W - 1) / 2) - 1  # scale to -1~1
        src_grid[1] = src_grid[1] / ((H - 1) / 2) - 1  # scale to -1~1
        # src_grid = src_grid.permute(0, 2, 1)  # (B, H*W, 2)
        src_grid = src_grid.T.view(1, H_pad, W_pad, 2)

    B, W_pad, H_pad = src_grid.shape[:3]
    warped_src_feat = F.grid_sample(src_feat.unsqueeze(0), src_grid.view(B, W_pad, H_pad, 2),
                                    mode='bilinear', padding_mode='reflection',
                                    align_corners=True)  # (C, H*W)
    warped_src_feat = warped_src_feat.view(C, W_pad,H_pad)
    # src_grid = src_grid.view(B, 1, H_pad, W_pad, 2)
    return warped_src_feat, src_grid