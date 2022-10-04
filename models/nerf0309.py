from models.encoder import SpatialEncoder
import torch
from torch import nn
from .encoder import *

class PosEmbedding(nn.Module):
    def __init__(self, max_logscale, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        """
        super().__init__()
        self.funcs = [torch.sin, torch.cos]

        if logscale:
            self.freqs = 2**torch.linspace(0, max_logscale, N_freqs)
        else:
            self.freqs = torch.linspace(1, 2**max_logscale, N_freqs)

    def forward(self, x):
        """
        Inputs:
            x: (B, 3)

        Outputs:
            out: (B, 6*N_freqs+3)
        """
        out = [x]
        for freq in self.freqs:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)

def repeat_interleave(input, repeats, dim=0):
    """
    Repeat interleave along axis 0
    torch.repeat_interleave is currently very slow
    https://github.com/pytorch/pytorch/issues/31980
    """
    output = input.unsqueeze(1).expand(-1, repeats, *input.shape[1:])
    return output.reshape(-1, *input.shape[1:])

class NeRF(nn.Module):
    def __init__(self, typ,
                 D=8, W=256, skips=[4],
                 in_channels_xyz=63, in_channels_dir=27,
                 encode_appearance=False, in_channels_a=48,
                 encode_transient=False, in_channels_t=16, in_channels_f=512,
                 beta_min=0.03):
        """
        ---Parameters for the original NeRF---
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        skips: add skip connection in the Dth layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        in_channels_t: number of input channels for t
        in_channels_f: number of features, which will be concatenated to xyz.

        ---Parameters for NeRF-W (used in fine model only as per section 4.3)---
        ---cf. Figure 3 of the paper---
        encode_appearance: whether to add appearance encoding as input (NeRF-A)
        in_channels_a: appearance embedding dimension. n^(a) in the paper
        encode_transient: whether to add transient encoding as input (NeRF-U)
        in_channels_t: transient embedding dimension. n^(tau) in the paper
        beta_min: minimum pixel color variance
        """
        super().__init__()
        self.typ = typ
        self.D = D
        self.W = W
        self.skips = skips
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir

        self.encode_appearance = False if typ=='coarse' else encode_appearance
        self.in_channels_a = in_channels_a if encode_appearance else 0
        self.encode_transient = False if typ=='coarse' else encode_transient
        self.in_channels_t = in_channels_t
        self.in_channels_f = in_channels_f
        self.beta_min = beta_min

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ELU(inplace=True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)
        # self.encoder = SpatialEncoder() # spatial encoder
        # direction encoding layers
        self.dir_encoding = nn.Sequential(
                        nn.Linear(W+in_channels_dir+self.in_channels_a, W//2), nn.ELU(inplace=True))

        # static output layers
        self.static_sigma = nn.Sequential(nn.Linear(W, 1), nn.Softplus())
        # self.static_rgb = nn.Sequential(nn.Linear(in_channels_f+W//2, 3), nn.Sigmoid())
        self.static_rgb = nn.Sequential(nn.Linear(in_channels_f+W//2, W//2),nn.ELU(inplace=True),
                                        nn.Linear(W//2, W//2),nn.ELU(inplace=True),
                                        nn.Linear(W//2, 3),nn.Sigmoid())

        if self.encode_transient:
            # transient encoding layers
            self.transient_encoding = nn.Sequential(
                                        nn.Linear(W+in_channels_t, W//2), nn.ELU(inplace=True),
                                        nn.Linear(W//2, W//2), nn.ELU(inplace=True),
                                        nn.Linear(W//2, W//2), nn.ELU(inplace=True),
                                        nn.Linear(W//2, W//2), nn.ELU(inplace=True))
            # transient output layers
            self.transient_sigma = nn.Sequential(nn.Linear(W//2, 1), nn.Softplus())
            self.transient_rgb = nn.Sequential(nn.Linear(W//2, 3), nn.Sigmoid())
            self.transient_beta = nn.Sequential(nn.Linear(W//2, 1), nn.Softplus())

    def encode(self, images, poses, focal):
        """
        :param images (NS, 3, H, W)
        NS is number of input (aka source or reference) views
        :param poses (NS, 4, 4)
        :param focal focal length () or (2) or (NS) or (NS, 2) [fx, fy]
        default is center of image
        """
        self.num_objs = images.size(0)
        if len(images.shape) == 5:
            assert len(poses.shape) == 4
            assert poses.size(1) == images.size(
                1
            )  # Be consistent with NS = num input views
            self.num_views_per_obj = images.size(1)
            images = images.reshape(-1, *images.shape[2:])
            poses = poses.reshape(-1, 4, 4)
        else:
            self.num_views_per_obj = 1

        # self.encoder(images)
        rot = poses[:, :3, :3].transpose(1, 2)  # (B, 3, 3)
        trans = -torch.bmm(rot, poses[:, :3, 3:])  # (B, 3, 1)
        self.poses = torch.cat((rot, trans), dim=-1)  # (B, 3, 4)

        self.image_shape[0] = images.shape[-1]
        self.image_shape[1] = images.shape[-2]

        # Handle various focal length/principal point formats
        if len(focal.shape) == 0:
            # Scalar: fx = fy = value for all views
            focal = focal[None, None].repeat((1, 2))
        elif len(focal.shape) == 1:
            # Vector f: fx = fy = f_i *for view i*
            # Length should match NS (or 1 for broadcast)
            focal = focal.unsqueeze(-1).repeat((1, 2))
        else:
            focal = focal.clone()
        self.focal = focal.float()
        self.focal[..., 1] *= -1.0

        # Default principal point is center of image
        self.c = (self.image_shape * 0.5).unsqueeze(0)

    def forward(self, x, sigma_only=False, output_transient=True):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: the embedded vector of position (+ direction + appearance + transient)
            sigma_only: whether to infer sigma only.
            has_transient: whether to infer the transient component.

        Outputs (concatenated):
            if sigma_ony:
                static_sigma
            elif output_transient:
                static_rgb, static_sigma, transient_rgb, transient_sigma, transient_beta
            else:
                static_rgb, static_sigma
        """
        if sigma_only:
            input_xyz, code_B = torch.split(x,[self.in_channels_xyz,self.in_channels_f],dim=-1)
        elif output_transient:
            input_xyz, code_B, input_dir_a, input_t = \
                torch.split(x, [self.in_channels_xyz, self.in_channels_f,
                                self.in_channels_dir+self.in_channels_a,
                                self.in_channels_t], dim=-1)
        else:
            input_xyz, code_B, input_dir_a = \
                torch.split(x, [self.in_channels_xyz, self.in_channels_f,
                                self.in_channels_dir+self.in_channels_a], dim=-1)
            

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], 1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        static_sigma = self.static_sigma(xyz_) # (B, 1)
        if sigma_only:
            return static_sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)
        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir_a], 1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        static_input = torch.cat([dir_encoding, code_B], 1)

        static_rgb = self.static_rgb(static_input)
        static = torch.cat([static_rgb, static_sigma], 1) # (B, 4)

        if not output_transient:
            return static

        transient_encoding_input = torch.cat([xyz_encoding_final, input_t], 1)
        transient_encoding = self.transient_encoding(transient_encoding_input)
        transient_sigma = self.transient_sigma(transient_encoding) # (B, 1)
        transient_rgb = self.transient_rgb(transient_encoding) # (B, 3)
        transient_beta = self.transient_beta(transient_encoding) # (B, 1)

        transient = torch.cat([transient_rgb, transient_sigma,
                               transient_beta], 1) # (B, 5)

        return torch.cat([static, transient], 1) # (B, 9)