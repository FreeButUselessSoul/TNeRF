"""
Image encoders
"""
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import functools
import math
import torch.autograd.profiler as profiler

def calc_same_pad_conv2d(t_shape, kernel_size=3, stride=1):
    in_height, in_width = t_shape[-2:]
    out_height = math.ceil(in_height / stride)
    out_width = math.ceil(in_width / stride)

    pad_along_height = max((out_height - 1) * stride + kernel_size - in_height, 0)
    pad_along_width = max((out_width - 1) * stride + kernel_size - in_width, 0)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    return pad_left, pad_right, pad_top, pad_bottom

def same_pad_conv2d(t, padding_type="reflect", kernel_size=3, stride=1, layer=None):
    """
    Perform SAME padding on tensor, given kernel size/stride of conv operator
    assumes kernel/stride are equal in all dimensions.
    Use before conv called.
    Dilation not supported.
    :param t image tensor input (B, C, H, W)
    :param padding_type padding type constant | reflect | replicate | circular
    constant is 0-pad.
    :param kernel_size kernel size of conv
    :param stride stride of conv
    :param layer optionally, pass conv layer to automatically get kernel_size and stride
    (overrides these)
    """
    if layer is not None:
        if isinstance(layer, nn.Sequential):
            layer = next(layer.children())
        kernel_size = layer.kernel_size[0]
        stride = layer.stride[0]
    return F.pad(
        t, calc_same_pad_conv2d(t.shape, kernel_size, stride), mode=padding_type
    )

def same_unpad_deconv2d(t, kernel_size=3, stride=1, layer=None):
    """
    Perform SAME unpad on tensor, given kernel/stride of deconv operator.
    Use after deconv called.
    Dilation not supported.
    """
    if layer is not None:
        if isinstance(layer, nn.Sequential):
            layer = next(layer.children())
        kernel_size = layer.kernel_size[0]
        stride = layer.stride[0]
    h_scaled = (t.shape[-2] - 1) * stride
    w_scaled = (t.shape[-1] - 1) * stride
    pad_left, pad_right, pad_top, pad_bottom = calc_same_pad_conv2d(
        (h_scaled, w_scaled), kernel_size, stride
    )
    if pad_right == 0:
        pad_right = -10000
    if pad_bottom == 0:
        pad_bottom = -10000
    return t[..., pad_top:-pad_bottom, pad_left:-pad_right]

def get_norm_layer(norm_type="instance", group_norm_groups=32):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == "batch":
        norm_layer = functools.partial(
            nn.BatchNorm2d, affine=True, track_running_stats=True
        )
    elif norm_type == "instance":
        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False
        )
    elif norm_type == "group":
        norm_layer = functools.partial(nn.GroupNorm, group_norm_groups)
    elif norm_type == "none":
        norm_layer = None
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


class ConvEncoder(nn.Module):
    """
    Basic, extremely simple convolutional encoder
    """
    def __init__(
        self,
        dim_in=3,
        norm_layer=get_norm_layer("group"),
        padding_type="reflect",
        use_leaky_relu=True,
        use_skip_conn=True,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.norm_layer = norm_layer
        self.activation = nn.LeakyReLU() if use_leaky_relu else nn.ReLU()
        self.padding_type = padding_type
        self.use_skip_conn = use_skip_conn

        # TODO: make these configurable
        first_layer_chnls = 64
        mid_layer_chnls = 128
        last_layer_chnls = 128
        n_down_layers = 3
        self.n_down_layers = n_down_layers

        self.conv_in = nn.Sequential(
            nn.Conv2d(dim_in, first_layer_chnls, kernel_size=7, stride=2, bias=False),
            norm_layer(first_layer_chnls),
            self.activation,
        )

        chnls = first_layer_chnls
        for i in range(0, n_down_layers):
            conv = nn.Sequential(
                nn.Conv2d(chnls, 2 * chnls, kernel_size=3, stride=2, bias=False),
                norm_layer(2 * chnls),
                self.activation,
            )
            setattr(self, "conv" + str(i), conv)

            deconv = nn.Sequential(
                nn.ConvTranspose2d(
                    4 * chnls, chnls, kernel_size=3, stride=2, bias=False
                ),
                norm_layer(chnls),
                self.activation,
            )
            setattr(self, "deconv" + str(i), deconv)
            chnls *= 2

        self.conv_mid = nn.Sequential(
            nn.Conv2d(chnls, mid_layer_chnls, kernel_size=4, stride=4, bias=False),
            norm_layer(mid_layer_chnls),
            self.activation,
        )

        self.deconv_last = nn.ConvTranspose2d(
            first_layer_chnls, last_layer_chnls, kernel_size=3, stride=2, bias=True
        )

        self.dims = [last_layer_chnls]

    def forward(self, x):
        x = same_pad_conv2d(x, padding_type=self.padding_type, layer=self.conv_in)
        x = self.conv_in(x)

        inters = []
        for i in range(0, self.n_down_layers):
            conv_i = getattr(self, "conv" + str(i))
            x = same_pad_conv2d(x, padding_type=self.padding_type, layer=conv_i)
            x = conv_i(x)
            inters.append(x)

        x = same_pad_conv2d(x, padding_type=self.padding_type, layer=self.conv_mid)
        x = self.conv_mid(x)
        x = x.reshape(x.shape[0], -1, 1, 1).expand(-1, -1, *inters[-1].shape[-2:])

        for i in reversed(range(0, self.n_down_layers)):
            if self.use_skip_conn:
                x = torch.cat((x, inters[i]), dim=1)
            deconv_i = getattr(self, "deconv" + str(i))
            x = deconv_i(x)
            x = same_unpad_deconv2d(x, layer=deconv_i)
        x = self.deconv_last(x)
        x = same_unpad_deconv2d(x, layer=self.deconv_last)
        return x


class SpatialEncoder(nn.Module):
    """
    2D (Spatial/Pixel-aligned/local) image encoder
    """

    def __init__(
        self,
        backbone="resnet34",
        pretrained=True,
        num_layers=4,
        index_interp="bilinear",
        index_padding="border",
        upsample_interp="bilinear",
        feature_scale=2.0,
        use_first_pool=False,
        norm_type="batch",
    ):
        """
        :param backbone Backbone network. Either custom, in which case
        model.custom_encoder.ConvEncoder is used OR resnet18/resnet34, in which case the relevant
        model from torchvision is used
        :param num_layers number of resnet layers to use, 1-5
        :param pretrained Whether to use model weights pretrained on ImageNet
        :param index_interp Interpolation to use for indexing
        :param index_padding Padding mode to use for indexing, border | zeros | reflection
        :param upsample_interp Interpolation to use for upscaling latent code
        :param feature_scale factor to scale all latent by. Useful (<1) if image
        is extremely large, to fit in memory.
        :param use_first_pool if false, skips first maxpool layer to avoid downscaling image
        features too much (ResNet only)
        :param norm_type norm type to applied; pretrained model must use batch
        """
        super().__init__()

        if norm_type != "batch":
            assert not pretrained

        self.use_custom_resnet = backbone == "custom"
        self.feature_scale = feature_scale
        self.use_first_pool = use_first_pool
        norm_layer = get_norm_layer(norm_type)

        if self.use_custom_resnet:
            print("WARNING: Custom encoder is experimental only")
            print("Using simple convolutional encoder")
            self.model = ConvEncoder(3, norm_layer=norm_layer)
            self.latent_size = self.model.dims[-1]
        else:
            # print("Using torchvision", backbone, "encoder")
            self.model = getattr(torchvision.models, backbone)(
                pretrained=pretrained, norm_layer=norm_layer
            )
            # Following 2 lines need to be uncommented for older configs
            self.model.fc = nn.Sequential()
            self.model.avgpool = nn.Sequential()
            self.latent_size = [0, 64, 128, 256, 512, 1024][num_layers]

        self.num_layers = num_layers
        self.index_interp = index_interp
        self.index_padding = index_padding
        self.upsample_interp = upsample_interp
        self.register_buffer("latent", torch.empty(1, 1, 1, 1), persistent=False)
        self.register_buffer(
            "latent_scaling", torch.empty(2, dtype=torch.float32), persistent=False
        )
        # self.latent (B, L, H, W)

    def index(self, uv, cam_z=None, image_size=(), z_bounds=None):
        """
        Get pixel-aligned image features at 2D image coordinates
        :param uv (B, N, 2) image points (x,y)
        :param cam_z ignored (for compatibility)
        :param image_size image size, either (width, height) or single int.
        if not specified, assumes coords are in [-1, 1]
        :param z_bounds ignored (for compatibility)
        :return (B, L, N) L is latent size
        """
        with profiler.record_function("encoder_index"):
            if uv.shape[0] == 1 and self.latent.shape[0] > 1:
                uv = uv.expand(self.latent.shape[0], -1, -1)

            with profiler.record_function("encoder_index_pre"):
                if len(image_size) > 0:
                    if len(image_size) == 1:
                        image_size = (image_size, image_size)
                    scale = self.latent_scaling / image_size
                    uv = uv * scale - 1.0

            uv = uv.unsqueeze(2)  # (B, N, 1, 2)
            samples = F.grid_sample(
                self.latent,
                uv,
                align_corners=True,
                mode=self.index_interp,
                padding_mode=self.index_padding,
            )
            return samples[:, :, :, 0]  # (B, C, N)

    def forward(self, x):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size, H, W)
        """
        if self.feature_scale != 1.0:
            x = F.interpolate(
                x,
                scale_factor=self.feature_scale,
                mode="bilinear" if self.feature_scale > 1.0 else "area",
                align_corners=True if self.feature_scale > 1.0 else None,
                recompute_scale_factor=True,
            )
        x = x.to(device=self.latent.device)

        if self.use_custom_resnet:
            self.latent = self.model(x)
        else:
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)

            latents = [x]
            if self.num_layers > 1:
                if self.use_first_pool:
                    x = self.model.maxpool(x)
                x = self.model.layer1(x)
                latents.append(x)
            if self.num_layers > 2:
                x = self.model.layer2(x)
                latents.append(x)
            if self.num_layers > 3:
                x = self.model.layer3(x)
                latents.append(x)
            if self.num_layers > 4:
                x = self.model.layer4(x)
                latents.append(x)

            self.latents = latents
            align_corners = None if self.index_interp == "nearest" else True
            latent_sz = latents[0].shape[-2:]
            for i in range(len(latents)):
                latents[i] = F.interpolate(
                    latents[i],
                    latent_sz,
                    mode=self.upsample_interp,
                    align_corners=align_corners,
                )
            self.latent = torch.cat(latents, dim=1)
        self.latent_scaling[0] = self.latent.shape[-1]
        self.latent_scaling[1] = self.latent.shape[-2]
        self.latent_scaling = self.latent_scaling / (self.latent_scaling - 1) * 2.0
        return self.latent

    # @classmethod
    # def from_conf(cls, conf):
    #     return cls(
    #         conf.get_string("backbone"),
    #         pretrained=conf.get_bool("pretrained", True),
    #         num_layers=conf.get_int("num_layers", 4),
    #         index_interp=conf.get_string("index_interp", "bilinear"),
    #         index_padding=conf.get_string("index_padding", "border"),
    #         upsample_interp=conf.get_string("upsample_interp", "bilinear"),
    #         feature_scale=conf.get_float("feature_scale", 1.0),
    #         use_first_pool=conf.get_bool("use_first_pool", True),
    #     )


class ImageEncoder(nn.Module):
    """
    Global image encoder
    """

    def __init__(self, backbone="resnet34", pretrained=True, latent_size=128):
        """
        :param backbone Backbone network. Assumes it is resnet*
        e.g. resnet34 | resnet50
        :param num_layers number of resnet layers to use, 1-5
        :param pretrained Whether to use model pretrained on ImageNet
        """
        super().__init__()
        self.model = getattr(torchvision.models, backbone)(pretrained=pretrained)
        self.model.fc = nn.Sequential()
        self.register_buffer("latent", torch.empty(1, 1), persistent=False)
        # self.latent (B, L)
        self.latent_size = latent_size
        if latent_size != 512:
            self.fc = nn.Linear(512, latent_size)

    def index(self, uv, cam_z=None, image_size=(), z_bounds=()):
        """
        Params ignored (compatibility)
        :param uv (B, N, 2) only used for shape
        :return latent vector (B, L, N)
        """
        return self.latent.unsqueeze(-1).expand(-1, -1, uv.shape[1])

    def forward(self, x):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size)
        """
        x = x.to(device=self.latent.device)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)

        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)

        if self.latent_size != 512:
            x = self.fc(x)

        self.latent = x  # (B, latent_size)
        return self.latent

    # @classmethod
    # def from_conf(cls, conf):
    #     return cls(
    #         conf.get_string("backbone"),
    #         pretrained=conf.get_bool("pretrained", True),
    #         latent_size=conf.get_int("latent_size", 128),
    #     )
