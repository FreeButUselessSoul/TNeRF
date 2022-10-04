import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

def standardize(x):
    return (x-torch.mean(x))/torch.std(x)

def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = torch.nonzero(det, as_tuple=False)
    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def mse_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))

    return reduction(image_loss, 2 * M)


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


class MSELoss(nn.Module):
    def __init__(self, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask=None):
        if mask is None:
            mask = torch.ones_like(prediction)
        return mse_loss(prediction, target, mask, reduction=self.__reduction)


class GradientLoss_midas(nn.Module):
    def __init__(self, scales=4, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask=None):
        if mask is None:
            mask = torch.ones_like(prediction)
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                   mask[:, ::step, ::step], reduction=self.__reduction)

        return total


class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction='batch-based'):
        super().__init__()

        self.__data_loss = MSELoss(reduction=reduction)
        self.__regularization_loss = GradientLoss_midas(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(self, prediction, target, mask=None):
        device = prediction.device
        # import ipdb;ipdb.set_trace()
        if mask is None:
            mask = ~torch.isnan(target)
            target[torch.isnan(target)]=0 # remove NaN
        scale, shift = compute_scale_and_shift(prediction, target, mask)
        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)
        total = torch.Tensor([0]).to(device)
        total += self.__data_loss(self.__prediction_ssi, target, mask)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(self.__prediction_ssi, target, mask)

        return total

class ColorLoss(nn.Module):
    def __init__(self, coef=1,bdc=False):
        super().__init__()
        self.coef = coef
        self.bdc = bdc
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        ret = {}
        ret['c_l'] = 0.3*self.loss(inputs['rgb_coarse'], targets)
        if 'rgb_fine' in inputs:
            ret['f_l'] = 3*self.loss(inputs['rgb_fine'], targets)
        if self.bdc and 'bdc' in inputs.keys():
            ret['bdc'] = torch.mean(inputs['bdc'])*0.1
            ret['dS'] = 0.1*torch.mean(torch.exp(-torch.sum(torch.abs(self.gaussian_color(inputs['_rgb_fine_static'])-inputs['_rgb_fine_static']),1))*torch.abs(self.gaussian_depth(inputs['depth_fine_static'])-inputs['depth_fine_static'])[:,0])
        for k, v in ret.items():
            ret[k] = self.coef * v
        return ret

class DepthLoss(nn.Module):
    def __init__(self, coef=1, lambda_u=0.01,kernel_size=5):
        """
        lambda_u: in equation 13
        """
        super().__init__()
        self.coef = coef
        self.lambda_u = lambda_u
        self.gaussian_ = self.get_gaussian_kernel(kernel_size)
        self.grad_loss = GradientLoss()
        self.__data_loss = MSELoss(reduction='batch-based')
        self.__regularization_loss = GradientLoss_midas(scales=3, reduction='batch-based')

    def get_gaussian_kernel(self,kernel_size=3, sigma=2, channels=1):
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
        mean = (kernel_size - 1)/2.
        variance = sigma**2.
        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1./(2.*3.141592654*variance)) *\
                        torch.exp(
                            -torch.sum((xy_grid - mean)**2., dim=-1) /\
                            (2*variance)
                        )
        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                    kernel_size=kernel_size, groups=channels,padding=kernel_size//2, bias=False)

        gaussian_filter.weight.data = gaussian_kernel
        gaussian_filter.weight.requires_grad = False
        
        return gaussian_filter

    def forward(self, inputs, targets, depth_gt, epoch=20):
        ret = {}
        ret['c_l'] = 1 * (torch.abs(inputs['rgb_coarse']-targets)).mean()
        if 'rgb_fine' in inputs:
            ret['f_l'] = 2 * (((inputs['rgb_fine']-targets)**2)).mean()
            ret['b_l'] = 5e-3 * (inputs['beta']**2).mean()
            ret['cost'] = 0.01*torch.mean(inputs['cost'])
            ret['depth_smoothness'] = 0.01*self.grad_loss(inputs['depth_fine_static'].reshape(-1,1,targets.shape[-2],targets.shape[-1]),1)
        return ret

class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp, number=1):
        if number == 1:
            grad_x = torch.abs(inp[..., 2:] + inp[..., :-2] - 2*inp[...,1:-1])
            grad_y = torch.abs(inp[..., 2:, :] + inp[..., :-2, :] - 2*inp[...,1:-1,:])
            grad_mag = torch.sum(grad_x) + torch.sum(grad_y)
        elif number == 2:
            grad_x = torch.sum((inp[..., 2:] + inp[..., :-2] - 2*inp[...,1:-1])**2)
            grad_y = torch.sum((inp[..., 2:] + inp[..., :-2] - 2*inp[...,1:-1])**2)
            grad_mag = (grad_x + grad_y + 1e-10)**0.5
        return grad_mag
        

class MutexLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, im1,im2, x_kernel):
        img1 = torch.mean(im1,1,keepdim=True)
        img2 = torch.mean(im2,1,keepdim=True)
        y_kernel = x_kernel.transpose(-1,-2)
        G_x1 = F.conv2d(img1,x_kernel,padding=0)
        G_y1 = F.conv2d(img1,y_kernel,padding=0)
        G_x2 = F.conv2d(img2,x_kernel,padding=0)
        G_y2 = F.conv2d(img2,y_kernel,padding=0)
        n1 = (G_x1 ** 2 + G_y1 ** 2)
        n2 = (G_x2 ** 2 + G_y2 ** 2)
        G1=torch.sqrt(n1+1e-10)
        G2=torch.sqrt(n2+1e-10)
        n1=torch.sqrt(torch.sum(n1)+1e-10)
        n2=torch.sqrt(torch.sum(n2)+1e-10)
        norm_factor = torch.sqrt((n1+1e-10)/(n2+1e-10))
        grad_mag1 = torch.tanh(G1/norm_factor)
        grad_mag2 = torch.tanh(G2*norm_factor)
        return torch.sqrt(torch.sum((grad_mag1*grad_mag2)**2+1e-10)) # Frob. norm

class NerfTLoss(nn.Module):
    def __init__(self, coef=1, lambda_u=0.01):
        """
        lambda_u: in equation 13
        """
        super().__init__()
        self.coef = coef
        self.lambda_u = lambda_u
        # self.__data_loss = MSELoss(reduction='batch-based')
        # self.__regularization_loss = GradientLoss_midas(scales=3, reduction='batch-based')
        self.depth_smooth_loss = GradientLoss()

    def forward(self, inputs, targets, BG_edge=None):
        ret = {}
        ret['c_l'] = 0.5 * (torch.abs(inputs['rgb_coarse']-targets)).mean()
        if 'rgb_fine' in inputs:
            if BG_edge is not None:
                # ret['f_l'] = 2*(torch.abs(inputs['rgb_fine']-targets)).mean()#\
                ret['f_l'] = 2*((inputs['rgb_fine']-targets)**2 * BG_edge).mean()\
                    +2*(torch.abs(inputs['rgb_fine']-targets)).mean()#\
                    # +0.1*(torch.abs(inputs['rgb_fine']-targets)/(1e-2+2*inputs['beta']**2)).mean()
            else:
                ret['f_l'] = 4*((inputs['rgb_fine']-targets)**2).mean()

        for k, v in ret.items():
            ret[k] = self.coef * v

        return ret

class NerfWLoss(nn.Module):
    """
    Equation 13 in the NeRF-W paper.
    Name abbreviations:
        c_l: coarse color loss
        f_l: fine color loss (1st term in equation 13)
        b_l: beta loss (2nd term in equation 13)
        s_l: sigma loss (3rd term in equation 13)
    """
    def __init__(self, coef=1, lambda_u=0.01, bdc=False):
        """
        lambda_u: in equation 13
        """
        super().__init__()
        self.coef = coef
        self.lambda_u = lambda_u
        self.bdc = bdc
        self.gaussian_color=self.get_gaussian_kernel(3,2,3)
        self.gaussian_depth=self.get_gaussian_kernel(3,2,1)

    def get_gaussian_kernel(self,kernel_size=3, sigma=2, channels=1):
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
        mean = (kernel_size - 1)/2.
        variance = sigma**2.
        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1./(2.*3.141592654*variance)) *\
                        torch.exp(
                            -torch.sum((xy_grid - mean)**2., dim=-1) /\
                            (2*variance)
                        )
        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                    kernel_size=kernel_size, groups=channels,padding=kernel_size//2, bias=False)

        gaussian_filter.weight.data = gaussian_kernel
        gaussian_filter.weight.requires_grad = False
        
        return gaussian_filter

    
    def forward(self, inputs, targets):
        ret = {}
        ret['c_l'] = 0.5 * ((inputs['rgb_coarse']-targets)**2).mean()
        if 'rgb_fine' in inputs:
            if 'beta' not in inputs: # no transient head, normal MSE loss
                ret['f_l'] = 0.5 * ((inputs['rgb_fine']-targets)**2).mean()
            else:
                ret['f_l'] = \
                    ((inputs['rgb_fine']-targets)**2/(2*inputs['beta'].unsqueeze(1)**2)).mean()
                ret['b_l'] = 3 + torch.log(inputs['beta']).mean() # +3 to make it positive
                ret['s_l'] = self.lambda_u * inputs['transient_sigmas'].mean()
        if self.bdc and 'bdc' in inputs.keys():
            ret['bdc'] = torch.mean(inputs['bdc'])*0.5
            ret['dS'] = 0.1*torch.mean(torch.exp(-torch.sum(torch.abs(self.gaussian_color(inputs['_rgb_fine_static'])-inputs['_rgb_fine_static']),1))*torch.abs(self.gaussian_depth(inputs['depth_fine_static'])-inputs['depth_fine_static'])[:,0])


        for k, v in ret.items():
            ret[k] = self.coef * v

        return ret

loss_dict = {'color': ColorLoss,
             'nerfw': NerfWLoss,
             'depth':DepthLoss,
             'gradient':GradientLoss,
             'mutex':MutexLoss,
             'nerft':NerfTLoss}