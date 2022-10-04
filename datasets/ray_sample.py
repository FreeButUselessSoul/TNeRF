import torch
from math import exp,ceil

class ImgToPatch(object):
    def __init__(self, ray_sampler, hw):
        self.ray_sampler = ray_sampler
        self.len = self.ray_sampler.len
        self.hw = hw      # hw of samples

    def __call__(self, inputs:list, idx=-1,output_grid=False): # img:(3,h,w)
        _, pixels_i = self.ray_sampler(idx=idx)
        pixels_i = pixels_i.to(inputs[0].device)
        outputs = []
        for item in inputs:
            item = torch.nn.functional.grid_sample(item.unsqueeze(0), pixels_i.unsqueeze(0), mode='bilinear', align_corners=True)[0]
            item = item.flatten(1, 2).t()
            outputs.append(item.detach())
        if len(inputs)>1:
            if (output_grid):
                return outputs, pixels_i
            return outputs
        if output_grid:
            return outputs[0], pixels_i
        return outputs[0]

class RaySampler(object):
    def __init__(self, hw, orthographic=False, img_wh=(504,378)):
        super(RaySampler, self).__init__()
        self.N_samples = hw ** 2
        self.scale = torch.ones(1,).float()
        self.orthographic = orthographic
        self.W, self.H = img_wh
        self.nW = int(1.2*((self.W // hw) )+1)
        self.nH = int(1.2*((self.H // hw) )+1) # Some overlap
        self.len = self.nW * self.nH
        self.stepW = ((self.W-hw) / (self.nW-1))
        self.stepH = ((self.H-hw) / (self.nH-1))

    def __call__(self, idx=-1):

        return_indices,select_inds = self.sample_rays(idx)

        if return_indices:
            # return select_inds, select_inds
            h,w=select_inds
            h = 2*h / float(self.H) - 1.0
            w = 2*w / float(self.W) - 1.0

            hw = torch.cat([h,w],-1)

        else:
            hw = select_inds
            select_inds = None

        return select_inds, hw

    def sample_rays(self):
        raise NotImplementedError

class FlexGridRaySampler(RaySampler):
    def __init__(self, hw, random_shift=True, random_scale=True, min_scale=0.25,
                 max_scale=1., scale_anneal=-1,img_wh=(504,378),
                 **kwargs):
        super(FlexGridRaySampler, self).__init__(hw,img_wh=img_wh, **kwargs)

        self.random_shift = random_shift
        self.random_scale = random_scale
        self.hw = hw
        self.min_scale = min_scale
        self.max_scale = max_scale

        # nn.functional.grid_sample grid value range in [-1,1]
        self.w, self.h = torch.meshgrid([torch.linspace(-1,1,hw),
                                         torch.linspace(-1,1,hw)])
        self.h = self.h.unsqueeze(2)
        self.w = self.w.unsqueeze(2)

        # directly return grid for grid_sample
        self.return_indices = False

        self.iterations = 0
        self.scale_anneal = scale_anneal

    def sample_rays(self, idx=-1):

        if self.scale_anneal>0:
            self.iterations += 1
            k_iter = self.iterations // 1000 * 3
            min_scale = max(self.min_scale, self.max_scale * exp(-k_iter*self.scale_anneal))
            min_scale = min(0.9, min_scale)
        else:
            min_scale = self.min_scale

        if idx<0 or idx>=self.len:
            if self.random_scale:
                scale = torch.Tensor(1).uniform_(min_scale, self.max_scale)
                h = self.h * scale 
                w = self.w * scale 

            if self.random_shift:
                max_offset = 1-scale.item()
                h_offset = torch.Tensor(1).uniform_(0, max_offset) * (torch.randint(2,(1,)).float()-0.5)*2
                w_offset = torch.Tensor(1).uniform_(0, max_offset) * (torch.randint(2,(1,)).float()-0.5)*2

                h += h_offset
                w += w_offset
            self.scale = scale

            return False, torch.cat([h, w], dim=2)
        else:
            left = int((idx % self.nW) * self.stepW)
            top = int((idx // self.nW) * self.stepH)
            w, h = torch.meshgrid([torch.arange(left,left+self.hw).float(),
                                         torch.arange(top,top+self.hw).float()])
            h = h.unsqueeze(2)
            w = w.unsqueeze(2)
            return True,(h, w)
