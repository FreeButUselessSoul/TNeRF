import torch
from einops import rearrange, reduce, repeat
import torch.nn.functional as F

__all__ = ['render_rays']

from kornia.utils import create_meshgrid

def repeat_interleave(input, repeats, dim=0):
    output = input.unsqueeze(1).expand(-1, repeats, *input.shape[1:])
    return output.reshape(-1, *input.shape[1:])

@torch.no_grad()
def homo_warp_with_depth(src_feat, proj_mat, depth_values, src_grid=None, ref_g=None, pad=0):
    """
    src_feat: (B, C, H, W)
    proj_mat: (B, 3, 4) equal to "src_proj @ ref_proj_inv"
    depth_values: (B,hw, D)
    out: (B, C, D, H, W)
    """
    if len(src_feat.shape)==5:
        # (B,n,C,H,W)
        depth_values = repeat_interleave(depth_values,src_feat.shape[1],0)
        if (ref_g is not None):
            ref_g = ref_g.repeat(src_feat.shape[1],1,1,1)
        src_feat = src_feat.flatten(0,1)
        proj_mat = proj_mat.flatten(0,1)

    assert(src_grid==None)
    B, C, H, W = src_feat.shape
    device = src_feat.device

    if pad>0:
        H_pad, W_pad = H + pad*2, W + pad*2
    else:
        H_pad, W_pad = H, W

    D = depth_values.shape[-1]

    R = proj_mat[:, :, :3]  # (B, 3, 3)
    T = proj_mat[:, :, 3:]  # (B, 3, 1)

    # create grid from the ref frame
    if ref_g is None:
        ref_grid = create_meshgrid(H_pad, W_pad, normalized_coordinates=False, device=device)  # (1, H, W, 2)
        if pad>0:
            ref_grid -= pad
    else:
        ref_grid = ref_g
        H_pad,W_pad = ref_g.shape[1:3]
        # if (depth_values.shape!=(B,D,H_pad,W_pad)):
        #     depth_values = F.grid_sample(depth_values, ref_g, mode='bilinear',align_corners=True)
        
    ref_grid = ref_grid.permute(0, 3, 1, 2)  # (B, 2, H, W)
    ref_grid = ref_grid.reshape(B, 2, W_pad * H_pad)  # (B, 2, H*W)
    ref_grid = torch.cat((ref_grid, torch.ones_like(ref_grid[:, :1])), 1)  # (B, 3, H*W)
    ref_grid_d = ref_grid.repeat(1, 1, D)  # (B, 3, D*H*W)
    src_grid_d = R @ ref_grid_d + T / depth_values.view(B, 1, D * W_pad * H_pad)
    del ref_grid_d, ref_grid, proj_mat, R, T, depth_values  # release (GPU) memory

    src_grid = src_grid_d[:, :2] / src_grid_d[:, 2:]  # divide by depth (B, 2, D*H*W)
    del src_grid_d
    src_grid[:, 0] = src_grid[:, 0] / ((W - 1) / 2) - 1  # scale to -1~1
    src_grid[:, 1] = src_grid[:, 1] / ((H - 1) / 2) - 1  # scale to -1~1
    src_grid = src_grid.permute(0, 2, 1)  # (B, D*H*W, 2)
    src_grid = src_grid.view(B, D, W_pad * H_pad, 2)
    warped_src_feat = F.grid_sample(src_feat, src_grid,
                                    mode='linear', padding_mode='zeros',
                                    align_corners=True)  # (B, C, D, H*W)
    warped_src_feat = warped_src_feat.view(B, -1, D, H_pad, W_pad)
    # src_grid = src_grid.view(B, 1, D, H_pad, W_pad, 2)
    return warped_src_feat, src_grid

def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero
    Outputs:
        samples: the sampled samples
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps # prevent division by zero (don't do inplace op!)
    pdf = weights / reduce(weights, 'n1 n2 -> n1 1', 'sum') # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1) 
                                                               # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = rearrange(torch.stack([below, above], -1), 'n1 n2 c -> n1 (n2 c)', c=2)
    cdf_g = rearrange(torch.gather(cdf, 1, inds_sampled), 'n1 (n2 c) -> n1 n2 c', c=2)
    bins_g = rearrange(torch.gather(bins, 1, inds_sampled), 'n1 (n2 c) -> n1 n2 c', c=2)

    denom = cdf_g[...,1]-cdf_g[...,0]
    denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                         # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
    return samples


def render_rays(models,
                embeddings,
                rays,
                ts,
                N_samples=64,
                use_disp=False,
                perturb=0,
                noise_std=1,
                N_importance=0,
                chunk=1024*32,
                white_back=False,
                test_time=False,
                proj_mat=torch.zeros((1,3,4)),
                full_code_B=torch.Tensor(),pixels_i=None,
                **kwargs
                ):
    """
    Render rays by computing the output of @model applied on @rays and @ts
    Inputs:
        models: dict of NeRF models (coarse and fine) defined in nerf.py
        embeddings: dict of embedding models of origin and direction defined in nerf.py
        rays: (N_rays, 3+3), ray origins and directions
        ts: (N_rays), ray time as embedding index
        N_samples: number of coarse samples per ray
        use_disp: whether to sample in disparity space (inverse depth)
        perturb: factor to perturb the sampling position on the ray (for coarse model only)
        noise_std: factor to perturb the model's prediction of sigma
        N_importance: number of fine samples per ray
        chunk: the chunk size in batched inference
        white_back: whether the background is white (dataset dependent)
        test_time: whether it is test (inference only) or not. If True, it will not do inference
                   on coarse rgb to save time
        features: The input features along with xyz coord.
    Outputs:
        result: dictionary containing final rgb and depth maps for coarse and fine models
    """

    def inference(results, model, xyz, z_vals, test_time=False, **kwargs):
        """
        Helper function that performs model inference.
        Inputs:
            results: a dict storing all results
            model: NeRF model (coarse or fine)
            xyz: (N_rays, N_samples_, 3) sampled positions
                  N_samples_ is the number of sampled points on each ray;
                             = N_samples for coarse model
                             = N_samples+N_importance for fine model
            z_vals: (N_rays, N_samples_) depths of the sampled positions
            test_time: test time or not
        """
        typ = model.typ
        N_samples_ = xyz.shape[1]
        xyz_ = rearrange(xyz, 'n1 n2 c -> (n1 n2) c', c=3)
        # Perform model inference to get rgb and raw sigma
        B = xyz_.shape[0]
        out_chunks = []
        warpped_code_B_ = torch.mean(warpped_code_B,0)
        if typ=='coarse' and test_time:
            if len(full_code_B):
                for i in range(0, B, chunk):
                    xyz_embedded = torch.cat((embedding_xyz(xyz_[i:i+chunk]),warpped_code_B_[i:i+chunk]),-1)
                    out_chunks += [model(xyz_embedded, sigma_only=True)]
            else:
                for i in range(0, B, chunk):
                    xyz_embedded = embedding_xyz(xyz_[i:i+chunk])
                    out_chunks += [model(xyz_embedded, sigma_only=True)]
            out = torch.cat(out_chunks, 0)
            static_sigmas = rearrange(out, '(n1 n2) 1 -> n1 n2', n1=N_rays, n2=N_samples_)
        else: # infer rgb and sigma and others
            dir_embedded_ = repeat(dir_embedded, 'n1 c -> (n1 n2) c', n2=N_samples_)
            if output_transient:
                t_embedded_ = repeat(t_embedded, 'n1 c -> (n1 n2) c', n2=N_samples_)
            for i in range(0, B, chunk):
                # inputs for original NeRF
                inputs = [torch.cat((embedding_xyz(xyz_[i:i+chunk]),warpped_code_B_[i:i+chunk]),-1), dir_embedded_[i:i+chunk]]
                if output_transient:
                    inputs += [t_embedded_[i:i+chunk]]
                out_chunks += [model(torch.cat(inputs, 1), output_transient=output_transient)]

            out = torch.cat(out_chunks, 0)
            out = rearrange(out, '(n1 n2) c -> n1 n2 c', n1=N_rays, n2=N_samples_)
            static_rgbs = out[..., :3] # (N_rays, N_samples_, 3)
            static_sigmas = out[..., 3] # (N_rays, N_samples_)
            if output_transient:
                transient_rgbs = out[..., 4:7]
                transient_sigmas = out[..., 7]
                transient_betas = out[..., 8]

        # Convert these values using volume rendering
        deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples_-1)
        delta_inf = 1e2 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
        deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)
        if output_transient:
            static_alphas = 1-torch.exp(-deltas*static_sigmas)
            transient_alphas = 1-torch.exp(-deltas*transient_sigmas)
            alphas = 1-torch.exp(-deltas*(static_sigmas+transient_sigmas))
            # import ipdb;ipdb.set_trace()
            # results['Q_s'] = torch.sum(static_alphas,-1)
            # results['Q_t'] = torch.sum(transient_alphas,-1)
            # results['p_s'] = static_alphas / (results['Q_s'].unsqueeze(-1)+1e-7)
            # results['p_t'] = transient_alphas / (results['Q_t'].unsqueeze(-1)+1e-7)
            # results['Q_0'] = torch.sum(static_alphas[:,:-1],-1)
            # results['p_0'] = static_alphas[:,:-1] / (results['Q_0'].unsqueeze(-1)+1e-7)
        else:
            noise = torch.randn_like(static_sigmas) * noise_std
            alphas = 1-torch.exp(-deltas*torch.relu(static_sigmas+noise))

        alphas_shifted = \
            torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas], -1) # [1, 1-a1, 1-a2, ...]
        transmittance = torch.cumprod(alphas_shifted[:, :-1], -1) # [1, 1-a1, (1-a1)(1-a2), ...]

        if output_transient:
            # if 'r' in embeddings:
            static_alphas_shifted = \
                torch.cat([torch.ones_like(static_alphas[:, :1]), 1-static_alphas], -1) # [1, 1-a1, 1-a2, ...]
            static_transmittance = torch.cumprod(static_alphas_shifted[:, :-1], -1) # [1, 1-a1, (1-a1)(1-a2), ...]
            transient_alphas_shifted = \
                torch.cat([torch.ones_like(transient_alphas[:, :1]), 1-transient_alphas], -1) # [1, 1-a1, 1-a2, ...]
            transient_transmittance = torch.cumprod(transient_alphas_shifted[:, :-1], -1) # [1, 1-a1, (1-a1)(1-a2), ...]
            # results['static_transmittance'] = static_transmittance
            # results['transient_transmittance'] = transient_transmittance
            # static_weights = static_alphas * transmittance
            # transient_weights = transient_alphas * transmittance
            static_weights = static_alphas * static_transmittance
            transient_weights = transient_alphas * transient_transmittance
            # else:
            #     static_weights = static_alphas * transmittance
            #     transient_weights = transient_alphas * transmittance

        weights = alphas * transmittance
        weights_sum = reduce(weights, 'n1 n2 -> n1', 'sum')

        results[f'weights_{typ}'] = weights
        results[f'opacity_{typ}'] = weights_sum
        results['static_sigmas'] = static_sigmas / torch.max(static_sigmas)
        if output_transient:
            results['transient_sigmas'] = transient_sigmas/ torch.max(transient_sigmas)
        if test_time and typ == 'coarse':
            return


        if output_transient:
            static_rgb_map = reduce(rearrange(static_weights, 'n1 n2 -> n1 n2 1')*static_rgbs,
                                    'n1 n2 c -> n1 c', 'sum')
            if white_back:
                static_rgb_map += 1-rearrange(weights_sum, 'n -> n 1')
            
            transient_rgb_map = \
                reduce(rearrange(transient_weights, 'n1 n2 -> n1 n2 1')*transient_rgbs,
                       'n1 n2 c -> n1 c', 'sum')
            # Add beta_min AFTER the beta composition. Different from eq 10~12 in the paper.
            # See "Notes on differences with the paper" in README.
            results['beta'] = reduce(transient_weights*transient_betas, 'n1 n2 -> n1', 'sum').unsqueeze(-1)
            results['beta'] *= 0.5
            results['beta'] += model.beta_min
            results['beta'] = torch.clamp(results['beta'],0,0.5)
            # the rgb maps here are when both fields exist
            results['_rgb_fine_static'] = static_rgb_map
            results['_rgb_fine_transient'] = transient_rgb_map
            # print(results['beta'].shape,static_rgb_map.shape)
            # results['rgb_fine'] = static_rgb_map + transient_rgb_map * results['beta']
            results['rgb_fine'] = static_rgb_map * (1-results['beta']) + transient_rgb_map * results['beta']
            if not test_time:
                results['cost'] = torch.mean( torch.var(warpped_code_B,-1) *static_weights.flatten(0,1),0,True)
            results['depth_fine_static'] = reduce(static_weights*z_vals, 'n1 n2 -> n1', 'sum')
            results['depth_fine_transient'] = reduce(transient_weights*z_vals, 'n1 n2 -> n1', 'sum')
            

            if test_time:
                # Compute also static and transient rgbs when only one field exists.
                # The result is different from when both fields exist, since the transimttance
                # will change.
                static_alphas_shifted = \
                    torch.cat([torch.ones_like(static_alphas[:, :1]), 1-static_alphas], -1) # [1, 1-a1, 1-a2, ...]
                static_transmittance = torch.cumprod(static_alphas_shifted[:, :-1], -1) # [1, 1-a1, (1-a1)(1-a2), ...]
                static_weights_ = static_alphas * static_transmittance
                static_rgb_map_ = \
                    reduce(rearrange(static_weights_, 'n1 n2 -> n1 n2 1')*static_rgbs,
                           'n1 n2 c -> n1 c', 'sum')
                if white_back:
                    static_rgb_map_ += 1-rearrange(weights_sum, 'n -> n 1')
                results['rgb_fine_static'] = static_rgb_map_
                results['depth_fine_static'] = \
                    reduce(static_weights_*z_vals, 'n1 n2 -> n1', 'sum')

                transient_alphas_shifted = \
                    torch.cat([torch.ones_like(transient_alphas[:, :1]), 1-transient_alphas], -1)
                transient_transmittance = torch.cumprod(transient_alphas_shifted[:, :-1], -1)
                transient_weights_ = transient_alphas * transient_transmittance
                results['rgb_fine_transient'] = \
                    reduce(rearrange(transient_weights_, 'n1 n2 -> n1 n2 1')*transient_rgbs,
                           'n1 n2 c -> n1 c', 'sum')
                results['depth_fine_transient'] = \
                    reduce(transient_weights_*z_vals, 'n1 n2 -> n1', 'sum')
        else: # no transient field
            rgb_map = reduce(rearrange(weights, 'n1 n2 -> n1 n2 1')*static_rgbs,
                             'n1 n2 c -> n1 c', 'sum')
            if white_back:
                rgb_map += 1-rearrange(weights_sum, 'n -> n 1')
            results[f'rgb_{typ}'] = rgb_map

        results[f'depth_{typ}'] = reduce(weights*z_vals, 'n1 n2 -> n1', 'sum')
        return


    if not(proj_mat.shape[-2]==3 and proj_mat.shape[-1]==4):
        import ipdb;ipdb.set_trace()
    
    embedding_xyz, embedding_dir = embeddings['xyz'], embeddings['dir']

    # Decompose the inputs
    if (len(rays.shape)==2):
        N_rays = rays.shape[0]
    else:
        N_rays = rays.shape[0]*rays.shape[1]
    rays_o, rays_d = rays[..., 0:3], rays[..., 3:6] # both (N_rays, 3)
    near, far = rays[..., 6:7], rays[..., 7:8] # both (N_rays, 1)
    # Embed direction
    dir_embedded = embedding_dir(kwargs.get('view_dir', rays_d))
    if (len(rays.shape)!=2):
        dir_embedded = dir_embedded.flatten(0,1)
        near = near.flatten(0,1)
        far = far.flatten(0,1)
        ts = ts.flatten(0,1)


    # Sample depth points
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device)
    if not use_disp: # use linear sampling in depth space
        z_vals = near * (1.-z_steps) + far * z_steps
    else: # use linear sampling in disparity space
        z_vals = 1./(1./near * (1.-z_steps) + 1./far * z_steps)
    # z_vals_plain = z_vals
    if (len(rays.shape)==2):
        rays_o = rearrange(rays_o, 'n1 c -> n1 1 c')
        rays_d = rearrange(rays_d, 'n1 c -> n1 1 c')
        z_vals = z_vals.expand(N_rays, N_samples)
    else:
        rays_o = rays_o.flatten(0,1)[:,None,:]
        rays_d = rays_d.flatten(0,1)[:,None,:]
    
    if perturb > 0: # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[... ,:-1] + z_vals[... ,1:]) # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[... ,-1:]], -1)
        lower = torch.cat([z_vals[... ,:1], z_vals_mid], -1)
        
        perturb_rand = perturb * torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * perturb_rand

    # if (len(rays.shape)==2):
    xyz_coarse = rays_o + rays_d * rearrange(z_vals, 'n1 n2 -> n1 n2 1')
    # else:
    #     xyz_coarse = rays_o[:,:,None,:] + rays_d[:,:,None,:] * z_vals[...,None]
    #     xyz_coarse = xyz_coarse.flatten(0,1)


    # warp features w.r.t. depth
    # z_vals_ = torch.clamp(z_vals,1e-7,1-1e-7)
    warpped_code_B, _ = homo_warp_with_depth(full_code_B,proj_mat,z_vals,ref_g=pixels_i) # buggy
    # if len(rays.shape)!=2:
    if pixels_i is not None:
        # C, D = warpped_code_B.shape[1:3]
        warpped_code_B = warpped_code_B.squeeze().flatten(-2,-1).transpose(-1,-2)
    else:
        # warpped_code_B = warpped_code_B[0].flatten(-2, -1).T # (h*w, D, C)
        raise NotImplementedError
    # warpped_code_B = warpped_code_B[st:ed].flatten(0,1)

    results = {}
    output_transient = False
    
    if (len(rays.shape)!=2):
        # z_vals = z_vals.flatten(0,1)
        warpped_code_B = warpped_code_B.reshape(2,-1,warpped_code_B.shape[-1])
        # import ipdb;ipdb.set_trace()

    inference(results, models['coarse'], xyz_coarse, z_vals, test_time, **kwargs)

    if N_importance > 0: # sample points for fine model
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        # import ipdb;ipdb.set_trace()
        z_vals_ = sample_pdf(z_vals_mid, results['weights_coarse'][:, 1:-1].detach(),
                             N_importance, det=(perturb==0))
                  # detach so that grad doesn't propogate to weights_coarse from here
        z_vals = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)[0]
        # if (len(rays.shape)==2):
        xyz_fine = rays_o + rays_d * rearrange(z_vals, 'n1 n2 -> n1 n2 1')
        # else:
        #     xyz_fine = rays_o[:,:,None,:] + rays_d[:,:,None,:] * z_vals[...,None]
        #     xyz_fine = xyz_fine.flatten(0,1)

        model = models['fine']
        output_transient = kwargs.get('output_transient', True)
        if output_transient:
            if 't_embedded' in kwargs:
                t_embedded = kwargs['t_embedded']
            else:
                t_embedded = embeddings['t'](ts)
        # z_vals_local = torch.mean(z_vals,axis=0,keepdims=True)
        warpped_code_B, _ = homo_warp_with_depth(full_code_B,proj_mat,z_vals,ref_g=pixels_i)
        # how to incorporate with importance sampling?
        if pixels_i is not None:
            warpped_code_B = warpped_code_B.squeeze().flatten(-2,-1).transpose(-1,-2)
        else:
            raise NotImplementedError
        if len(rays.shape)!=2:
            warpped_code_B = warpped_code_B.reshape(2,-1,warpped_code_B.shape[-1])
            # warpped_code_B = warpped_code_B[0].flatten(-2, -1).T # (h*w, D, C)
        # warpped_code_B = warpped_code_B[st:ed].flatten(0,1)
        inference(results, model, xyz_fine, z_vals, test_time, **kwargs)

    return results