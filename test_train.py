import torch
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import imageio
from argparse import ArgumentParser

# from models.rendering import render_rays
# from models.nerf import *
from models.rendering_warp import render_rays
from models.nerf_warp import *

from utils import load_ckpt
import metrics

from datasets import dataset_dict
from datasets.depth_utils import *

torch.backends.cudnn.benchmark = True

def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        default='/home/ubuntu/data/nerf_example_data/nerf_synthetic/lego',
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='blender',
                        help='which dataset to validate')
    parser.add_argument('--scene_name', type=str, default='test',
                        help='scene name, used as output folder name')
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[800, 800],
                        help='resolution (img_w, img_h) of the image')
    # for phototourism
    parser.add_argument('--img_downscale', type=int, default=1,
                        help='how much to downscale the images for phototourism dataset')
    parser.add_argument('--use_cache', default=False, action="store_true",
                        help='whether to use ray cache (make sure img_downscale is the same)')

    # original NeRF parameters
    parser.add_argument('--N_emb_xyz', type=int, default=10,
                        help='number of xyz embedding frequencies')
    parser.add_argument('--N_emb_dir', type=int, default=4,
                        help='number of direction embedding frequencies')
    parser.add_argument('--N_samples', type=int, default=64,
                        help='number of coarse samples')
    parser.add_argument('--N_importance', type=int, default=128,
                        help='number of additional fine samples')
    parser.add_argument('--latent_dim', type=int, default=512,
                        help='number of latent dimension')
    parser.add_argument('--use_disp', default=False, action="store_true",
                        help='use disparity depth sampling')

    # NeRF-W parameters
    parser.add_argument('--N_vocab', type=int, default=100,
                        help='''number of vocabulary (number of images) 
                                in the dataset for nn.Embedding''')
    parser.add_argument('--encode_a', default=False, action="store_true",
                        help='whether to encode appearance (NeRF-A)')
    parser.add_argument('--N_a', type=int, default=48,
                        help='number of embeddings for appearance')
    parser.add_argument('--gpu', type=int, default=0,
                        help='device ID')
    parser.add_argument('--encode_t', default=False, action="store_true",
                        help='whether to encode transient object (NeRF-U)')
    parser.add_argument('--N_tau', type=int, default=16,
                        help='number of embeddings for transient objects')
    parser.add_argument('--beta_min', type=float, default=0.1,
                        help='minimum color variance for each ray')

    parser.add_argument('--chunk', type=int, default=32*1024*4,
                        help='chunk size to split the input to avoid OOM')

    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='pretrained checkpoint path to load')

    parser.add_argument('--video_format', type=str, default='gif',
                        choices=['gif', 'mp4'],
                        help='video format, gif or mp4')

    return parser.parse_args()


@torch.no_grad()
def batched_inference(models, embeddings,
                      rays, ts, N_samples, N_importance, use_disp,
                      chunk,
                      white_back,full_code_B,proj_mat,pixels_i,
                      **kwargs):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    results = defaultdict(list)
    for i in range(0, B, chunk):
        rendered_ray_chunks = \
            render_rays(models,
                        embeddings,
                        rays[i:i+chunk],
                        ts[i:i+chunk] if ts is not None else None,
                        N_samples,
                        use_disp,
                        0,
                        0,
                        N_importance,
                        chunk,
                        white_back,
                        test_time=True,
                        full_code_B=full_code_B,
                        proj_mat=proj_mat,
                        pixels_i=pixels_i[:,i:i+chunk],
                        **kwargs)

        for k, v in rendered_ray_chunks.items():
            results[k] += [v.cpu()]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results


if __name__ == "__main__":
    args = get_opts()
    scene = os.path.basename(args.root_dir.strip('/'))
    kwargs = {'root_dir': args.root_dir,
              'split': args.split}
    # kwargs['test_appearance_idx'] = 0
    kwargs['test_img_w'], kwargs['test_img_h'] = args.img_wh
    kwargs['img_wh'] = tuple(args.img_wh)
    dataset = dataset_dict[args.dataset_name](**kwargs)

    embedding_xyz = PosEmbedding(args.N_emb_xyz-1, args.N_emb_xyz)
    embedding_dir = PosEmbedding(args.N_emb_dir-1, args.N_emb_dir)
    embeddings = {'xyz': embedding_xyz, 'dir': embedding_dir}
    if args.encode_a:
        embedding_a = torch.nn.Embedding(args.N_vocab, args.N_a).cuda(args.gpu)
        load_ckpt(embedding_a, args.ckpt_path, model_name='embedding_a')
        embeddings['a'] = embedding_a
    if args.encode_t:
        embedding_t = torch.nn.Embedding(args.N_vocab, args.N_tau).cuda(args.gpu)
        load_ckpt(embedding_t, args.ckpt_path, model_name='embedding_t')
        embeddings['t'] = embedding_t

    nerf_coarse = NeRFWarp('coarse',
                        in_channels_xyz=6*args.N_emb_xyz+3,
                        in_channels_dir=6*args.N_emb_dir+3,
                        in_channels_f=args.latent_dim).cuda(args.gpu)
    models = {'coarse': nerf_coarse}
    nerf_fine = NeRFWarp('fine',
                     in_channels_xyz=6*args.N_emb_xyz+3,
                     in_channels_dir=6*args.N_emb_dir+3,
                     encode_transient=args.encode_t,
                     in_channels_t=args.N_tau,
                     beta_min=args.beta_min,
                     in_channels_f=args.latent_dim).cuda(args.gpu)

    load_ckpt(nerf_coarse, args.ckpt_path, model_name='nerf_coarse')
    load_ckpt(nerf_fine, args.ckpt_path, model_name='nerf_fine')

    models = {'coarse': nerf_coarse, 'fine': nerf_fine}

    imgs, psnrs = [], []
    ssims, piqs = [], []
    img_reflection = []
    dir_name = f'results/{args.dataset_name}/{args.scene_name}'
    os.makedirs(dir_name, exist_ok=True)

    kwargs = {}

    kwargs['output_transient'] = False # Don't need transient layer!
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        rays = sample['rays']
        ts = sample['ts']
        full_code_B = sample['full_code_B'].unsqueeze(0)
        proj_mat = sample['proj_mat'].unsqueeze(0)
        index_w, index_h = torch.meshgrid([torch.linspace(-1,1,args.img_wh[0]),
                                         torch.linspace(-1,1,args.img_wh[1])])
        pixels_i = torch.dstack([index_w,index_h]).flatten(0,1)[None,:,None,:].cuda(args.gpu)
        results = batched_inference(models, embeddings, rays.cuda(args.gpu), ts.cuda(args.gpu),
                                    args.N_samples, args.N_importance, args.use_disp,
                                    args.chunk,
                                    dataset.white_back, full_code_B.cuda(args.gpu), proj_mat=proj_mat.cuda(args.gpu),
                                    pixels_i=pixels_i,
                                    **kwargs)

        w, h = args.img_wh
        # temp = results['rgb_fine'].view(h, w, 3).cpu().numpy()
        # img_pred = (temp-np.min(temp)) / (np.max(temp)-np.min(temp))
        img_pred = np.clip(results['rgb_fine'].view(h, w, 3).cpu().numpy(), 0, 1)
        # beta = results['beta'].view(h, w, 1).cpu().numpy()
        # img_trans = results['_rgb_fine_transient'].view(h, w, 3).cpu().numpy()
        
        img_pred_ = (img_pred*255).astype(np.uint8)
        # img_trans_ = (img_trans*255).astype(np.uint8)
        # beta_ = (beta*255).astype(np.uint8)
        imgs += [img_pred_]
        imageio.imwrite(os.path.join(dir_name, f'{args.scene_name}-B-{i}.png'), img_pred_)
        # imageio.imwrite(os.path.join(dir_name, f'{args.scene_name}-R-{i}.png'), img_trans_)
        # imageio.imwrite(os.path.join(dir_name, f'{args.scene_name}-beta-{i}.png'), beta)
        if 'original' in sample:
            img_gt = sample['original'].view(h, w, 3)
            psnrs += [metrics.psnr(img_gt, img_pred).item()]
            piqs += [piq.LPIPS(reduction='none')(img_pred, img_gt)]
            ssims += [piq.ssim(img_pred, img_gt)]
        
    # if args.dataset_name in {'blender','llff'} or \
    #   (args.dataset_name == 'phototourism' and args.split == 'test'):
    imageio.mimsave(os.path.join(dir_name, f'{args.scene_name}.{args.video_format}'),
                        imgs, fps=24) # duration=0.04)
        # imageio.mimsave(os.path.join(dir_name, f'R_{args.scene_name}.{args.video_format}'),img_reflection, fps=30)
    if psnrs:
        mean_psnr = np.mean(psnrs)
        mean_piq = np.mean(piqs)
        mean_ssim = np.mean(ssims)
        print(f'Mean PSNR : {mean_psnr:.2f}')
        print(f'Mean SSIM : {mean_ssim:.2f}')
        print(f'Mean LPIPS : {mean_piq:.2f}')