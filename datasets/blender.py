import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image, ImageDraw
from torchvision import transforms as T
from models.encoder import *
from .ray_utils import *
import cv2

pic_pool = []
def add_perturbation(img, perturbation, seed):
    if 'color' in perturbation:
        np.random.seed(seed)
        img_np = np.array(img)/255.0
        s = np.random.uniform(0.8, 1.2, size=3)
        b = np.random.uniform(-0.2, 0.2, size=3)
        img_np[..., :3] = np.clip(s*img_np[..., :3]+b, 0, 1)
        img = Image.fromarray((255*img_np).astype(np.uint8))
    if 'occ' in perturbation:
        draw = ImageDraw.Draw(img)
        np.random.seed(seed)
        left = np.random.randint(200, 400)
        top = np.random.randint(200, 400)
        for i in range(10):
            np.random.seed(10*seed+i)
            random_color = tuple(np.random.choice(range(256), 3))
            draw.rectangle(((left+20*i, top), (left+20*(i+1), top+200)),
                            fill=random_color)
    if 'opq_occ' in perturbation:
        draw = ImageDraw.Draw(img)
        np.random.seed(seed)
        left = np.random.randint(200, 400)
        top = np.random.randint(200, 400)
        for i in range(10):
            np.random.seed(10*seed+i)
            # random_color = tuple(np.random.choice(range(256), 4))
            random_color = tuple([*np.random.choice(range(256), 3),np.random.randint(64,192)])
            draw.rectangle(((left+20*i, top), (left+20*(i+1), top+200)),
                            fill=random_color)
    if 'llff_im' in perturbation:
        w,h = img.size
        img = np.array(img,dtype='float') / 255.
        np.random.seed(seed)
        left = 0 #np.random.randint(0, 0.05*w)
        top = 0 #np.random.randint(0, 0.05*h)
        if not pic_pool:
            for dir in os.listdir('data/nerf_llff_data/'):
                dirname = os.path.join('data/nerf_llff_data/',dir,'images/')
                for pic in os.listdir(dirname):
                    temp = Image.open(os.path.join(dirname,pic)).convert('RGB')
                    pic_pool.append(temp)
        selected = pic_pool[np.random.choice(len(pic_pool))]
        selected = selected.resize((w,h))
        selected.putalpha(255)
        selected = np.array(selected,dtype='float')/255.

        im2 = np.zeros_like(img)
        im2[top:top+h,left:left+w]=selected
        reflection = cv2.GaussianBlur(im2,ksize=None, sigmaX=2)
        im2 = img + reflection # img is background
        average = np.mean(im2[im2>1])
        reflection2 = reflection - 0.5*(average-0.9)
        reflection2 = np.clip(reflection2,0,1) # reflection
        mixture = 0.3 * reflection2 + 0.8 * img
        mixture = np.clip(mixture,0,1)
        return Image.fromarray(np.uint8(mixture*255))
    return img


class BlenderDataset(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(800, 800),
                 perturbation=[], p_model=None):
        self.root_dir = root_dir
        self.split = split
        assert img_wh[0] == img_wh[1], 'image width must equal image height!'
        self.img_wh = img_wh
        self.p_model = p_model
        self.w_reflection = None
        if p_model is not None:
            self.p_encoder = SpatialEncoder(self.p_model)
            self.p_encoder.eval()
        self.define_transforms()

        # assert set(perturbation).issubset({"color", "occ"}), \
        #     'Only "color" and "occ" perturbations are supported!'
        self.perturbation = perturbation
        if self.split == 'train':
            print(f'add {self.perturbation} perturbation!')
        self.read_meta()
        self.white_back = True

    def read_meta(self):
        with open(os.path.join(self.root_dir,
                               f"transforms_{self.split.split('_')[-1]}.json"), 'r') as f:
            self.meta = json.load(f)

        w, h = self.img_wh
        self.focal = 0.5*800/np.tan(0.5*self.meta['camera_angle_x']) # original focal length
                                                                     # when W=800

        self.focal *= self.img_wh[0]/800 # modify focal length to match size self.img_wh
        self.K = np.eye(3)
        self.K[0, 0] = self.K[1, 1] = self.focal
        self.K[0, 2] = w/2
        self.K[1, 2] = h/2

        # bounds, common for all scenes
        self.near = 2.0
        self.far = 6.0
        self.bounds = np.array([self.near, self.far])
        
        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(h, w, self.K) # (h, w, 3)
            
        if self.split == 'train': # create buffer of all rays and rgb data
            self.all_rays = []
            self.all_rgbs = []
            self.all_originals = []
            if self.p_model is not None:
                self.all_features = []
            for t, frame in enumerate(self.meta['frames']):
                pose = np.array(frame['transform_matrix'])[:3, :4]
                c2w = torch.FloatTensor(pose)

                image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
                img = Image.open(image_path)
                img_ori = img.resize(self.img_wh, Image.LANCZOS)
                img_ori = self.transform(img_ori)
                img_ori = img_ori.view(4, -1).permute(1, 0) # (h*w, 4) RGBA
                img_ori = img_ori[:, :3]*img_ori[:, -1:] + (1-img_ori[:, -1:]) # blend A to RGB
                self.all_originals += [img_ori]

                if t != 0: # perturb everything except the first image.
                           # cf. Section D in the supplementary material
                    img = add_perturbation(img, self.perturbation, t)

                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img) # (4, h, w)
                img = img[:3] * img[3] + (1-img[3]) # blend A to RGB
                if self.p_model is not None:
                    pointwise_feature = self.p_encoder(img.unsqueeze(0)).detach().squeeze(0)
                    pointwise_feature = pointwise_feature.view(self.p_encoder.latent_size,-1).permute(1,0) # (h*w, ?)
                    self.all_features += [pointwise_feature]
                img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGBA
                self.all_rgbs += [img]
                
                rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)
                rays_t = t * torch.ones(len(rays_o), 1)

                self.all_rays += [torch.cat([rays_o, rays_d,
                                             self.near*torch.ones_like(rays_o[:, :1]),
                                             self.far*torch.ones_like(rays_o[:, :1]),
                                             rays_t],
                                             1)] # (h*w, 8)

            self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_originals = torch.cat(self.all_originals, 0)
            if self.p_model is not None:
                self.all_features = torch.cat(self.all_features, 0)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        if self.split == 'val':
            return 8 # only validate 8 images (to support <=8 gpus)
        return len(self.meta['frames'])

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            sample = {'rays': self.all_rays[idx, :8],
                      'ts': self.all_rays[idx, 8].long(),
                      'rgbs': self.all_rgbs[idx],
                      'features': torch.Tensor() if self.p_model is None else self.all_features[idx],
                      'original': self.all_originals[idx]}

        else: # create data for each image separately
            frame = self.meta['frames'][idx]
            c2w = torch.FloatTensor(frame['transform_matrix'])[:3, :4]
            t = 0 # transient embedding index, 0 for val and test (no perturbation)

            img = Image.open(os.path.join(self.root_dir, f"{frame['file_path']}.png"))
            img_ori = img.resize(self.img_wh, Image.LANCZOS)
            img_ori = self.transform(img_ori)
            img_ori = img_ori.view(4, -1).permute(1, 0) # (h*w, 4) RGBA
            img_ori = img_ori[:, :3]*img_ori[:, -1:] + (1-img_ori[:, -1:]) # blend A to RGB
            if (self.split == 'test_train' and idx != 0) or self.p_model is not None:
                t = idx % 50
                img = add_perturbation(img, 'llff_im', t)
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img) # (4, H, W)
            valid_mask = (img_ori[-1]>0).flatten() # (H*W) valid color area
            img = img[:3] * img[3] + (1-img[3])
            if self.p_model is not None:
                features = self.p_encoder(img.unsqueeze(0)).detach().squeeze(0)
                features = features.view(self.p_encoder.latent_size,-1).permute(1,0)
            else:
                features = torch.Tensor()
            img = img.view(3, -1).permute(1, 0) # (H*W, 3)
            rays_o, rays_d = get_rays(self.directions, c2w)

            rays = torch.cat([rays_o, rays_d, 
                              self.near*torch.ones_like(rays_o[:, :1]),
                              self.far*torch.ones_like(rays_o[:, :1])],
                              1) # (H*W, 8)

            sample = {'rays': rays,
                      'ts': t * torch.ones(len(rays), dtype=torch.long),
                      'rgbs': img,
                      'original': img_ori,
                      'features':features,
                      'c2w': c2w,
                      'valid_mask': valid_mask}

            if self.split == 'test_train' and self.perturbation:
                 # append the original (unperturbed) image
                img = Image.open(os.path.join(self.root_dir, f"{frame['file_path']}.png"))
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img) # (4, H, W)
                valid_mask = (img[-1]>0).flatten() # (H*W) valid color area
                img = img.view(4, -1).permute(1, 0) # (H*W, 4) RGBA
                img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB
                sample['original_rgbs'] = img
                sample['original_valid_mask'] = valid_mask

        return sample