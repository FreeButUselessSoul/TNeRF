import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import os
import pandas as pd
import cv2
import pickle
from PIL import Image
from torchvision import transforms as T
from models.encoder import *
from .ray_utils import *
from .colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary

pic_pool = []
def add_reflection(img, perturbation, seed):
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
                    # tw,th=temp.size
                    # temp = temp.resize((tw//3,th//3))
                    # temp = np.array(temp)
                    pic_pool.append(temp)
        selected = pic_pool[np.random.choice(len(pic_pool))]
        selected = selected.resize((w,h))
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
        # reflection.putalpha(int((np.random.rand()*0.4+0.1)*256)) # Random alpha
        # reflection = reflection.crop((0, 0, int(0.5*w), int(0.5*h)))
        # img.paste(reflection,(left,top,left+int(0.5*w),top+int(0.5*h)),reflection) 
    return img

# im2tensor = T.Compose([
#     T.ToTensor(),
#     T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
# ])
    

class PhototourismDataset(Dataset):
    def __init__(self, root_dir, split='train', img_downscale=1, val_num=1,
                 use_cache=False, perturbation=[], a_model=None, N_vocab=1500, p_model=None, test_image=None, **kwargs):
        """
        img_downscale: how much scale to downsample the training images.
                       The original image sizes are around 500~100, so value of 1 or 2
                       are recommended.
                       ATTENTION! Value of 1 will consume large CPU memory,
                       about 40G for brandenburg gate.
        val_num: number of val images (used for multigpu, validate same image for all gpus)
        use_cache: during data preparation, use precomputed rays (useful to accelerate
                   data loading, especially for multigpu!)
        """
        self.root_dir = root_dir
        self.split = split
        assert img_downscale >= 1, 'image can only be downsampled, please set img_downscale>=1!'
        self.img_downscale = img_downscale
        if split == 'val': # image downscale=1 will cause OOM in val mode
            self.img_downscale = max(2, self.img_downscale)
        self.val_num = max(1, val_num) # at least 1
        self.use_cache = use_cache
        self.perturbation = perturbation
        self.a_model = a_model
        self.p_model = p_model
        self.N_vocab = N_vocab
        if self.split == 'train':
            print(f'add {self.perturbation} perturbation!')
        if self.split == 'test':
            self.test_img_w = kwargs['test_img_w']
            self.test_img_h = kwargs['test_img_h']
        self.define_transforms()
        self.test_image = test_image
        self.w_reflection = None

        self.read_meta()
        self.white_back = False

    def read_meta(self):
        # read all files in the tsv first (split to train and test later)
        tsv = glob.glob(os.path.join(self.root_dir, '*.tsv'))[0]
        self.scene_name = os.path.basename(tsv)[:-4]
        self.files = pd.read_csv(tsv, sep='\t')
        self.files = self.files[~self.files['id'].isnull()] # remove data without id
        self.files.reset_index(inplace=True, drop=True)
        #### ENCODE APPEARANCE ####
        if self.a_model is not None:
            encoder = ImageEncoder(self.a_model)
            encoder.eval()
        if self.p_model is not None:
            p_encoder = SpatialEncoder(self.p_model)
            p_encoder.eval()
        # Step 1. load image paths
        # Attention! The 'id' column in the tsv is BROKEN, don't use it!!!!
        # Instead, read the id from images.bin using image file name!
        if self.use_cache:
            with open(os.path.join(self.root_dir, f'cache/img_ids.pkl'), 'rb') as f:
                self.img_ids = pickle.load(f)
            with open(os.path.join(self.root_dir, f'cache/image_paths.pkl'), 'rb') as f:
                self.image_paths = pickle.load(f)
        else:
            imdata = read_images_binary(os.path.join(self.root_dir, 'dense/sparse/images.bin'))
            img_path_to_id = {}
            for v in imdata.values():
                img_path_to_id[v.name] = v.id
            self.img_ids = []
            self.image_paths = {} # {id: filename}
            for filename in list(self.files['filename']):
                id_ = img_path_to_id[filename]
                self.image_paths[id_] = filename
                self.img_ids += [id_]

        # Step 2: read and rescale camera intrinsics
        if self.use_cache:
            with open(os.path.join(self.root_dir, f'cache/Ks{self.img_downscale}.pkl'), 'rb') as f:
                self.Ks = pickle.load(f)
        else:
            self.Ks = {} # {id: K}
            camdata = read_cameras_binary(os.path.join(self.root_dir, 'dense/sparse/cameras.bin'))
            for id_ in self.img_ids:
                K = np.zeros((3, 3), dtype=np.float32)
                cam = camdata[id_]
                img_w, img_h = int(cam.params[2]*2), int(cam.params[3]*2)
                img_w_, img_h_ = img_w//self.img_downscale, img_h//self.img_downscale
                K[0, 0] = cam.params[0]*img_w_/img_w # fx
                K[1, 1] = cam.params[1]*img_h_/img_h # fy
                K[0, 2] = cam.params[2]*img_w_/img_w # cx
                K[1, 2] = cam.params[3]*img_h_/img_h # cy
                K[2, 2] = 1
                self.Ks[id_] = K

        # Step 3: read c2w poses (of the images in tsv file only) and correct the order
        if self.use_cache:
            self.poses = np.load(os.path.join(self.root_dir, 'cache/poses.npy'))
        else:
            w2c_mats = []
            bottom = np.array([0, 0, 0, 1.]).reshape(1, 4)
            for id_ in self.img_ids:
                im = imdata[id_]
                R = im.qvec2rotmat()
                t = im.tvec.reshape(3, 1)
                w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
            w2c_mats = np.stack(w2c_mats, 0) # (N_images, 4, 4)
            self.poses = np.linalg.inv(w2c_mats)[:, :3] # (N_images, 3, 4)
            # Original poses has rotation in form "right down front", change to "right up back"
            self.poses[..., 1:3] *= -1

        # Step 4: correct scale
        if self.use_cache:
            self.xyz_world = np.load(os.path.join(self.root_dir, 'cache/xyz_world.npy'))
            with open(os.path.join(self.root_dir, f'cache/nears.pkl'), 'rb') as f:
                self.nears = pickle.load(f)
            with open(os.path.join(self.root_dir, f'cache/fars.pkl'), 'rb') as f:
                self.fars = pickle.load(f)
        else:
            pts3d = read_points3d_binary(os.path.join(self.root_dir, 'dense/sparse/points3D.bin'))
            self.xyz_world = np.array([pts3d[p_id].xyz for p_id in pts3d])
            xyz_world_h = np.concatenate([self.xyz_world, np.ones((len(self.xyz_world), 1))], -1)
            # Compute near and far bounds for each image individually
            self.nears, self.fars = {}, {} # {id_: distance}
            for i, id_ in enumerate(self.img_ids):
                xyz_cam_i = (xyz_world_h @ w2c_mats[i].T)[:, :3] # xyz in the ith cam coordinate
                xyz_cam_i = xyz_cam_i[xyz_cam_i[:, 2]>0] # filter out points that lie behind the cam
                self.nears[id_] = np.percentile(xyz_cam_i[:, 2], 0.1)
                self.fars[id_] = np.percentile(xyz_cam_i[:, 2], 99.9)

            max_far = np.fromiter(self.fars.values(), np.float32).max()
            scale_factor = max_far/5 # so that the max far is scaled to 5
            self.poses[..., 3] /= scale_factor
            for k in self.nears:
                self.nears[k] /= scale_factor
            for k in self.fars:
                self.fars[k] /= scale_factor
            self.xyz_world /= scale_factor
        self.poses_dict = {id_: self.poses[i] for i, id_ in enumerate(self.img_ids)}
            
        # Step 5. split the img_ids (the number of images is verfied to match that in the paper)
        self.img_ids_train = [id_ for i, id_ in enumerate(self.img_ids) 
                                    if self.files.loc[i, 'split']=='train']
        self.img_ids_test = [id_ for i, id_ in enumerate(self.img_ids)
                                    if self.files.loc[i, 'split']=='test']
        self.N_images_train = len(self.img_ids_train)
        self.N_images_test = len(self.img_ids_test)
        if self.split == 'train': # create buffer of all rays and rgb data
            if self.use_cache:
                all_rays = np.load(os.path.join(self.root_dir,
                                                f'cache/rays{self.img_downscale}.npy'))
                self.all_rays = torch.from_numpy(all_rays)
                all_rgbs = np.load(os.path.join(self.root_dir,
                                                f'cache/rgbs{self.img_downscale}.npy'))
                self.all_rgbs = torch.from_numpy(all_rgbs)
                all_originals = np.load(os.path.join(self.root_dir,
                                                f'cache/ori{self.img_downscale}.npy'))
                self.all_originals = torch.from_numpy(all_originals)
                if self.p_model is not None:
                    all_features = np.load(os.path.join(self.root_dir,
                                                    f'cache/features{self.img_downscale}.npy'))
                    self.all_features = torch.from_numpy(all_features)
                if self.a_model is not None:
                    img2embedding = np.load(os.path.join(self.root_dir, f'cache/embedding_a.npy'))
                    self.img2embedding = torch.from_numpy(img2embedding)
            else:
                if self.a_model is not None:
                    self.img2embedding = torch.zeros((self.N_vocab,encoder.latent_size))
                self.all_rays = []
                self.all_rgbs = []
                self.all_originals = []
                if self.p_model is not None:
                    self.all_features = []
                for id_ in self.img_ids_test:
                    img = Image.open(os.path.join(self.root_dir, 'dense/images',
                                                  self.image_paths[id_])).convert('RGB')
                    if self.a_model is not None:
                        embedding_a = encoder(T.ToTensor()(img).unsqueeze(0)).detach().squeeze()
                        self.img2embedding[id_] = embedding_a
                for t,id_ in enumerate(self.img_ids_train):
                    c2w = torch.FloatTensor(self.poses_dict[id_])

                    img = Image.open(os.path.join(self.root_dir, 'dense/images',
                                                  self.image_paths[id_])).convert('RGB')
                    # Add reflection
                    img_w, img_h = img.size
                    if self.perturbation:
                        img_r = add_reflection(img, self.perturbation, t)
                    img_r.save(f'debug/{t}.png')
                    if self.img_downscale > 1:
                        img_w = img_w//self.img_downscale
                        img_h = img_h//self.img_downscale
                        img_r = img_r.resize((img_w, img_h), Image.LANCZOS)
                        img = img.resize((img_w,img_h), Image.LANCZOS)
                    img = self.transform(img) # (3, h, w)
                    img_r = self.transform(img_r)
                    if self.p_model is not None: # Point-wise feature
                        pointwise_feature = p_encoder(img_r.unsqueeze(0)).detach().squeeze(0)
                        pointwise_feature = pointwise_feature.view(p_encoder.latent_size,-1).permute(1,0) # (h*w, ?)
                        self.all_features += [pointwise_feature]
                        # import ipdb;ipdb.set_trace()
                    #### ENCODE APPEARANCE ####
                    if self.a_model is not None:
                        embedding_a = encoder(img_r.unsqueeze(0)).detach().squeeze()
                        self.img2embedding[id_] = embedding_a
                    img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGB
                    img_r = img_r.view(3, -1).permute(1, 0)
                    self.all_rgbs += [img_r]
                    self.all_originals += [img]
                    
                    directions = get_ray_directions(img_h, img_w, self.Ks[id_])
                    rays_o, rays_d = get_rays(directions, c2w)
                    rays_t = id_ * torch.ones(len(rays_o), 1)

                    self.all_rays += [torch.cat([rays_o, rays_d,
                                                self.nears[id_]*torch.ones_like(rays_o[:, :1]),
                                                self.fars[id_]*torch.ones_like(rays_o[:, :1]),
                                                rays_t],
                                                1)] # (h*w, 8)
                                  
                self.all_rays = torch.cat(self.all_rays, 0) # ((N_images-1)*h*w, 8)
                self.all_rgbs = torch.cat(self.all_rgbs, 0) # ((N_images-1)*h*w, 3)
                self.all_originals = torch.cat(self.all_originals, 0)
                if self.p_model is not None:
                    self.all_features = torch.cat(self.all_features, 0)
                
        
        elif self.split in ['val', 'test_train']: # use the first image as val image (also in train)
            self.val_id = self.img_ids_train[0]

        else: # for testing, create a parametric rendering path
            # test poses and appearance index are defined in eval.py
            if self.test_image:
                image = Image.open(os.path.join(self.root_dir, 'dense/images',
                                          self.image_paths[self.test_image])).convert('RGB')
                image = image.resize((self.test_img_w, self.test_img_h))
                self.w_reflection = add_reflection(image, 'llff_im', 0)
                img_r = self.transform(self.w_reflection) # (3, h, w)
                img = self.transform(image)
                self.all_rgbs = img_r
                self.all_originals = img
                pointwise_feature = p_encoder(image.unsqueeze(0)).detach().squeeze(0)
                pointwise_feature = pointwise_feature.view(p_encoder.latent_size,-1).permute(1,0) # (h*w, ?)
                self.all_features = pointwise_feature

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        if self.split == 'test_train':
            return self.N_images_train
        if self.split == 'val':
            return self.val_num
        return len(self.poses_test)

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            sample = {'rays': self.all_rays[idx, :8],
                      'ts': self.all_rays[idx, 8].long(),
                      'rgbs': self.all_rgbs[idx],
                      'features': torch.Tensor(),
                      'original': self.all_originals[idx]}
            if self.p_model is not None:
                sample['features']=self.all_features[idx]

        elif self.split in ['val', 'test_train']:
            sample = {'features': torch.Tensor()}
            if self.split == 'val':
                id_ = self.val_id
            else:
                id_ = self.img_ids_train[idx]
            sample['c2w'] = c2w = torch.FloatTensor(self.poses_dict[id_])

            img = Image.open(os.path.join(self.root_dir, 'dense/images',
                                          self.image_paths[id_])).convert('RGB')
            if self.perturbation:
                img_r = add_reflection(img, self.perturbation, idx)
            img_w, img_h = img.size
            if self.img_downscale > 1:
                img_w = img_w//self.img_downscale
                img_h = img_h//self.img_downscale
                img = img.resize((img_w, img_h), Image.LANCZOS)
                img_r = img_r.resize((img_w,img_h), Image.LANCZOS)
            img = self.transform(img) # (3, h, w)
            img_r = self.transform(img_r)
            sample['rgbs'] = img_r.view(3, -1).permute(1, 0) # (h*w, 3) RGB
            sample['original'] = img.view(3, -1).permute(1, 0)
            if self.p_model is not None:
                p_encoder = SpatialEncoder(self.p_model)
                p_encoder.eval()
                temp = p_encoder(img_r.unsqueeze(0)).squeeze(0).detach().view(p_encoder.latent_size,-1).permute(1,0) # (h*w, ?)
                sample['features'] = temp

            directions = get_ray_directions(img_h, img_w, self.Ks[id_])
            rays_o, rays_d = get_rays(directions, c2w)
            rays = torch.cat([rays_o, rays_d,
                              self.nears[id_]*torch.ones_like(rays_o[:, :1]),
                              self.fars[id_]*torch.ones_like(rays_o[:, :1])],
                              1) # (h*w, 8)
            sample['rays'] = rays
            sample['ts'] = id_ * torch.ones(len(rays), dtype=torch.long)
            sample['img_wh'] = torch.LongTensor([img_w, img_h])

        else:
            sample = {}
            sample['c2w'] = c2w = torch.FloatTensor(self.poses_test[idx])
            directions = get_ray_directions(self.test_img_h, self.test_img_w, self.test_K)
            rays_o, rays_d = get_rays(directions, c2w)
            near, far = 0, 5
            rays = torch.cat([rays_o, rays_d,
                              near*torch.ones_like(rays_o[:, :1]),
                              far*torch.ones_like(rays_o[:, :1])],
                              1)
            sample['rays'] = rays
            sample['ts'] = self.test_appearance_idx * torch.ones(len(rays), dtype=torch.long)
            sample['img_wh'] = torch.LongTensor([self.test_img_w, self.test_img_h])
            if self.test_image:
                sample['features'] = self.all_features
                sample['rgbs'] = self.all_rgbs
                sample['original'] = self.all_originals
            # print(rays.shape,sample['ts'].shape,sample['features'].shape)

        return sample
