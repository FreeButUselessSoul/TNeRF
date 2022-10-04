import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import os
from PIL import Image, ImageDraw
from torchvision import transforms as T
import cv2
from .ray_utils import *
from .ray_sample import FlexGridRaySampler,ImgToPatch

pic_pool = []
def add_perturbation(img, perturbation, seed):
    if 'llff_im' in perturbation or 'llff_im2' in perturbation:
        if not pic_pool:
            dirname ='data/nerf_llff_data/horns/images/'
            for pic in os.listdir(dirname):
                temp = Image.open(os.path.join(dirname,pic)).convert('RGB')
                pic_pool.append(temp)
        w,h = img.size
        img = np.array(img,dtype='float') / 255.
        np.random.seed(seed*3)
        left = np.random.randint(0, w//8)
        top = np.random.randint(0, h//8)
        
        selected = pic_pool[np.random.choice(len(pic_pool))]
        selected = selected.crop((left,top,left+w,top+h))
        selected = np.array(selected,dtype='float')/255.

        reflection = np.zeros_like(img)
        # im2[top:top+h//3,left:left+w//3]=selected
        # reflection[top:top+h//2,left:left+w//2] = cv2.GaussianBlur(selected,ksize=None, sigmaX=2)
        reflection = cv2.GaussianBlur(selected,ksize=None, sigmaX=2)
        im2 = img + reflection # img is background
        average = np.mean(im2[im2>1])
        reflection2 = reflection - 0.5*(average-0.9)
        reflection2 = np.clip(reflection2,0,1) # reflection
        mixture = img
        if 'llff_im' in perturbation:
            mixture = (0.3 * reflection2 + 0.8 * img)
        elif 'llff_im2' in perturbation:
            mixture = (0.5 * reflection2 + 0.6 * img)
        mixture = np.clip(mixture,0,1)
        return Image.fromarray(np.uint8(mixture*255))
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
    return img

def normalize(v):
    """Normalize a vector."""
    return v/np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.
    
    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0) # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0)) # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0) # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z)) # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x) # (3)

    pose_avg = np.stack([x, y, z, center], 1) # (3, 4)

    return pose_avg


def center_poses(poses):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """

    pose_avg = average_poses(poses) # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg # convert to homogeneous coordinate for faster computation
                                 # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1)) # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1) # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3] # (N_images, 3, 4)

    return poses_centered, pose_avg


def create_spiral_poses(radii, focus_depth, n_poses=120):
    """
    Computes poses that follow a spiral path for rendering purpose.
    See https://github.com/Fyusion/LLFF/issues/19
    In particular, the path looks like:
    https://tinyurl.com/ybgtfns3

    Inputs:
        radii: (3) radii of the spiral for each axis
        focus_depth: float, the depth that the spiral poses look at
        n_poses: int, number of poses to create along the path

    Outputs:
        poses_spiral: (n_poses, 3, 4) the poses in the spiral path
    """

    poses_spiral = []
    for t in np.linspace(0, 4*np.pi, n_poses+1)[:-1]: # rotate 4pi (2 rounds)
        # the parametric function of the spiral (see the interactive web)
        center = np.array([np.cos(t), -np.sin(t), -np.sin(0.5*t)]) * radii

        # the viewing z axis is the vector pointing from the @focus_depth plane
        # to @center
        z = normalize(center - np.array([0, 0, -focus_depth]))
        
        # compute other axes as in @average_poses
        y_ = np.array([0, 1, 0]) # (3)
        x = normalize(np.cross(y_, z)) # (3)
        y = np.cross(z, x) # (3)

        poses_spiral += [np.stack([x, y, z, center], 1)] # (3, 4)

    return np.stack(poses_spiral, 0) # (n_poses, 3, 4)


def create_spheric_poses(radius, n_poses=120):
    """
    Create circular poses around z axis.
    Inputs:
        radius: the (negative) height and the radius of the circle.

    Outputs:
        spheric_poses: (n_poses, 3, 4) the poses in the circular path
    """
    def spheric_pose(theta, phi, radius):
        trans_t = lambda t : np.array([
            [1,0,0,0],
            [0,1,0,-0.9*t],
            [0,0,1,t],
            [0,0,0,1],
        ])

        rot_phi = lambda phi : np.array([
            [1,0,0,0],
            [0,np.cos(phi),-np.sin(phi),0],
            [0,np.sin(phi), np.cos(phi),0],
            [0,0,0,1],
        ])

        rot_theta = lambda th : np.array([
            [np.cos(th),0,-np.sin(th),0],
            [0,1,0,0],
            [np.sin(th),0, np.cos(th),0],
            [0,0,0,1],
        ])

        c2w = rot_theta(theta) @ rot_phi(phi) @ trans_t(radius)
        c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
        return c2w[:3]

    spheric_poses = []
    for th in np.linspace(0, 2*np.pi, n_poses+1)[:-1]:
        spheric_poses += [spheric_pose(th, -np.pi/5, radius)] # 36 degree view downwards
    return np.stack(spheric_poses, 0)


class LLFFPatched(Dataset):
    def __init__(self,
                root_dir, split='train',
                img_wh=(504, 378),
                spheric_poses=False,
                val_num=1, perturbation=[],
                sample_wh=64):
        """
        spheric_poses: whether the images are taken in a spheric inward-facing manner
                       default: False (forward-facing)
        val_num: number of val images (used for multigpu training, validate same image for all gpus)
        """
        print("Using LLFF Patched dataset.")
        self.root_dir = root_dir
        self.split = split
        self.img_wh = img_wh
        self.spheric_poses = spheric_poses
        self.val_num = max(1, val_num) # at least 1
        self.perturbation = perturbation
        if split=='train':
            print(f'add {self.perturbation} perturbation!')
        self.define_transforms()
        self.img2patch = ImgToPatch(FlexGridRaySampler(sample_wh,img_wh=self.img_wh), sample_wh)
        self.read_meta()
        self.white_back = False

    def read_meta(self):
        poses_bounds = np.load(os.path.join(self.root_dir,
                                            'poses_bounds.npy')) # (N_images, 17)
        temp=sorted(glob.glob(os.path.join(self.root_dir, 'images/*')))
        self.filenames = sorted(glob.glob(os.path.join(self.root_dir, 'images/*')))
        self.image_paths = sorted(glob.glob(os.path.join(self.root_dir, 'images/*.jpg')))
                        # load full resolution image then resize
        indices = [temp.index(x) for x in self.image_paths]
        print(indices)
        poses_bounds = poses_bounds[indices]
        self.train_list =list(range(len(self.image_paths)))#[int(filename.split('.')[0]) for filename in self.image_paths]
        if self.split in ['train', 'val']:
            assert len(poses_bounds) == len(self.image_paths), \
                'Mismatch between number of images and number of poses! Please rerun COLMAP!'

        poses = poses_bounds[:, :15].reshape(-1, 3, 5) # (N_images, 3, 5)
        self.bounds = poses_bounds[:, -2:] # (N_images, 2)

        # Step 1: rescale focal length according to training resolution
        H, W, self.focal = poses[0, :, -1] # original intrinsics, same for all images
        # assert H*self.img_wh[0] == W*self.img_wh[1], \
        #     f'You must set @img_wh to have the same aspect ratio as ({W}, {H}) !'
        
        self.focal *= self.img_wh[0]/W
        self.K = np.eye(3)
        self.K[0, 0] = self.K[1, 1] = self.focal
        self.K[0, 2] = self.img_wh[0]/2
        self.K[1, 2] = self.img_wh[1]/2

        # Step 2: correct poses
        # Original poses has rotation in form "down right back", change to "right up back"
        # See https://github.com/bmild/nerf/issues/34
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
                # (N_images, 3, 4) exclude H, W, focal
        self.poses, self.pose_avg = center_poses(poses)
        distances_from_center = np.linalg.norm(self.poses[..., 3], axis=1)
        val_idx = VAL_IDX#np.argmin(distances_from_center) # choose val image as the closest to
                                                   # center image

        # Step 3: correct scale so that the nearest depth is at a little more than 1.0
        # See https://github.com/bmild/nerf/issues/34
        near_original = self.bounds.min()
        scale_factor = near_original*0.75 # 0.75 is the default parameter
                                          # the nearest depth is at 1/0.75=1.33
        self.bounds /= scale_factor
        self.poses[..., 3] /= scale_factor

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(self.img_wh[1], self.img_wh[0], self.K) # (H, W, 3)
            
        if self.split == 'train': # create buffer of all rays and rgb data
                                  # use first N_images-1 to train, the LAST is val
            # self.all_rays = []
            # self.all_rgbs = []
            # self.all_originals = []
            self.all_rays_o = []
            self.all_rays_d = []
            self.near_far = []
            self.all_rgbs = [] # {id: tensor of (3,H,W)}
            del(self.train_list[val_idx]) # exclude the val image
            for i, image_path in enumerate(self.image_paths):
                # if i == val_idx: # exclude the val image
                if i not in self.train_list:
                    continue
                pose_index = self.filenames.index(image_path)
                c2w = torch.FloatTensor(self.poses[pose_index])

                img = Image.open(image_path).convert('RGB')
                # img_ori = img.resize(self.img_wh, Image.LANCZOS)
                if i != 0:
                    img = add_perturbation(img, self.perturbation, i)
                img = img.resize(self.img_wh, Image.LANCZOS)
                # img.save(f'debug2/{i}.png')
                img = self.transform(img) # (3, h, w)
                # img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGB
                self.all_rgbs += [img]
                rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)
                if not self.spheric_poses:
                    near, far = 0, 1
                    rays_o, rays_d = get_ndc_rays(self.img_wh[1], self.img_wh[0],
                                                  self.focal, 1.0, rays_o, rays_d)
                                     # near plane is always at 1.0
                                     # near and far in NDC are always 0 and 1
                                     # See https://github.com/bmild/nerf/issues/34
                else:
                    near = self.bounds.min()
                    far = min(8 * near, self.bounds.max()) # focus on central object only
                rays_o = rays_o.reshape(img.shape)
                rays_d = rays_d.reshape(img.shape)
                self.all_rays_o += [rays_o]
                self.all_rays_d += [rays_d]
                self.near_far += [torch.Tensor([near,far])]

            self.N_pics = len(self.all_rays_o)
            # self.all_originals = torch.cat(self.all_originals, 0)
            # self.all_rgbs = torch.cat(self.all_rgbs, 0) # ((N_images-1)*h*w, 3)
        
        elif self.split == 'val':
            print('val image is', self.image_paths[val_idx])
            self.c2w_val = self.poses[val_idx]
            self.image_path_val = self.image_paths[val_idx]

        else: # for testing, create a parametric rendering path
            if self.split.endswith('train'): # test on training set
                self.poses_test = self.poses
            elif not self.spheric_poses:
                focus_depth = 3.5 # hardcoded, this is numerically close to the formula
                                  # given in the original repo. Mathematically if near=1
                                  # and far=infinity, then this number will converge to 4
                radii = np.percentile(np.abs(self.poses[..., 3]), 90, axis=0)
                self.poses_test = create_spiral_poses(radii, focus_depth)
            else:
                radius = 1.1 * self.bounds.min()
                self.poses_test = create_spheric_poses(radius)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return self.N_pics * (self.img2patch.len)
        if self.split == 'val':
            return self.val_num
        return len(self.poses_test)

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            pic_idx = idx%self.N_pics
            rays_o = self.all_rays_o[pic_idx]
            rays_d = self.all_rays_d[pic_idx]
            rgb = self.all_rgbs[pic_idx]
            patchID = idx // self.N_pics
            rgb,rays_o,rays_d = self.img2patch([rgb,rays_o,rays_d],patchID)
            rays = torch.cat([rays_o,rays_d,self.near_far[pic_idx]*torch.ones((len(rays_o),2))],1)
            sample = {'rays': rays,
                      'ts': pic_idx * torch.ones((len(rays_o),1), dtype=torch.long),
                      'rgbs': rgb}

        else:
            if self.split == 'val':
                c2w = torch.FloatTensor(self.c2w_val)
            else:
                c2w = torch.FloatTensor(self.poses_test[idx])

            rays_o, rays_d = get_rays(self.directions, c2w)
            if not self.spheric_poses:
                near, far = 0, 1
                rays_o, rays_d = get_ndc_rays(self.img_wh[1], self.img_wh[0],
                                              self.focal, 1.0, rays_o, rays_d)
            else:
                near = self.bounds.min()
                far = min(8 * near, self.bounds.max())

            rays = torch.cat([rays_o, rays_d, 
                              near*torch.ones_like(rays_o[:, :1]),
                              far*torch.ones_like(rays_o[:, :1])],
                              1) # (h*w, 8)

            sample = {'rays': rays,
                      'ts': torch.zeros(len(rays), dtype=torch.long),
                      'features': torch.Tensor(),
                      'c2w': c2w}

            if self.split == 'val':
                img = Image.open(self.image_path_val).convert('RGB')
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img) # (3, h, w)
                img = img.view(3, -1).permute(1, 0) # (h*w, 3)
                sample['rgbs'] = img
                sample['original'] = img

        return sample
