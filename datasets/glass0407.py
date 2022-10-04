import torch
from torch.utils.data import Dataset
from torch import nn
import glob
import numpy as np
import os
from PIL import Image, ImageDraw
from torchvision import transforms as T
import cv2
from .ray_utils import *
from .ray_sample import FlexGridRaySampler,ImgToPatch
import torch.nn.functional as F
from torchvision import models
from .depth_utils import load_colmap_depth
from .utils import Enc0,DRNet


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        # vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.vgg_pretrained_features = models.vgg19(pretrained=True).features
        # self.slice1 = torch.nn.Sequential()
        # self.slice2 = torch.nn.Sequential()
        # self.slice3 = torch.nn.Sequential()
        # self.slice4 = torch.nn.Sequential()
        # self.slice5 = torch.nn.Sequential()
        # for x in range(2):
        #     self.slice1.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(2, 7):
        #     self.slice2.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(7, 12):
        #     self.slice3.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(12, 21):
        #     self.slice4.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(21, 30):
        #     self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X, indices=None):
        if indices is None:
            indices = [2, 7, 12, 21, 30]
        out = []
        #indices = sorted(indices)
        for i in range(indices[-1]):
            X = self.vgg_pretrained_features[i](X)
            if (i+1) in indices:
                out.append(X)
        
        return out

class interp(nn.Module):
    def __init__(self,size) -> None:
        super().__init__()
        self.size=size
    def forward(self,x):
        return F.interpolate(x,size=self.size, mode='bilinear',align_corners=False)

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

B_CODING_VARIANTS = 2

def closest_k(all_poses,i):
    cr_rotation = all_poses[i,:3,:3].T
    rotation_distance = []
    for rotation in all_poses[:,:3,:3]:
        rot = rotation @ cr_rotation
        rotation_distance.append(-np.trace(rot))
    positions = all_poses[:,:3,3]
    dis = np.sum((positions-all_poses[i,:3,3].T)**2,axis=-1)
    pair_idx = np.argsort(dis+10*np.array(rotation_distance))
    return pair_idx

from .utils import *
class LLFFBerrRllff_G(Dataset):
    def __init__(self,
                root_dir, split='train',
                img_wh=(504, 378),
                spheric_poses=False,
                val_num=1, perturbation=[],
                sample_wh=64,img2patch=None,train_list=None):
        """
        spheric_poses: whether the images are taken in a spheric inward-facing manner
                       default: False (forward-facing)
        val_num: number of val images (used for multigpu training, validate same image for all gpus)
        """
        print("Using LLFF Patched dataset.")
        self.root_dir = root_dir
        # self.depth_list = (torch.from_numpy(load_colmap_depth(self.root_dir,img_wh)).float()).unsqueeze(1)
        depth_ = (torch.from_numpy(load_colmap_depth('./data/nerf_llff_data/trex/',img_wh)).float()).unsqueeze(1)
        self.depth_list = torch.cat([depth_[:7],depth_[1:7]])
        # self.depth_list /= torch.max(self.depth_list,dim=0)[0]
        # factor = torch.max(self.depth_list.reshape(len(self.depth_list),-1),dim=1)[0]
        self.split = split
        self.sample_wh = sample_wh
        self.img_wh = img_wh
        self.spheric_poses = spheric_poses
        self.val_num = max(1, val_num) # at least 1
        self.perturbation = perturbation
        self.train_list = train_list
        if split=='train':
            print(f'add {self.perturbation} perturbation!')
        self.define_transforms()
        if img2patch is None:
            self.img2patch = ImgToPatch(FlexGridRaySampler(sample_wh,img_wh=self.img_wh), sample_wh)
        else:
            self.img2patch = img2patch

        # enc_path = '/home/CORP.PKUSC.ORG/zcxpku/errnet_160_00485320.pt'
        enc_path = './stage1/errnet_060_00463920.pt'
        # enc_path = './stage1/encoder_B.ckpt'
        self.errnet_B = True
        if self.errnet_B:
            in_channels=3
            self.vgg = Vgg19(requires_grad=False)
            in_channels += 1472 
            self.enc_B = DRNet(in_channels,3,256,13, norm=None, res_scale=0.1, se_reduction=8, bottom_kernel_size=1, pyramid=True)
            state_dict = torch.load(enc_path,map_location='cuda:0')['icnn']
            state_dict = {p:state_dict[p] for p in set(state_dict)&set(self.enc_B.state_dict())}
            self.enc_B.load_state_dict(state_dict)
        else:
            self.enc_B = UNet(3,3)
            state_dict = torch.load(enc_path,map_location='cpu')['icnn']
            self.enc_B.load_state_dict(state_dict)
        for p in self.enc_B.parameters():
            p.requires_grad = False
        # import ipdb;ipdb.set_trace()
        self.upsample = interp(size=(self.img_wh[1],self.img_wh[0]))
        # self.upsample = nn.UpsamplingBilinear2d(size=(self.img_wh[1],self.img_wh[0]))
        self.read_meta()
        self.white_back = False


    def read_meta(self):
        poses_bounds = np.load(os.path.join(self.root_dir,
                                            'poses_bounds.npy')) # (N_images, 17)
        # self.image_paths = sorted(os.listdir(os.path.join(self.root_dir, 'images_4/')))
        self.image_paths = sorted(os.listdir(os.path.join(self.root_dir, 'images_4/')),key=lambda x:int(x[:-4]))
        print(self.image_paths)
        if self.train_list is None:
            self.train_list = [int(filename.split('.')[0]) for filename in self.image_paths]
            # self.train_list = [0,1,2,3,4,5,6]
        

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
        val_idx = 1#np.argmin(distances_from_center) # choose val image as the closest to
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

        # self.train_list = [2,4,6,8,10, 12,14,16,18,22,24,26,28,30,32]
        # self.train_list = [2,13,28,35,44,51]
        # self.train_list = [0,4,8,12,26,32,38,42,49,52]
        # self.train_list = list(range(20))
        self.err_output = []
        if self.split == 'train' or self.split == 'test_train':
            # create buffer of all rays and rgb data.
            # use first N_images-1 to train, the LAST is val

            # self.all_rays = []
            # self.all_rgbs = []
            # self.all_originals = []
            self.all_rays_o = []
            self.all_rays_d = []
            self.near_far = []
            self.all_rgbs = [] # {id: tensor of (3,H,W)}
            self.all_originals = []
            self.code_B = []
            # del(self.image_paths[val_idx]) # exclude the val image
            del(self.train_list[val_idx])
            for i, image_path in enumerate(self.image_paths):
                if i not in self.train_list: # exclude the val image
                    continue
                reflection_path = os.path.join(self.root_dir, 'images_4/')+image_path
                background_path = os.path.join(self.root_dir, 'gt/')+image_path[:-4]+'.png'
                c2w = torch.FloatTensor(self.poses[i] if i<=6 else self.poses[i-6])

                img_ori = Image.open(background_path).convert('RGB')
                img = Image.open(reflection_path).convert('RGB')
                # TODO
                ### B
                inp = self.transform(img).unsqueeze(0)
                if self.errnet_B:
                    hypercolumn = self.vgg(inp)
                    _, C, H, W = inp.shape
                    hypercolumn = [F.interpolate(feature.detach(), size=(H, W), mode='bilinear', align_corners=False) for feature in hypercolumn]
                    input_i = [inp]
                    input_i.extend(hypercolumn)
                    input_i = torch.cat(input_i, dim=1)
                else:
                    input_i = inp

                if self.split == 'train':
                    if self.errnet_B:
                        global_B,local_B, _ = self.enc_B(input_i)
                        self.code_B += [torch.cat((self.upsample(global_B)[0],self.upsample(local_B)[0]),0)]
                    else:
                        B_encodeds = self.enc_B(input_i)
                        temp = []
                        for B_encoded in B_encodeds:
                            temp += [self.upsample(B_encoded)[0]]
                        temp = torch.cat(temp,0)
                        # warped_src_feat, _ = homo_warp(torch.cat(temp,0),proj_mat[:3], self.depth_nearest[ref_idx],pad=0)
                        self.code_B += [temp]#[warped_src_feat]
                else:
                    if self.errnet_B:
                        global_B,local_B, err_output = self.enc_B(input_i)
                        self.code_B += [torch.cat((self.upsample(global_B)[0],self.upsample(local_B)[0]),0)]
                        self.err_output += [err_output[0]]
                    else:
                        B_encodeds = self.enc_B(input_i)
                        temp = []
                        for B_encoded in B_encodeds:
                            temp += [self.upsample(B_encoded)[0]]
                        temp = torch.cat(temp,0)
                        # warped_src_feat, _ = homo_warp(torch.cat(temp,0),proj_mat[:3], self.depth_nearest[ref_idx],pad=0)
                        self.code_B += temp#[warped_src_feat]
                ### R
                img_ori = img_ori.resize(self.img_wh, Image.LANCZOS)
                img_ori = self.transform(img_ori)
                # img = self.transform(img) # (3, h, w)
                # img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGB
                if i<7 and i>0:
                    rgb_idx = i+6
                else:
                    rgb_idx = i
                img = Image.open(os.path.join(self.root_dir, 'images_4/')+self.image_paths[rgb_idx]).convert('RGB')
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img)
                self.all_rgbs += [img] # Use the real image, not the reference image!!!
                self.all_originals += [img_ori]
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
                rays_o = rays_o.t().view(3,self.img_wh[1],self.img_wh[0])
                rays_d = rays_d.t().view(3,self.img_wh[1],self.img_wh[0])
                self.all_rays_o += [rays_o]
                self.all_rays_d += [rays_d]
                self.near_far += [torch.Tensor([near,far])]
            # self.code_B = self.code_B[0]#torch.mean(torch.stack(self.code_B), 0)
            # self.code_B = self.code_B[0]
            self.code_B = torch.stack(self.code_B)
            if self.split == 'test_train':
                self.err_output = torch.stack(self.err_output)
            # self.all_originals = torch.cat(self.all_originals, 0)
            # self.all_rgbs = torch.cat(self.all_rgbs, 0) # ((N_images-1)*h*w, 3)
            self.N_pics = len(self.all_rays_o)

        elif self.split == 'val':
            print('val image is', self.image_paths[val_idx])
            self.c2w_val = self.poses[val_idx]
            self.image_path_val = os.path.join(self.root_dir, 'images_4/')+self.image_paths[val_idx]
            self.val_img = Image.open(self.image_path_val).convert('RGB')
            temp_img_ori = Image.open(os.path.join(self.root_dir, 'images_4/')+self.image_paths[val_idx]) # ORIGINAL
            # temp_img = self.transform(add_perturbation(temp_img,'llff_im',1)) # self.val_img
            temp_img = np.array(temp_img_ori)[...,:3]
            # temp_img[temp_img==0]+=np.array(self.val_img)[temp_img==0]
            # temp_img = np.array(add_perturbation(Image.fromarray(temp_img_ori),'llff_im',0))
            # temp_img_ori = np.array(temp_img_ori)
            # temp_img[temp_img_ori==0]=0
            Image.fromarray(temp_img).save(f'debug2/val_ref.png')
            self.val_img = self.val_img.resize(self.img_wh, Image.LANCZOS)
            self.val_img_ori = Image.open(os.path.join(self.root_dir, 'gt/')+self.image_paths[val_idx][:-4]+'.png').convert('RGB')
            self.val_img_ori = self.val_img_ori.resize(self.img_wh, Image.LANCZOS)
            self.val_img.save(f'debug2/val_img.png')
            temp_img = self.transform(temp_img).unsqueeze(0)
            inp=temp_img 
            # self.code_B = self.enc_B(self.transform(self.val_img).unsqueeze(0))[0]
            ### B
            if self.errnet_B:
                if self.vgg is not None:
                    hypercolumn = self.vgg(inp)
                    _, C, H, W = inp.shape
                    hypercolumn = [F.interpolate(feature.detach(), size=(H, W), mode='bilinear', align_corners=False) for feature in hypercolumn]
                    input_i = [inp]
                    input_i.extend(hypercolumn)
                    input_i = torch.cat(input_i, dim=1)
                global_B, local_B, err_output = self.enc_B(input_i)
                self.err_output += [err_output[0]]
                self.code_B = torch.cat((self.upsample(global_B)[0],self.upsample(local_B)[0]),0)
            else:
                B_encodeds = self.enc_B(inp)
                temp = []
                for B_encoded in B_encodeds:
                    temp += [self.upsample(B_encoded)[0]]
                temp = torch.cat(temp,0)
                self.code_B = temp

            self.val_img = self.transform(self.val_img) # (3, h, w)
            self.val_img_ori = self.transform(self.val_img_ori)
            self.val_img = self.val_img.view(3, -1).permute(1, 0)
            self.val_img_ori = self.val_img_ori.view(3, -1).permute(1,0)

        if self.split.startswith('test'): # for testing, create a parametric rendering path
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
            return (self.img2patch.len+20)*(self.N_pics)
        if self.split == 'val':  
            return self.val_num
        if self.split == 'test_train':
            return self.N_pics
        return len(self.poses_test)

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            idx_pic = idx % self.N_pics
            rays_o = self.all_rays_o[idx_pic]
            rays_d = self.all_rays_d[idx_pic]
            rgb = self.all_rgbs[idx_pic]
            original=self.all_originals[idx_pic]
            rgb,rays_o,rays_d,original,random_patch,code_B,depth_gt = \
                self.img2patch([
                    rgb, rays_o, rays_d, original, self.all_originals[(idx+1) % self.N_pics],
                    self.code_B[idx_pic], self.depth_list[self.train_list[idx_pic]]
                ],idx//self.N_pics) # (h*w,3)
            random_patch = random_patch.t().view(3,self.sample_wh,self.sample_wh)

            rays = torch.cat([rays_o,rays_d,self.near_far[idx_pic]*torch.ones((len(rays_o),2))],1)
            sample = {'rays': rays,
                      'ts': idx_pic * torch.ones((len(rays_o),1), dtype=torch.long),
                      'rgbs': rgb,
                      'original': original,
                      'random_patch': random_patch,
                      'local': idx//self.N_pics>=0 and idx//self.N_pics<self.img2patch.len,
                      'code_B': code_B,
                      'depth_gt':depth_gt
                        }

        else:
            idx_pic = idx
            if self.split == 'val':
                c2w = torch.FloatTensor(self.c2w_val)
            elif self.split == 'test_train':
                c2w = torch.FloatTensor(self.poses_test[idx])
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
                      'ts': idx_pic * torch.ones((len(rays_o),1), dtype=torch.long),
                      'features': torch.Tensor(),
                      'c2w': c2w}

            if self.split == 'val':
                sample['rgbs'] = self.val_img
                sample['original'] = self.val_img_ori
                sample['code_B'] = self.code_B.view(self.code_B.shape[0], -1).permute(1, 0)
                if self.errnet_B:
                    sample['err_output']= self.err_output[idx]
            elif self.split == 'test_train':
                sample['rgbs'] = self.all_rgbs[idx].flatten(1,2).t()
                sample['original'] = self.all_originals[idx].flatten(1,2).t()
                sample['code_B'] = self.code_B[idx].flatten(1,2).t()
                sample['depth_gt'] = self.depth_list[idx].flatten(1,2).t()
                if self.errnet_B:
                    sample['err_output']= self.err_output[idx]

        return sample
