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
from .utils import *

B_CODING_VARIANTS = 4
code_selection = [[a,b] for a in range(B_CODING_VARIANTS-1) for b in range(B_CODING_VARIANTS-1) if a<b]

def closest_k(all_poses,i,exclude=set()):
    cr_rotation = all_poses[i,:3,:3].T
    rotation_distance = []
    for rotation in all_poses[:,:3,:3]:
        rot = rotation @ cr_rotation
        rotation_distance.append(-np.trace(rot))
    positions = all_poses[:,:3,3]
    dis = np.sum((positions-all_poses[i,:3,3].T)**2,axis=-1)
    # index_distance = np.abs(np.arange(len(all_poses))-i)
    pair_idx = list(np.argsort(dis+1*np.array(rotation_distance)))
    pair_idx = [pidx for pidx in pair_idx if pidx not in exclude]
    return np.array(pair_idx)


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        self.vgg_pretrained_features = models.vgg19(pretrained=True).features
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X, indices=None):
        if indices is None:
            indices = [2, 7, 12, 21, 30]
        out = []
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
        return F.interpolate(x,size=self.size, mode='nearest')


def normalize(v):
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

class LLFF_BWarp(Dataset):
    def __init__(self,
                root_dir, split='train',
                img_wh=(504, 378),
                spheric_poses=False,
                val_num=1, perturbation=[],
                sample_wh=64,img2patch=None,**kwargs):
        """
        spheric_poses: whether the images are taken in a spheric inward-facing manner
                       default: False (forward-facing)
        val_num: number of val images (used for multigpu training, validate same image for all gpus)
        """
        print("Using LLFF Patched dataset.")
        self.root_dir = root_dir
        # self.depth_list = (torch.from_numpy(load_colmap_depth(self.root_dir,img_wh,factor=4,use_raw=True)).float()).unsqueeze(1)
        # self.depth_nearest = (torch.from_numpy(load_colmap_depth(self.root_dir,img_wh,factor=4,mode='blend',use_raw=True)).float()).unsqueeze(1)
        self.split = split
        self.sample_wh = sample_wh
        self.img_wh = img_wh
        self.spheric_poses = spheric_poses
        self.val_num = max(1, val_num) # at least 1
        self.perturbation = perturbation
        if split=='train':
            print(f'add {self.perturbation} perturbation!')
        if split=='test':
            self.test_img_w = kwargs['test_img_w']
            self.test_img_h = kwargs['test_img_h']
        self.define_transforms()
        if img2patch is None:
            self.img2patch = ImgToPatch(FlexGridRaySampler(sample_wh,img_wh=self.img_wh,scale_anneal=3e-5), sample_wh)
        else:
            self.img2patch = img2patch
        if 'enc_path' not in kwargs.keys():
            enc_path = './ckpt_ERRNet/errnet_060_00463920.pt'
        else:
            enc_path = kwargs['enc_path']
        self.errnet_B = True
        if self.errnet_B:
            in_channels=3
            self.vgg = Vgg19(requires_grad=False)
            in_channels += 1472 
            self.enc_B = DRNet(in_channels,3,256,13, norm=None, res_scale=0.1, se_reduction=8, bottom_kernel_size=1, pyramid=True)
            state_dict = torch.load(enc_path,map_location='cpu')['icnn']
            state_dict = {p:state_dict[p] for p in set(state_dict)&set(self.enc_B.state_dict())}
            self.enc_B.load_state_dict(state_dict)
        else:
            self.enc_B = Vgg19(requires_grad=False)#UNet(3,3)
            # import ipdb;ipdb.set_trace()
            # state_dict = torch.load(enc_path,map_location='cpu')['icnn']
            # state_dict = {p:state_dict[p] for p in set(state_dict)&set(self.enc_B.state_dict())}
            # self.enc_B.load_state_dict(state_dict)
            # load_ckpt(self.enc_B,enc_path,model_name='icnn')
        # self.enc_R = UNet(3,3)
        for p in self.enc_B.parameters():
            p.requires_grad = False
        self.upsample = interp(size=(self.img_wh[1],self.img_wh[0]))
        self.read_meta()
        self.white_back = False


    def read_meta(self):
        poses_bounds = np.load(os.path.join(self.root_dir,
                                            'poses_bounds.npy')) # (N_images, 17)
        self.image_paths = sorted(glob.glob(os.path.join(self.root_dir, 'images/*')))
        if os.path.exists(os.path.join(self.root_dir,'gt/')):
            self.gt_paths = sorted(glob.glob(os.path.join(self.root_dir, 'gt/*')))
        else:
            self.gt_paths = self.image_paths

        if os.path.exists(os.path.join(self.root_dir,'edge/')):
            self.use_edge = True
            print("USING EDGE")
        else:
            self.use_edge = False
        self.edge_paths = sorted(glob.glob(os.path.join(self.root_dir, 'edge/*')))
                        # load full resolution image then resize
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
        self.K[1, 2] = self.img_wh[1]/2 # intrinsic!

        # Step 2: correct poses
        # Original poses has rotation in form "down right back", change to "right up back"
        # See https://github.com/bmild/nerf/issues/34
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
                # (N_images, 3, 4) exclude H, W, focal
        self.poses, self.pose_avg = center_poses(poses)
        distances_from_center = np.linalg.norm(self.poses[..., 3], axis=1)
        val_idx = np.argmin(distances_from_center) # choose val image as the closest to
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
        self.K = torch.Tensor(self.K)

        # self.train_list = np.array([2,13,28,35,40,47])
        self.err_output = []
        self.proj_mats = []
        self.train_list = list(range(len(self.poses)))
        if self.split[-5:] == 'train':
            self.train_list.remove(val_idx)
        self.train_list = self.train_list[::3][:6]

        print(f"LEN: {len(self.train_list)}")

        if self.split == 'train' or self.split == 'test_train':
            # create buffer of all rays and rgb data.
            self.all_rays_o = []
            self.all_rays_d = []
            self.near_far = []
            self.all_rgbs = [] # {id: tensor of (3,H,W)}
            self.all_originals = []
            self.code_B = []
            self.all_edges = []
            # self.train_list.remove(val_idx) # exclude the val image
            # self.poses = self.poses[self.train_list]
            # self.image_paths = [self.image_paths[x] for x in self.train_list]
            for i, image_path in enumerate(self.image_paths):
                if i not in self.train_list:
                    continue
                c2w = torch.FloatTensor(self.poses[i])
                img_ori = Image.open(self.gt_paths[i]).convert('RGB')
                if self.use_edge:
                    edge = Image.open(self.edge_paths[i]).convert('RGB')
                    edge = edge.resize(self.img_wh, Image.BICUBIC)
                    edge = self.transform(edge)

                # img = add_perturbation(img_ori, self.perturbation, i, output_ori=False)
                img = Image.open(image_path).convert('RGB')
                img = img.resize(self.img_wh, Image.BICUBIC)
                
                pair_idx = closest_k(self.poses,i,exclude={val_idx})[:B_CODING_VARIANTS]
                for ref_idx in pair_idx:
                    tempc2w=torch.eye(4)
                    tempc2w[:3]=torch.FloatTensor(self.poses[ref_idx])
                    w2c = torch.inverse(tempc2w)
                    proj_mat_l = torch.eye(4)
                    proj_mat_l[:3,:4]=self.K @ w2c[:3,:4]
                    if (ref_idx == i):
                        ref_proj_inv = torch.inverse(proj_mat_l)
                        proj_mat = torch.eye(4) # reference image
                        continue # drop i
                    else:
                        proj_mat = proj_mat_l @ ref_proj_inv
                    self.proj_mats.append(proj_mat)

                    ref_img = Image.open(self.image_paths[ref_idx]).convert('RGB')
                    # ref_img = add_perturbation(ref_img, self.perturbation, ref_idx ,output_ori=False)
                    ref_img = ref_img.resize(self.img_wh, Image.BICUBIC)
                    inp = self.transform(ref_img).unsqueeze(0)
                    if self.errnet_B:
                        hypercolumn = self.vgg(inp)
                        _, C, H, W = inp.shape
                        hypercolumn = [F.interpolate(feature.detach(), size=(H, W), mode='bilinear', align_corners=False) for feature in hypercolumn]
                        input_i = [inp]
                        input_i.extend(hypercolumn)
                        input_i = torch.cat(input_i, dim=1)
                    else:
                        input_i=inp

                    ### B
                    if self.split == 'train':
                        if self.errnet_B:
                            global_B,local_B, _ = self.enc_B(input_i)
                            # self.code_B += [torch.cat((self.upsample(global_B)[0],self.upsample(local_B)[0]),0)]
                            self.code_B += [self.upsample(local_B)[0]] # GLOBAL ONLY
                        else:
                            B_encodeds = self.enc_B(input_i)
                            temp = []
                            for B_encoded in B_encodeds:
                                temp += [self.upsample(B_encoded)[0]]
                            temp = torch.cat(temp,0)
                            # warped_src_feat, _ = homo_warp(temp,proj_mat[:3], self.depth_nearest[ref_idx],pad=0)
                            # self.code_B += [warped_src_feat]
                            self.code_B += [temp]
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
                            # warped_src_feat, _ = homo_warp(temp, proj_mat[:3], self.depth_nearest[ref_idx],pad=0)
                            self.code_B += [temp]
                
                # img = img.resize(self.img_wh, Image.LANCZOS)
                img_ori = img_ori.resize(self.img_wh, Image.LANCZOS)
                img_ori = self.transform(img_ori)
                img = self.transform(img) # (3, h, w)
                # img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGB
                self.all_rgbs += [img]
                if self.use_edge:
                    self.all_edges += [edge]
                self.all_originals += [img_ori] #!!!!!
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

            self.code_B = torch.stack(self.code_B)
            if self.split == 'test_train' and self.errnet_B:
                self.err_output = torch.stack(self.err_output)
            # self.all_originals = torch.cat(self.all_originals, 0)
            # self.all_rgbs = torch.cat(self.all_rgbs, 0) # ((N_images-1)*h*w, 3)
            self.N_pics = len(self.all_rays_o)
        elif self.split == 'val':
            print(f'val image is No.{val_idx}', self.image_paths[val_idx])
            self.code_B = []
            self.c2w_val = self.poses[val_idx]
            self.val_img = Image.open(self.image_paths[val_idx]).convert('RGB')
            self.val_gt = Image.open(self.gt_paths[val_idx]).convert('RGB')
            self.val_img = self.val_img.resize(self.img_wh, Image.BICUBIC)
            self.val_gt = self.val_gt.resize(self.img_wh, Image.BICUBIC)
            self.val_img.save(f'debug2/val_img.png')
            self.val_img = self.transform(self.val_img)
            self.val_gt = self.transform(self.val_gt)
            self.val_img = self.val_img.flatten(1,2).t()
            self.val_gt = self.val_gt.flatten(1,2).t()
            pair_idx = closest_k(self.poses,val_idx)
            # ref_idx = pair_idx[1] # 22
            for ref_idx in [pair_idx[p] for p in range(1,3)]:
                temp_img = Image.open(self.image_paths[ref_idx]).convert('RGB')
                temp_img.save('debug2/ref_img.png')
                temp_img = temp_img.resize(self.img_wh, Image.LANCZOS)
                # temp_img_ori = np.array(temp_img_ori)
                # Image.fromarray(temp_img).save(f'debug2/val_ref.png')
                temp_img = self.transform(temp_img).unsqueeze(0)
                inp=temp_img
                tempc2w=torch.eye(4)
                tempc2w[:3]=torch.FloatTensor(self.c2w_val)
                w2c=torch.inverse(tempc2w)
                proj_mat_l = torch.eye(4)
                proj_mat_l[:3,:4]=self.K @ w2c[:3,:4]
                ref_proj_inv = torch.inverse(proj_mat_l)

                proj_mat_l = torch.eye(4)
                tempc2w=torch.eye(4)
                tempc2w[:3]=torch.FloatTensor(self.poses[ref_idx])
                w2c=torch.inverse(tempc2w)
                proj_mat_l[:3,:4] = self.K @ w2c[:3,:4]
                proj_mat = proj_mat_l @ ref_proj_inv
                self.proj_mats.append(proj_mat)
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
                    # self.code_B += [torch.cat((self.upsample(global_B)[0],self.upsample(local_B)[0]),0)]
                    self.code_B += [self.upsample(local_B)[0]] # GLOBAL ONLY
                else:
                    B_encodeds = self.enc_B(inp)
                    temp = []
                    for B_encoded in B_encodeds:
                        temp += [self.upsample(B_encoded)[0]]
                    # warped_src_feat, _ = homo_warp(torch.cat(temp,0),proj_mat[:3], self.depth_nearest[ref_idx],pad=0)
                    # self.code_B = warped_src_feat
                    self.code_B += [torch.cat(temp,0)]
            self.code_B = torch.stack(self.code_B)
            self.proj_mats=torch.stack(self.proj_mats)

        else: # TEST
            self.code_B = []
            for i, image_path in enumerate(self.image_paths):
                if i not in self.train_list:
                    continue
                c2w = torch.FloatTensor(self.poses[i])
                img_ori = Image.open(image_path).convert('RGB')
                img = add_perturbation(img_ori, self.perturbation, i, output_ori=False)
                img = img.resize(self.img_wh, Image.LANCZOS)
                inp = self.transform(img).unsqueeze(0)
                if self.errnet_B:
                    hypercolumn = self.vgg(inp)
                    _, C, H, W = inp.shape
                    hypercolumn = [F.interpolate(feature.detach(), size=(H, W), mode='bilinear', align_corners=False) for feature in hypercolumn]
                    input_i = [inp]
                    input_i.extend(hypercolumn)
                    input_i = torch.cat(input_i, dim=1)
                else:
                    input_i=inp
                if self.errnet_B:
                    global_B,local_B, _ = self.enc_B(input_i)
                    # self.code_B += [torch.cat((self.upsample(global_B)[0],self.upsample(local_B)[0]),0)]
                    self.code_B += [self.upsample(local_B)[0]] # GLOBAL ONLY
                else:
                    B_encodeds = self.enc_B(input_i)
                    temp = []
                    for B_encoded in B_encodeds:
                        temp += [self.upsample(B_encoded)[0]]
                    temp = torch.cat(temp,0)
                    # warped_src_feat, _ = homo_warp(temp,proj_mat[:3], self.depth_nearest[ref_idx],pad=0)
                    # self.code_B += [warped_src_feat]
                    self.code_B += [temp]
            self.poses = self.poses[self.train_list]


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
            return int(self.img2patch.len*1.5)*self.N_pics*len(code_selection)
        if self.split == 'val':  
            return self.val_num
        if self.split == 'test_train':
            return self.N_pics
        return len(self.poses_test)

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            idx_pic = (idx//len(code_selection)) % self.N_pics
            idx_selection = idx % len(code_selection)
            rays_o = self.all_rays_o[idx_pic]
            rays_d = self.all_rays_d[idx_pic]
            rgb = self.all_rgbs[idx_pic]
            patchID = idx//(len(code_selection) * self.N_pics)
            original=self.all_originals[idx_pic] 
            if self.use_edge:
                res, pixels_i = \
                    self.img2patch([
                        rgb, rays_o, rays_d, original,
                        self.all_edges[idx_pic]
                    ],patchID,True)
                rgb,rays_o,rays_d,original,edge = res
            else:
                res, pixels_i = \
                    self.img2patch([
                        rgb, rays_o, rays_d, original,
                    ],patchID,True) # (h*w,3)
                rgb,rays_o,rays_d,original = res

            rays = torch.cat([rays_o,rays_d,self.near_far[idx_pic]*torch.ones((len(rays_o),2))],1)
            sample = {'rays': rays,
                      'ts': idx_pic * torch.ones((len(rays_o),1), dtype=torch.long),
                      'rgbs': rgb,
                      'original': original,
                      'pixels_i': pixels_i,
                      'proj_mat':torch.stack([self.proj_mats[idx_pic*(B_CODING_VARIANTS-1)+ x][:-1] for x in code_selection[idx_selection]]),
                      'full_code_B':torch.stack([self.code_B[idx_pic*(B_CODING_VARIANTS-1)+ x] for x in code_selection[idx_selection]]),
                    }
            if self.use_edge:
                sample['edge'] = edge

        else:
            idx_pic = idx
            if self.split == 'val':
                c2w = torch.FloatTensor(self.c2w_val)
            elif self.split == 'test_train':
                idx_pic = (idx//len(code_selection)) % self.N_pics
                idx_selection = idx % len(code_selection)
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
                              near*torch.ones_like(rays_o[..., :1]),
                              far*torch.ones_like(rays_o[..., :1])],
                              -1) # (h, w, 8)
            sample = {'rays': rays,#.permute(2,0,1),
                      'ts': idx_pic * torch.ones((len(rays_o)), dtype=torch.long),#.permute(2,0,1),
                      'c2w': c2w}
            # h_ind, w_ind = torch.meshgrid([torch.arange(self.img_wh[1]).float(),torch.arange(self.img_wh[0]).float()])
            # coord = torch.cat([(h_ind*2/self.img_wh[1]-1).unsqueeze(2),(2*w_ind/self.img_wh[0]-1).unsqueeze(2)],-1)
            if self.split == 'val':
                sample['rgbs'] = self.val_img
                sample['original'] = self.val_gt
                sample['full_code_B'] = self.code_B
                sample['proj_mat'] = self.proj_mats[:,:-1,:]
                # sample['pixels_i'] = coord.flatten(0,1)
                if self.errnet_B:
                    sample['err_output']= self.err_output[idx]
            elif self.split == 'test_train':
                sample['rgbs'] = self.all_rgbs[idx_pic].flatten(1,2).t()
                sample['original'] = self.all_originals[idx_pic].flatten(1,2).t()
                sample['proj_mat']=torch.stack([self.proj_mats[idx_pic*(B_CODING_VARIANTS-1)+ x][:-1] for x in code_selection[idx_selection]])
                sample['full_code_B']=torch.stack([self.code_B[idx_pic*(B_CODING_VARIANTS-1)+ x] for x in code_selection[idx_selection]])
                # sample['proj_mat'] = self.proj_mats[idx].unsqueeze(0)
                # sample['pixels_i'] = coord.flatten(0,1)
                if self.errnet_B:
                    sample['err_output']= self.err_output[idx]

            else:
                sample = {}
                sample['c2w'] = c2w = torch.FloatTensor(self.poses_test[idx])
                tempc2w=torch.eye(4)
                tempc2w[:3]=c2w
                w2c = torch.inverse(tempc2w)
                proj_mat_l = torch.eye(4)
                proj_mat_l[:3,:4]=self.K @ w2c[:3,:4]
                ref_proj_inv = torch.inverse(proj_mat_l) # from target view
                directions = get_ray_directions(self.test_img_h, self.test_img_w, self.K)
                rays_o, rays_d = get_rays(directions, c2w)
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
                rays = torch.cat([rays_o, rays_d,
                                near*torch.ones_like(rays_o[:, :1]),
                                far*torch.ones_like(rays_o[:, :1])],
                                1)
                sample['rays'] = rays
                sample['ts'] = torch.ones(len(rays), dtype=torch.long)
                sample['img_wh'] = torch.LongTensor([self.test_img_w, self.test_img_h])
                dist = np.linalg.norm((self.poses - self.poses_test[idx])[...,3],axis=-1)
                choice = np.argsort(dist)[:2]
                sample['full_code_B'] = torch.stack([self.code_B[x] for x in choice])
                proj_mats = []
                for ref_idx in choice:
                    tempc2w=torch.eye(4)
                    tempc2w[:3]=torch.FloatTensor(self.poses[ref_idx])
                    w2c = torch.inverse(tempc2w)
                    proj_mat_l = torch.eye(4)
                    proj_mat_l[:3,:4]=self.K @ w2c[:3,:4]
                    proj_mats += [proj_mat_l @ ref_proj_inv]
                sample['proj_mat'] = torch.stack(proj_mats)[:,:-1]
        return sample
