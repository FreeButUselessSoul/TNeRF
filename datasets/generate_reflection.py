import torch
from torch.utils.data import Dataset
from torch import nn
import glob
import numpy as np
import os
from PIL import Image, ImageDraw
from torchvision import transforms as T
import cv2
import torch.nn.functional as F
from depth_utils import load_colmap_depth
import shutil

pic_pool = []

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

def add_perturbation(img_ori, perturbation, seed):
    img = img_ori
    if 'llff_im' in perturbation or 'llff_im2' in perturbation:
        w,h = img.size
        img = np.array(img,dtype='float') / 255.
        np.random.seed(seed)
        left = np.random.randint(0, w//8)
        top = np.random.randint(0, h//8)
        
        selected = pic_pool[np.random.choice(len(pic_pool))]
        selected = selected.crop((left,top,left+w,top+h))
        selected = np.array(selected,dtype='float')/255.

        # reflection = np.zeros_like(img)
        # im2[top:top+h//3,left:left+w//3]=selected
        # reflection[top:top+h//2,left:left+w//2] = cv2.GaussianBlur(selected,ksize=None, sigmaX=2)
        reflection = cv2.GaussianBlur(selected,ksize=None, sigmaX=4)
        im2 = img + reflection # img is background
        average = np.mean(im2[im2>1])
        reflection2 = reflection - 0.5*(average-0.9)
        reflection2 = np.clip(reflection2,0,1) # reflection
        mixture = img
        if 'llff_im' in perturbation:
            mixture = (0.3 * reflection2 + 0.7 * img)
        mixture = np.clip(mixture,0,1)
        return Image.fromarray(np.uint8(mixture*255))
    return img



import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tar_dir', type=str,help='directory for background scene')
    parser.add_argument('--ref_dir', type=str,help='directory for reflection scene')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[504, 378],
                        help='resolution (img_w, img_h) of the image')
    parser.add_argument('--perturbation', type=str, default='llff_im',
                        help='type of reflection added on the image')
    parser.add_argument('--train_list', nargs="+", type=int, default=None,#[2,13,28,35,44,51],
                        help='list of serial numbers of images for training')
    parser.add_argument('--out_dir', type=str,default='',help='directory for output')
    args = parser.parse_args()
    if args.out_dir == '':
        out_dir = os.path.join(args.tar_dir,'./synthesize/')
    else:
        out_dir = args.out_dir
    image_paths = sorted(glob.glob(os.path.join(args.tar_dir, 'images/*')))
    dirname = os.path.join(args.ref_dir,'images/')
    os.makedirs(os.path.join(out_dir,'images/'),exist_ok=True)
    os.makedirs(os.path.join(out_dir,'gt/'),exist_ok=True)
    os.makedirs(os.path.join(out_dir,'edge/'),exist_ok=True)
    # reflection init.
    # poses_bounds = np.load(os.path.join(args.tar_dir,'poses_bounds.npy'))
    if args.train_list is None:
        args.train_list = np.arange(len(image_paths))
    # poses_bounds = poses_bounds[args.train_list]
    # np.save(os.path.join(args.tar_dir,'synthesize/poses_bounds.npy'),poses_bounds)
    # print(poses_bounds.shape)
    # poses = poses_bounds[:, :15].reshape(-1, 3, 5) # (N_images, 3, 5)
    # bounds = poses_bounds[:, -2:] # (N_images, 2)

    # Step 1: rescale focal length according to training resolution
    # H, W, focal = poses[0, :, -1] # original intrinsics, same for all images
    # focal *= args.img_wh[0]/W
    # K = np.eye(3)
    # K[0, 0] = K[1, 1] = focal
    # K[0, 2] = args.img_wh[0]/2
    # K[1, 2] = args.img_wh[1]/2

    # Step 2: correct poses
    # poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
            # (N_images, 3, 4) exclude H, W, focal
    # poses, pose_avg = center_poses(poses)
    # distances_from_center = np.linalg.norm(poses[..., 3], axis=1)
    # val_idx = np.argmin(distances_from_center)
    # print(f"val_idx={val_idx}")
    # if (val_idx not in args.train_list):
    #     args.train_list.append(val_idx)
    for pic in os.listdir(dirname):
        temp = Image.open(os.path.join(dirname,pic)).convert('RGB')
        temp = temp.resize((int(args.img_wh[0]*1.3),int(args.img_wh[1]*1.3)), Image.BICUBIC)
        pic_pool.append(temp)
    for i in args.train_list:
        image_path = image_paths[i]
        img_ori = Image.open(image_path).convert('RGB')
        img_ori = img_ori.resize(args.img_wh, Image.BICUBIC)
        # img_ori = img_ori.resize(args.img_wh, Image.LANCZOS)
        img = add_perturbation(img_ori, args.perturbation, i)
        # img = img.resize(args.img_wh, Image.BICUBIC)
        
        img.save(os.path.join(out_dir,'images/%02d.png' % i))
        img_ori.save(os.path.join(out_dir,'gt/%02d.png' % i))
        edge=cv2.Laplacian(np.array(img_ori),cv2.CV_64F)
        edge[np.all(edge<10,-1)]=0.
        edge[np.any(edge>10,-1)]=1.
        edge = Image.fromarray(np.uint8(edge*255))
        edge.save(os.path.join(out_dir,'edge/%02d.png' % i))

    otherfiles=['poses_bounds.npy','database.db']
    for filename in otherfiles:
        shutil.copyfile(os.path.join(args.tar_dir,filename), os.path.join(out_dir,filename))
    shutil.copytree(os.path.join(args.tar_dir,'sparse/'),os.path.join(out_dir,'sparse/'))