'''
Date: 2021-10-22 8:22:15 pm
Author: dihuangdh
Descriptions: 
-----
LastEditTime: 2023-01-08 6:07:27 pm
LastEditors: dihuangdh
'''

import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
import json
import random as r
from glob import glob
from icecream import ic
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
from tqdm import tqdm
from .colmap_utils import read_cameras_binary, read_images_binary, read_points3d_binary


def read_json(jsonname):
    with open(jsonname) as f:
        data = json.load(f)
    return data


# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf
        
        self.data_dir = conf.get_string('data_dir')
        camdata = read_cameras_binary(os.path.join(self.data_dir, 'colmap', 'cameras.bin'))
        imdata = read_images_binary(os.path.join(self.data_dir, 'colmap', 'images.bin'))
        pts3d = read_points3d_binary(os.path.join(self.data_dir, 'colmap', 'points3D.bin'))
        ignores, ign_idxs = [], []
        if os.path.exists(os.path.join(self.data_dir, 'colmap', 'ignore.txt')):
            ignores = np.loadtxt(os.path.join(self.data_dir, 'colmap', 'ignore.txt'), dtype=int)
            ignores = ignores.reshape(-1).tolist()
        ignores = [str(ignore).zfill(6) for ignore in ignores]
        for k in range(len(imdata)):
            if imdata[k+1].name.rstrip('.jpg') in ignores:
                ign_idxs.append(k)
        print(f'Remove: {ign_idxs}')

        # read origin and radius
        xyz_world = np.array([pts3d[p_id].xyz for p_id in pts3d])
        origin = xyz_world.mean(0)  # (3,)
        radius = np.percentile(np.sqrt(np.sum((xyz_world-origin), axis=1)**2), 99.9)
        self.origin = origin
        self.radius = radius * 2.5  # 2 vs 2.5

        # read successfully reconstructed images and ignore others
        self.images_lis = [os.path.join(self.data_dir, 'colmap', 'images_png', imdata[k+1].name.replace('.jpg', '.png')) for k in range(len(imdata))]
        self.sem_lis = [os.path.join(self.data_dir, 'colmap', 'semantics', imdata[k+1].name.replace('.jpg', '.png')) for k in range(len(imdata))]
        self.n_images = len(self.images_lis)
        # Remove ignored images
        self.n_images = self.n_images - len(ign_idxs)
        self.images_lis = np.array(self.images_lis)
        self.sem_lis = np.array(self.sem_lis)
        self.images_lis = np.delete(self.images_lis, ign_idxs, 0)
        self.sem_lis = np.delete(self.sem_lis, ign_idxs, 0)

        img0 = cv.imread(self.images_lis[0], cv.IMREAD_UNCHANGED).astype(np.float32)
        self.images_np = np.zeros((self.n_images, img0.shape[0], img0.shape[1], 4), dtype=np.float32)  # pre-allocation 
        self.semantics_np = np.zeros((self.n_images, img0.shape[0], img0.shape[1]), dtype=np.int8)  # pre-allocation 
        for idx, im_name in enumerate(tqdm(self.images_lis, desc='reading png images')):
            img = cv.imread(im_name, cv.IMREAD_UNCHANGED).astype(np.float32)
            self.images_np[idx] = img / 256.0
        self.masks_np = self.images_np[..., -1:]
        self.images_np = self.images_np[..., :3]
        for idx, sem_name in enumerate(tqdm(self.sem_lis, desc='reading semantics')):
            semantic_ = cv.imread(sem_name)
            semantic = np.zeros((semantic_.shape[0], semantic_.shape[1])).astype(np.uint8)  # void ~ 0
            semantic[semantic_[..., 0]==255] = 1  # hand ~ 1
            semantic[semantic_[..., 1]==255] = 2  # object ~ 2
            # import ipdb; ipdb.set_trace()  # np.count_nonzero(semantic_[..., 0]==255)
            self.semantics_np[idx] = semantic

        self.scale_mats_np = []
        self.scale_mats_np = [np.identity(4) for idx in range(self.n_images)]

        self.intrinsics_all = []
        self.pose_all = []

        K = np.array([
            [camdata[1].params[0], 0, camdata[1].params[2]],
            [0, camdata[1].params[1], camdata[1].params[3]],
            [0, 0, 1]
        ])
        for k in range(len(imdata)):
            im = imdata[k+1]

            # right
            R = im.qvec2rotmat()
            t = im.tvec.reshape(3, 1)

            Rt = np.concatenate([R, t], 1)
            P = K @ Rt  # (3, 4)
            intrinsics, pose = load_K_Rt_from_P(None, P)
            
            self.intrinsics_all.append(intrinsics.astype(np.float32))
            self.pose_all.append(pose.astype(np.float32))
        self.intrinsics_all = np.array(self.intrinsics_all, dtype=np.float32)
        self.pose_all = np.array(self.pose_all, dtype=np.float32)
        
        # Remove by ign_idxs
        self.intrinsics_all = np.delete(self.intrinsics_all, ign_idxs, 0)
        self.pose_all = np.delete(self.pose_all, ign_idxs, 0)

        self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()  # [n_images, H, W, 3]
        del self.images_np
        self.semantics = torch.from_numpy(self.semantics_np).cpu()[..., None]  # [n_images, H, W, 1]
        del self.semantics_np
        self.masks  = torch.from_numpy(self.masks_np.astype(np.float32)).cpu()   # [n_images, H, W, 1]
        del self.masks_np
        self.intrinsics_all = torch.from_numpy(self.intrinsics_all).to(self.device)   # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.from_numpy(self.pose_all).to(self.device)  # [n_images, 4, 4]
        self.H, self.W = self.images.shape[1], self.images.shape[2]
        self.image_pixels = self.H * self.W

        self.background_samplers = []
        self.hand_samplers = []
        self.object_samplers = []
        for idx in range(self.n_images):
            self.background_samplers.append([(self.semantics[idx] == 0).nonzero(as_tuple=True)[0].type(torch.int16), 
                (self.semantics[idx] == 0).nonzero(as_tuple=True)[1].type(torch.int16)])
            self.hand_samplers.append([(self.semantics[idx] == 1).nonzero(as_tuple=True)[0].type(torch.int16),
                (self.semantics[idx] == 1).nonzero(as_tuple=True)[1].type(torch.int16)])
            self.object_samplers.append([(self.semantics[idx] == 2).nonzero(as_tuple=True)[0].type(torch.int16),
                (self.semantics[idx] == 2).nonzero(as_tuple=True)[1].type(torch.int16)])
            # import ipdb; ipdb.set_trace()

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0]).reshape((4, 1))
        object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0]).reshape((4, 1))
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]

        print('Load data: End')
    
    @staticmethod
    def random_sum_to(n, num_terms = None):
        num_terms = (num_terms or r.randint(2, n)) - 1
        a = r.sample(range(1, n), num_terms) + [0, n]
        list.sort(a)
        return [a[i+1] - a[i] for i in range(len(a) - 1)]
    
    @staticmethod
    def equal_sum_to(n, num_terms=None):
        a = int(n / num_terms)
        b = n - a * num_terms
        a = [a] * num_terms
        a = np.array(a)
        a[:b] += 1
        return a

    def gen_rays_at(self, img_idx, resolution_level=1, pose_all=None):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_random_rays_at(self, img_idx, batch_size, pose_all=None):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])

        color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 1
        semantic = self.semantics[img_idx][(pixels_y, pixels_x)]  # batch_size, 1
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze(-1) # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        rays_v = torch.matmul(pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze(-1)  # batch_size, 3
        rays_o = pose_all[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3
        return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1], semantic[:, :1]], dim=-1).cuda()    # batch_size, 10

    def gen_split_rays_at(self, img_idx, bgnum, handnum, objectnum, pose_all=None):
        """
        Generate random rays at world space from one camera.
        """
        background_sampler = self.background_samplers[img_idx]
        hand_sampler = self.hand_samplers[img_idx]
        object_sampler = self.object_samplers[img_idx]
        background_random = torch.randint(low=0, high=background_sampler[0].shape[0], size=[bgnum])
        hand_random = torch.randint(low=0, high=hand_sampler[0].shape[0], size=[handnum])
        object_random = torch.randint(low=0, high=object_sampler[0].shape[0], size=[objectnum])
        
        background_y = background_sampler[0][background_random]
        background_x = background_sampler[1][background_random]
        hand_y = hand_sampler[0][hand_random]
        hand_x = hand_sampler[1][hand_random]
        object_y = object_sampler[0][object_random]
        object_x = object_sampler[1][object_random]

        pixels_x = torch.cat([background_x, hand_x, object_x]).to(self.device).long()
        pixels_y = torch.cat([background_y, hand_y, object_y]).to(self.device).long()

        color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 1
        semantic = self.semantics[img_idx][(pixels_y, pixels_x)]  # batch_size, 1
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze(-1) # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        rays_v = torch.matmul(pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze(-1)  # batch_size, 3
        rays_o = pose_all[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3
        return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1], semantic[:, :1]], dim=-1).cuda()    # batch_size, 10

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1, pose_all=None):
        """
        Interpolate pose between two cameras.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        trans = pose_all[idx_0, :3, 3] * (1.0 - ratio) + pose_all[idx_1, :3, 3] * ratio
        pose_0 = pose_all[idx_0].detach().cpu().numpy()
        pose_1 = pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3]).cuda()
        trans = torch.from_numpy(pose[:3, 3]).cuda()
        rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_lis[idx])
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)
    
    def mask_at(self, idx, resolution_level):
        img = cv.imread(self.images_lis[idx], cv.IMREAD_UNCHANGED)
        mask = img[..., -1:]
        return (cv.resize(mask, (self.W // resolution_level, self.H // resolution_level), interpolation=cv.INTER_NEAREST)).clip(0, 255)

    def sem_at(self, idx, resolution_level):
        sem = cv.imread(self.sem_lis[idx])
        return (cv.resize(sem, (self.W // resolution_level, self.H // resolution_level), interpolation=cv.INTER_NEAREST)).clip(0, 255)