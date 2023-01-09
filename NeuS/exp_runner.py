import os
import logging
import argparse
import numpy as np
import cv2 as cv
import trimesh
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from tqdm import tqdm
from pyhocon import ConfigFactory
from models.dataset import Dataset
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF
from models.warp import SE3Field
from models.renderer import NeuSRenderer
import models.camera as camera


class FocalLoss(nn.Module):
    '''Multi-class Focal loss implementation'''
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight)
        return loss
focal_loss = FocalLoss(gamma=1)


class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', is_continue=False):
        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        # self.base_exp_dir = os.path.join(os.getcwd() , self.base_exp_dir)
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.dataset = Dataset(self.conf['dataset'])
        self.iter_step = 0

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.val_all_freq = self.conf.get_int('train.val_all_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.sem_weight = self.conf.get_float('train.sem_weight')
        self.warp_weight = self.conf.get_float('train.warp_weight')
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        # Networks
        params_to_train = []
        self.nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device)
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network'], n_images=self.dataset.n_images, barf=self.conf['model.barf']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network'], n_images=self.dataset.n_images).to(self.device)
        self.warp_network = SE3Field(n_images=self.dataset.n_images, enable=self.conf['model.warp'])
        params_to_train += list(self.nerf_outside.parameters())
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())
        params_to_train += list(self.warp_network.parameters()) 

        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        self.renderer = NeuSRenderer(self.nerf_outside,
                                     self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     self.warp_network,
                                     self.dataset.origin,
                                     self.dataset.radius,
                                     **self.conf['model.neus_renderer'])

        # Load checkpoint
        latest_model_name = None
        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()

    def train(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        image_perm = self.get_image_perm()

        for iter_i in tqdm(range(res_step)):
            if self.conf['model.barf']:
                self.sdf_network.progress.data.fill_(iter_i/res_step)
                self.color_network.progress.data.fill_(iter_i/res_step)
            else:
                self.sdf_network.progress.data.fill_(1.)
                self.color_network.progress.data.fill_(1.)
            self.warp_network.progress.data.fill_(iter_i/res_step)

            pose_refine = camera.lie.se3_to_SE3(self.sdf_network.se3_refine.weight)
            pose_all = camera.pose.compose([pose_refine, self.dataset.pose_all[:, :3, :]])
            
            if self.conf['train.resample']:
                background_factor = self.conf['train.data_init_factor'][0] + (self.conf['train.data_final_factor'][0]-self.conf['train.data_init_factor'][0]) * (iter_i/res_step)
                hand_factor = self.conf['train.data_init_factor'][1] + (self.conf['train.data_final_factor'][1]-self.conf['train.data_init_factor'][1]) * (iter_i/res_step)
                background_snum = int(self.batch_size * background_factor)
                hand_snum = int(self.batch_size * hand_factor)
                object_snum = self.batch_size - background_snum - hand_snum
                background_ridxs = self.dataset.equal_sum_to(background_snum, num_terms=self.dataset.n_images)
                hand_ridxs = self.dataset.equal_sum_to(hand_snum, num_terms=self.dataset.n_images)
                object_ridxs = self.dataset.equal_sum_to(object_snum, num_terms=self.dataset.n_images)

                data = []
                img_idx = []
                for idx, (bgnum, handnum, objectnum) in enumerate(zip(background_ridxs, hand_ridxs, object_ridxs)):
                    data_ = self.dataset.gen_split_rays_at(image_perm[idx], bgnum, handnum, objectnum, pose_all=pose_all)
                    img_idx.append(image_perm[idx].repeat(bgnum + handnum + objectnum))
                    data.append(data_)
                img_idx = torch.cat(img_idx, 0)
                data = torch.cat(data, 0)

            else:
                ridxs = self.dataset.equal_sum_to(self.batch_size, num_terms=self.dataset.n_images)

                data = []
                img_idx = []
                for idx, num in enumerate(ridxs):
                    data_ = self.dataset.gen_random_rays_at(image_perm[idx], num, pose_all=pose_all)
                    img_idx.append(image_perm[idx].repeat(num))
                    data.append(data_)
                img_idx = torch.cat(img_idx, 0)
                data = torch.cat(data, 0)

            rays_o, rays_d, true_rgb, mask, semantic = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10], data[:, 10: 11]

            background_rgb = None
            if self.use_white_bkgd:
                background_rgb = torch.ones([1, 3])

            if self.mask_weight > 0.0:
                mask = (mask > 0.5).float()
            else:
                mask = torch.ones_like(mask)

            mask_sum = mask.sum() + 1e-5
            render_out = self.renderer.render(rays_o, rays_d, 
                                              # near, far,
                                              time=img_idx,
                                              background_rgb=background_rgb,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio())

            color_fine = render_out['color_fine']
            conf_fine = render_out['conf_fine']
            s_val = render_out['s_val']
            cdf_fine = render_out['cdf_fine']
            gradient_error = render_out['gradient_error']
            weights = render_out['weights']
            weight_max = render_out['weight_max']
            weight_sum = render_out['weight_sum']
            warped_div = render_out['warped_div']

            conf_idx = np.argmax(conf_fine.detach().cpu().numpy(), axis=1)
            indicator_hand = (conf_idx == 1)
            indicator_object = (conf_idx == 2)
            expand_factor = self.conf.get_float('train.expand_factor')

            # Loss
            color_error = (color_fine - true_rgb) * mask
            color_error[indicator_object] *= expand_factor
            color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
            psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb)**2 * mask).sum() / (mask_sum * 3.0)).sqrt())

            semantic_loss = focal_loss(conf_fine[mask[:, 0]>=0.5], semantic[mask[:, 0]>=0.5][:, 0].long())

            eikonal_loss = gradient_error

            mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)

            if self.conf['model.warp']:
                warped_div = warped_div.view(weights.shape)
                div_loss = weights.detach() * warped_div ** 2
                div_loss = div_loss.mean()
            else:
                div_loss = 0

            loss = color_fine_loss +\
                   semantic_loss * self.sem_weight +\
                   eikonal_loss * self.igr_weight +\
                   mask_loss * self.mask_weight +\
                   div_loss * self.warp_weight

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_step += 1

            self.writer.add_scalar('Loss/loss', loss, self.iter_step)
            self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
            self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
            self.writer.add_scalar('Loss/mask_loss', mask_loss, self.iter_step)
            self.writer.add_scalar('Loss/sem_loss', semantic_loss, self.iter_step)
            self.writer.add_scalar('Loss/div_loss', div_loss, self.iter_step)
            self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
            self.writer.add_scalar('Statistics/cdf', (cdf_fine[:, :1] * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)

            if self.iter_step % self.report_freq == 0:
                print(self.base_exp_dir)
                print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr']))

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            if self.iter_step % self.val_freq == 0:
                self.validate_image()

            if self.iter_step % self.val_mesh_freq == 0:
                self.validate_mesh()
            
            if self.iter_step % self.val_all_freq == 0:
                for idx in range(self.dataset.n_images):
                    self.validate_image(idx=idx, blend=True)

            self.update_learning_rate()

            image_perm = self.get_image_perm()

    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        self.warp_network.load_state_dict(checkpoint['warp_network_fine'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']

        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'nerf': self.nerf_outside.state_dict(),
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'warp_network_fine': self.warp_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def validate_image(self, idx=-1, resolution_level=-1, blend=False):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        pose_refine = camera.lie.se3_to_SE3(self.sdf_network.se3_refine.weight)
        pose_all = camera.pose.compose([pose_refine, self.dataset.pose_all[:, :3, :]])
        rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level, pose_all=pose_all)
        H, W, _ = rays_o.shape
        img_idx = torch.tensor(idx).repeat(H*W).long()
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)
        img_idx = img_idx.reshape(-1).split(self.batch_size)

        out_rgb_fine = []
        out_conf_fine = []
        out_normal_fine = []
        out_offset_mag = []

        for rays_o_batch, rays_d_batch, img_idx_batch in zip(rays_o, rays_d, img_idx):
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              # near,
                                              # far,
                                              time=img_idx_batch,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            def feasible(key): return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            if feasible('conf_fine'):
                conf_fine = render_out['conf_fine'].detach().cpu().numpy()
                weight_sum = render_out['weight_sum'].detach().cpu().numpy()
                conf_idx = np.argmax(conf_fine, axis=1)
                out_conf_fine_ = np.zeros_like(conf_fine)
                out_conf_fine_[:, 0][conf_idx == 1] = 1
                out_conf_fine_[:, 1][conf_idx == 2] = 1
                out_conf_fine_[:, 2][conf_idx == 1] = 1
                out_conf_fine_[:, 2][conf_idx == 2] = 1
                out_conf_fine_[weight_sum[:, 0] <= 0.5] = 0
                out_conf_fine.append(out_conf_fine_)
            if feasible('offset_mag'):
                out_offset_mag.append(render_out['offset_mag'].detach().cpu().numpy())
            if feasible('gradients') and feasible('weights'):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)
            del render_out

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)
        if len(out_conf_fine) > 0:
            conf_fine = (np.concatenate(out_conf_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)
        if len(out_offset_mag) > 0:
            offset_mag = (np.concatenate(out_offset_mag, axis=0).reshape([H, W, 1, -1]) * 256).clip(0, 255)

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            pose_refine = camera.lie.se3_to_SE3(self.sdf_network.se3_refine.weight)
            pose_all = camera.pose.compose([pose_refine, self.dataset.pose_all[:, :3, :]])
            rot = np.linalg.inv(pose_all[idx, :3, :3].detach().cpu().numpy())
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                          .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)

        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        # os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine_gt'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'validations_sem'), exist_ok=True)
        # os.makedirs(os.path.join(self.base_exp_dir, 'validations_sem_gt'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'offset'), exist_ok=True)

        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                if blend:
                    cv.imwrite(os.path.join(self.base_exp_dir,
                                            'validations_fine',
                                            '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)), img_fine[..., i])

                    psnr_img1 = img_fine[..., i] / 256.
                    psnr_img2 = self.dataset.image_at(idx, resolution_level=resolution_level).astype(np.float32) / 256.0

                    mask = self.dataset.sem_at(idx, resolution_level=resolution_level).astype(np.float32)
                    mask = (mask[..., 1] == 255).astype(np.float32)
                    mask_sum = mask.sum() + 1e-5
                    mask = mask[..., None]
                    psnr = 20.0 * np.log10(1.0 / np.sqrt(((psnr_img1 - psnr_img2)**2 * mask).sum() / (mask_sum * 3.0)))

                else:
                    psnr = 0.0
                    cv.imwrite(os.path.join(self.base_exp_dir,
                                            'validations_fine',
                                            '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                            np.concatenate([img_fine[..., i],
                                            self.dataset.image_at(idx, resolution_level=resolution_level)]))
            if len(out_conf_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                            'validations_sem',
                                            '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)), conf_fine[..., i])
            if len(out_normal_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'normals',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           normal_img[..., i])
            if len(out_offset_mag) > 0:
                offset_img = offset_mag[..., i]
                offset_img = cv.normalize(offset_img, None, 0, 255, cv.NORM_MINMAX)
                offset_img = cv.applyColorMap(offset_img.astype(np.uint8), cv.COLORMAP_JET)
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'offset',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           offset_img)
        
        return psnr


    def render_novel_image(self, idx_0, idx_1, ratio, resolution_level):
        """
        Interpolate view between two cameras.
        """
        pose_refine = camera.lie.se3_to_SE3(self.sdf_network.se3_refine.weight)
        pose_all = camera.pose.compose([pose_refine, self.dataset.pose_all[:, :3, :]])
        rays_o, rays_d = self.dataset.gen_rays_between(idx_0, idx_1, ratio, resolution_level=resolution_level, pose_all=pose_all)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              # near,
                                              # far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())

            del render_out

        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255).astype(np.uint8)
        return img_fine

    def validate_mesh(self, world_space=False, resolution=64, threshold=0.0):
        pose_refine = camera.lie.se3_to_SE3(self.sdf_network.se3_refine.weight)
        pose_all = camera.pose.compose([pose_refine, self.dataset.pose_all[:, :3, :]])
        dis_ = torch.linalg.norm(self.dataset.pose_all[:, :3, -1]-self.dataset.pose_all[:, :3, -1].mean(0))
        dis = torch.linalg.norm(pose_all[:, :, -1]-pose_all[:, :, -1].mean(0))
        barf_scale = dis / dis_

        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)

        vertices, triangles =\
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        # color
        colors = self.renderer.extract_color(vertices, triangles, bound_min, bound_max, resolution=resolution)
        colors = colors[:, ::-1]  # BGR to RGB

        # sem
        confs = self.renderer.extract_conf(vertices, bound_min, bound_max, resolution=resolution)
        conf_idx = np.argmax(confs, axis=1)
        sems = np.zeros_like(confs)
        sems[:, 0][conf_idx == 1] = 1
        sems[:, 1][conf_idx == 2] = 1
        sems[:, 2][conf_idx == 1] = 1
        sems[:, 2][conf_idx == 2] = 1
        sems = sems[:, ::-1]  # BGR to RGB

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        # back to input scale
        vertices = vertices / barf_scale.cpu().detach().numpy()
        vertices *= self.dataset.radius
        vertices += self.dataset.origin

        mesh_org = trimesh.Trimesh(vertices, triangles)
        mesh_color = trimesh.Trimesh(vertices, triangles, vertex_colors=colors)
        mesh_sem = trimesh.Trimesh(vertices, triangles, vertex_colors=sems)
        mesh_org.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}_org.ply'.format(self.iter_step)))        
        mesh_color.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}_color.ply'.format(self.iter_step)))
        mesh_sem.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}_sem.ply'.format(self.iter_step)))

        # seperate object
        mesh_obj = trimesh.Trimesh(vertices, triangles, vertex_colors=colors, process=False)
        mask = (conf_idx == 2)
        face_mask = mask[triangles].all(axis=1)
        mesh_obj.update_vertices(mask)
        mesh_obj.update_faces(face_mask)
        mesh_obj.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}_obj.ply'.format(self.iter_step)))

        # seperate hand
        mesh_hand = trimesh.Trimesh(vertices, triangles, vertex_colors=colors, process=False)
        mask = (conf_idx == 1)
        face_mask = mask[triangles].all(axis=1)
        mesh_hand.update_vertices(mask)
        mesh_hand.update_faces(face_mask)
        mesh_hand.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}_hand.ply'.format(self.iter_step)))

        logging.info('End')

    def interpolate_view(self, img_idx_0, img_idx_1):
        images = []
        n_frames = 60
        for i in range(n_frames):
            print(i)
            images.append(self.render_novel_image(img_idx_0,
                                                  img_idx_1,
                                                  np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5,
                          resolution_level=4))
        for i in range(n_frames):
            images.append(images[n_frames - i - 1])

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video_dir = os.path.join(self.base_exp_dir, 'render')
        os.makedirs(video_dir, exist_ok=True)
        h, w, _ = images[0].shape
        writer = cv.VideoWriter(os.path.join(video_dir,
                                             '{:0>8d}_{}_{}.mp4'.format(self.iter_step, img_idx_0, img_idx_1)),
                                fourcc, 30, (w, h))

        for image in images:
            writer.write(image)

        writer.release()


if __name__ == '__main__':
    print('Hello Wooden')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='')

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, args.mode, args.case, args.is_continue)

    if args.mode == 'train':
        runner.train()
    elif args.mode == 'validate_mesh':
        runner.validate_mesh(world_space=True, resolution=512, threshold=args.mcube_threshold)
    elif args.mode.startswith('interpolate'):  # Interpolate views given two image indices
        _, img_idx_0, img_idx_1 = args.mode.split('_')
        img_idx_0 = int(img_idx_0)
        img_idx_1 = int(img_idx_1)
        runner.interpolate_view(img_idx_0, img_idx_1)
    elif args.mode == 'psnr':
        psnrs = []
        for idx in range(runner.dataset.n_images):
            psnr = runner.validate_image(idx=idx, resolution_level=2, blend=True)
            psnrs.append(psnr)

            print(f'psnr: {psnr}')
   
        print(psnrs)
        print(np.mean(psnrs))
