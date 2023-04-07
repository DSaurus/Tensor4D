import os
import time
import logging
import argparse
import numpy as np
import cv2 as cv
import trimesh
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from icecream import ic
from tqdm import tqdm
from pyhocon import ConfigFactory
from models.dataset import Dataset
from models.fields import RenderingNetwork, FieldNetwork, SingleVarianceNetwork
from models.tensor4d import Tensor4D
from models.renderer import NeuSRenderer
from models.mask import Mask3D
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

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
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.dataset = Dataset(self.conf['dataset'])
        self.g_nums = self.conf['dataset']['g_nums']
        self.iter_step = 0
        self.flow = self.conf.get_bool('model.flow', default=False)

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.fine_level_iter = self.conf.get_int('train.fine_level_iter')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.warm_up_imgs = self.conf.get_int('train.warm_up_imgs', default=50)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)
        self.mask_color_loss = self.conf.get_bool('train.mask_color_loss')
        self.weighted_sample = self.conf.get_bool('train.weighted_sample')

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.tgr_weight = self.conf.get_float('train.tgr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.tv_weight = self.conf.get_float('train.tv_weight')
        if self.tv_weight > 0:
            self.reg_l2 = True
        else:
            self.reg_l2 = False
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        # Masks
        self.mask3d = Mask3D(**self.conf['model.mask3d'], num_frames=self.dataset.n_images // self.g_nums, device=self.device)

        # Networks
        self.tensor4d = Tensor4D(**self.conf['model.tensor4d']).to(self.device)
        self.sdf_network = FieldNetwork(d_t4d=self.tensor4d.dims, **self.conf['model.sdf_network']).to(self.device)
        if self.flow:
            self.flow_tensor4d = Tensor4D(**self.conf['model.flow_tensor4d']).to(self.device)
            self.flow_network = FieldNetwork(d_t4d=self.flow_tensor4d.dims, **self.conf['model.flow_network']).to(self.device)
        else:
            self.flow_network = None
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
        
        params_to_train = []
        params_to_train += list(self.deviation_network.parameters())
        if self.flow:
            params_to_train += list(self.flow_network.parameters())
            params_to_train += list(self.flow_tensor4d.get_network_parameters())
        params_to_train += list(self.color_network.parameters())
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.tensor4d.get_network_parameters())
        
        params_data = []
        if self.flow:
            params_data = self.flow_tensor4d.get_data_parameters()
        params_data = self.tensor4d.get_data_parameters()

        self.renderer = NeuSRenderer(self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     self.mask3d,
                                     **self.conf['model.neus_renderer'],
                                     flow_network=self.flow_network,
                                     reg_l2=self.reg_l2)
        
        
        self.optimizer_data =  torch.optim.Adam([{ 'params': params_data, 'lr': self.learning_rate * 5}])
        self.optimizer_net = torch.optim.Adam([{'params': params_to_train, 'lr': self.learning_rate }])
        
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
        
        if self.mask3d.mask_type == 'visualhull':
            self.mask3d.compute_image_mask(self.dataset.proj_all, self.dataset.masks, self.g_nums)

        self.sdf_network.set_tensor4d(self.tensor4d)
        if self.flow:
            self.flow_network.set_tensor4d(self.flow_tensor4d)

        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()

    def train(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        self.last_step = self.iter_step
        loss_list = []
        
        image_perm = self.get_image_perm()
        self.max_fid = self.dataset.n_images // self.g_nums
        
        self.tensor4d.feature_plane.level = 1
        for iter_i in tqdm(range(res_step)):
   
            img_idx = image_perm[self.iter_step % len(image_perm)]
            if self.iter_step < self.warm_up_end:
                img_idx = img_idx % self.warm_up_imgs
            # print(img_idx)
            data, pixel_y, pixel_x = self.dataset.gen_random_rays_at(img_idx, self.batch_size)
            fid = self.dataset.fid_all[img_idx]
            time_emb = self.dataset.time_emb_list[img_idx]

            if self.tensor4d.image_guide:
                conv_idx = img_idx // self.g_nums * self.g_nums
                self.tensor4d.set_images(self.dataset.images[conv_idx:conv_idx+self.g_nums].to(self.device), self.dataset.proj_all[conv_idx:conv_idx+self.g_nums].to(self.device))
            
            if self.iter_step > self.fine_level_iter:
                self.tensor4d.feature_plane.level = 2

            rays_o, rays_d, true_rgb, mask, rays_r = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10], data[:, 10: 11]
            near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)

            background_rgb = None
            if self.use_white_bkgd:
                background_rgb = torch.ones([1, 3])

            if self.mask_weight > 0.0:
                mask = (mask > 0.5).float()
            else:
                mask = torch.ones_like(mask)

            mask_sum = mask.sum() + 1e-5
            render_out = self.renderer.render(rays_o, rays_d, rays_r, near, far, fid, time_emb,
                                              background_rgb=background_rgb,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio())

            color_fine = render_out['color_fine']
            s_val = render_out['s_val']
            weights = render_out['weights'].unsqueeze(-1).detach()
            gradient_error = render_out['gradient_error']
            time_grad_error = render_out['time_grad_error']
            reg_l2 = render_out['reg_l2']
            weight_max = render_out['weight_max']
            weight_sum = render_out['weight_sum']

            # Loss
            if self.mask_color_loss:
                color_error = (color_fine - true_rgb) * mask
                error = torch.sum(torch.abs(color_fine * mask - true_rgb * mask), dim=-1)
            else:
                if self.use_white_bkgd:
                    true_rgb = true_rgb * mask + (1-mask)
                else:
                    true_rgb = true_rgb * mask
                color_error = color_fine - true_rgb
                error = torch.sum(torch.abs(color_fine - true_rgb), dim=-1)

            if self.weighted_sample:
                self.dataset.errors[img_idx][(pixel_y // 8, pixel_x // 8)] = self.dataset.errors[img_idx][(pixel_y // 8, pixel_x // 8)]*0.25 + error.detach().cpu()[:, None]

            batch_size, _ = rays_d.shape

            color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
            psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb)**2 * mask).sum() / (mask_sum * 3.0)).sqrt())

            eikonal_loss = gradient_error
            tv_loss = torch.mean(torch.sum(reg_l2*weights, dim=1))

            mask1 = mask > 0.5
            mask0 = mask < 0.5
            mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)
            # reg_loss = self.data_network.feature_plane.tv_loss(img_idx // self.g_nums)

            loss = color_fine_loss +\
                   eikonal_loss * self.igr_weight +\
                   mask_loss * self.mask_weight + time_grad_error * self.tgr_weight + tv_loss * self.tv_weight

            self.optimizer_net.zero_grad()
            self.optimizer_data.zero_grad()
            loss.backward()
            self.optimizer_net.step()
            self.optimizer_data.step()

            self.iter_step += 1
            loss_list.append(color_fine_loss.item())

            self.writer.add_scalar('Loss/loss', loss, self.iter_step)
            self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
            self.writer.add_scalar('Loss/mask_loss', mask_loss, self.iter_step)
            self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
            self.writer.add_scalar('Loss/tv_loss', tv_loss, self.iter_step)
            self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
            self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)
            
            if self.iter_step % self.report_freq == 0:
                print(self.base_exp_dir)
                print(self.max_fid, image_perm)
                print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss, self.optimizer_net.param_groups[0]['lr']))

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()
                torch.cuda.empty_cache()

            if self.iter_step % self.val_freq == 0:
                self.validate_image()
                torch.cuda.empty_cache()
            
            # self.validate_image()
            # torch.cuda.empty_cache()

            if self.iter_step % (len(image_perm)) == 0:
                image_perm = self.get_image_perm()

            self.update_learning_rate()

    def get_image_perm(self):
        return torch.arange(0, self.dataset.n_images).long()
        
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

        self.optimizer_net.param_groups[0]['lr'] = self.learning_rate * learning_factor
        self.optimizer_data.param_groups[0]['lr'] = self.learning_rate * learning_factor * 5
        

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
        self.tensor4d.load_state_dict(checkpoint['tensor4d'], strict=False)
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'], strict=False)
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'], strict=False)
        self.optimizer_net.load_state_dict(checkpoint['optimizer_net'])
        self.optimizer_data.load_state_dict(checkpoint['optimizer_data'])
        self.iter_step = checkpoint['iter_step']
        self.max_fid = checkpoint['max_fid']

        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'optimizer_net': self.optimizer_net.state_dict(),
            'optimizer_data': self.optimizer_data.state_dict(),
            'tensor4d': self.tensor4d.state_dict(),
            'iter_step': self.iter_step,
            'max_fid': self.max_fid
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def validate_image(self, idx=-1, resolution_level=-1):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)
        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))

        if self.tensor4d.image_guide:
            conv_idx = idx // self.g_nums * self.g_nums
            self.tensor4d.set_images(self.dataset.images[conv_idx:conv_idx+self.g_nums].to(self.device), self.dataset.proj_all[conv_idx:conv_idx+self.g_nums].to(self.device))
            
        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d, rays_r = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
        fid = self.dataset.fid_all[idx]
        time_emb = self.dataset.time_emb_list[idx]
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)
        rays_r = rays_r.reshape(-1, 1).split(self.batch_size)

        out_rgb_fine = []
        out_normal_fine = []
        out_grad_fine = []

        for rays_o_batch, rays_d_batch, rays_r_batch in zip(rays_o, rays_d, rays_r):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              rays_r_batch,
                                              near,
                                              far,
                                              fid,
                                              time_emb,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            def feasible(key): return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            if feasible('gradients') and feasible('weights'):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                grads = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                grads = grads.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)
                out_grad_fine.append(grads)
            del render_out

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                          .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)
            
            grad_img = np.concatenate(out_grad_fine, axis=0)
            grad_img = (np.matmul(rot[None, :, :], grad_img[:, :, None])
                          .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)

        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)

        error = self.dataset.errors[idx]
        error = error / torch.max(error + 1e-8)
        cv.imwrite(os.path.join(self.base_exp_dir,
                                        'validations_fine',
                                        '{:0>8d}_{}_{}_err.png'.format(self.iter_step, 0, idx)), (error.detach().cpu().numpy()*255).astype(np.uint8))

        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'validations_fine',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           np.concatenate([img_fine[..., i],
                                           self.dataset.image_at(idx, resolution_level=resolution_level)]))
            if len(out_normal_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'normals',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           normal_img[..., i])
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'normals',
                                        '{:0>8d}_{}_{}_grad.png'.format(self.iter_step, i, idx)),
                           grad_img[..., i])

    def render_novel_image(self, idx_0, idx_1, ratio, resolution_level, fid, time_emb):
        """
        Interpolate view between two cameras.
        """
        rays_o, rays_d, rays_r = self.dataset.gen_rays_between(idx_0, idx_1, ratio, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)
        rays_r = rays_r.reshape(-1, 1).split(self.batch_size)

        out_rgb_fine = []
        out_normal_fine = []
        for rays_o_batch, rays_d_batch, rays_r_batch in tqdm(zip(rays_o, rays_d, rays_r)):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None
            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              rays_r_batch,
                                              near,
                                              far, 
                                              torch.LongTensor(np.array([fid])),
                                              time_emb,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            n_samples = self.renderer.n_samples + self.renderer.n_importance
            normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
            
            def feasible(key): return (key in render_out) and (render_out[key] is not None)
            if feasible('inside_sphere'):
                normals = normals * render_out['inside_sphere'][..., None]
            normals = normals.sum(dim=1).detach().cpu().numpy()
            out_normal_fine.append(normals)
            
            del render_out

        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255).astype(np.uint8)
        normal_img = np.concatenate(out_normal_fine, axis=0)
        return img_fine, normal_img


    def interpolate_view(self, img_idx_0, img_idx_1):
        images = []
        n_frames = 30
        video_dir = os.path.join(self.base_exp_dir, 'render')
        os.makedirs(video_dir, exist_ok=True)
        self.tensor4d.feature_plane.level = 2
        self.tensor4d.conv_net.eval()
        N = 50
        for i in range(n_frames):
            fid = i % (2*N-2)
            if fid > (N-1):
                fid = (2*N-2) - fid
            time_emb = self.dataset.time_emb_list[fid]
            if self.tensor4d.image_guide:
                conv_idx = fid * self.g_nums
                self.tensor4d.set_images(self.dataset.images[conv_idx:conv_idx+self.g_nums].to(self.device), self.dataset.proj_all[conv_idx:conv_idx+self.g_nums].to(self.device))
            print(fid)
            rot = np.linalg.inv(self.dataset.pose_all[0, :3, :3].detach().cpu().numpy())
            image, normal = self.render_novel_image(img_idx_0,
                                                  img_idx_1,
                                                  i / n_frames,
                          resolution_level=2, fid=fid, time_emb=time_emb)
            # print(normal.shape)
            H, W, _ = image.shape
            normal = (np.matmul(rot[None, :, :], normal[:, :, None])
                          .reshape([H, W, 3]) * 128 + 128).clip(0, 255)
            cv.imwrite(os.path.join(self.base_exp_dir, 'render', '%04d.jpg' % i), image)
            cv.imwrite(os.path.join(self.base_exp_dir, 'render', 'normal_%04d.jpg' % i), normal)
            images.append(image)
        for i in range(n_frames):
            images.append(images[n_frames - i - 1])

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
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
    torch.set_num_threads(8)
    runner = Runner(args.conf, args.mode, args.case, args.is_continue)

    if args.mode == 'train':
        runner.train()
    elif args.mode.startswith('interpolate'):  # Interpolate views given two image indices
        # runner.validate_image(0)
        _, img_idx_0, img_idx_1 = args.mode.split('_')
        img_idx_0 = int(img_idx_0)
        img_idx_1 = int(img_idx_1)
        runner.interpolate_view(img_idx_0, img_idx_1)
