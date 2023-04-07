from configparser import NoOptionError
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from .embedder import get_embedder

def grid_sample(image, optical):
    N, C, IH, IW = image.shape
    image = image.view(N, C, IH * IW)
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    ix = ((ix + 1) / 2) * (IW-1)
    iy = ((iy + 1) / 2) * (IH-1)
    with torch.no_grad():
        ix_nw = torch.floor(ix)
        iy_nw = torch.floor(iy)
        ix_ne = ix_nw + 1
        iy_ne = iy_nw
        ix_sw = ix_nw
        iy_sw = iy_nw + 1
        ix_se = ix_nw + 1
        iy_se = iy_nw + 1

    nw = (ix_se - ix)    * (iy_se - iy)
    ne = (ix    - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix)    * (iy    - iy_ne)
    se = (ix    - ix_nw) * (iy    - iy_nw)
    
    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW-1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH-1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW-1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH-1, out=iy_ne)

        torch.clamp(ix_sw, 0, IW-1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH-1, out=iy_sw)

        torch.clamp(ix_se, 0, IW-1, out=ix_se)
        torch.clamp(iy_se, 0, IH-1, out=iy_se)


    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) + 
            ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
            sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
            se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    return out_val

class CompressLinear(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super(CompressLinear, self).__init__()
        self.in_layer = nn.Linear(in_dim, 256)
        self.activation = nn.LeakyReLU()
        self.out_layer = nn.Linear(256, out_dim)

    def forward(self, x):
        x = self.in_layer(x)
        x = self.activation(x)
        x = self.out_layer(x)
        return self.activation(x)
        

class CompressNetwork(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super(CompressNetwork, self).__init__()
        self.compress1 = CompressLinear(in_dim, out_dim)
        self.compress2 = CompressLinear(in_dim, out_dim)
        self.compress3 = CompressLinear(in_dim, out_dim)

class SpacePlane(nn.Module):
    def __init__(self, lr_resolution, hr_resolution):
        super(SpacePlane, self).__init__()
        # resolution [spatial, channel] 
        self.space_planes1 = torch.nn.Parameter(torch.zeros(3, lr_resolution[1], lr_resolution[0], lr_resolution[0]))
        self.space_planes = [self.space_planes1]
        self.dims = lr_resolution[1]*3

        if len(hr_resolution) > 0: 
            self.space_planes2 = torch.nn.Parameter(torch.zeros(3, hr_resolution[1], hr_resolution[0], hr_resolution[0]))
            self.space_planes.append(self.space_planes2)
            self.dims += hr_resolution[1]*3

        self.level = 1

    def sample(self, samples, idx, t_emb):
        pts_feature = torch.zeros((1, self.dims // 3, samples.shape[1], samples.shape[2]), device=samples.device)
        channel = 0
        for level in range(self.level):
            plane_coef_point = grid_sample(self.space_planes[level][idx].unsqueeze(0), samples)
            pts_feature[:, channel:channel + plane_coef_point.shape[1]] = plane_coef_point
            channel += plane_coef_point.shape[1]
        pts_feature = pts_feature.view(-1, samples.shape[1]*samples.shape[2])
        return pts_feature


class TimeSpacePlane(nn.Module):
    def __init__(self, lr_resolution, hr_resolution):
        super(TimeSpacePlane, self).__init__()
        # resolution [spatial, time, channel] 
        self.space_planes1 = torch.nn.Parameter(torch.zeros(3, lr_resolution[2], lr_resolution[0], lr_resolution[0]))
        self.time_space_planes1 = torch.nn.Parameter(torch.zeros(6, lr_resolution[2], lr_resolution[0], lr_resolution[1])) # x-axis time
        self.space_planes = [self.space_planes1]
        self.time_space_planes = [self.time_space_planes1]
        self.dims = lr_resolution[2]*3

        if len(hr_resolution) > 0: 
            self.space_planes2 = torch.nn.Parameter(torch.zeros(3, hr_resolution[2], hr_resolution[0], hr_resolution[0]))
            self.time_space_planes2 = torch.nn.Parameter(torch.zeros(6, hr_resolution[2], hr_resolution[0], hr_resolution[1])) # x-axis time
            self.space_planes.append(self.space_planes2)
            self.time_space_planes.append(self.time_space_planes2)
            self.dims += hr_resolution[2]*3

        self.level = 1

    def sample(self, samples, idx, t_emb):
        pts_feature = torch.zeros((1, self.dims, samples.shape[1], samples.shape[2]), device=samples.device)
        t_samples0 = torch.zeros_like(samples)
        t_samples1 = torch.zeros_like(samples)
        t_samples0[..., 0] = t_emb
        t_samples0[..., 1] = samples[..., 0]
        t_samples1[..., 0] = t_emb
        t_samples1[..., 1] = samples[..., 1]
        channel = 0
        for level in range(self.level):
            plane_coef_point = grid_sample(self.space_planes[level][idx].unsqueeze(0), samples)
            plane_coef_point1 = grid_sample(self.time_space_planes[level][idx*2].unsqueeze(0), t_samples0)
            plane_coef_point2 = grid_sample(self.time_space_planes[level][idx*2+1].unsqueeze(0), t_samples1)
            pts_feature[:, channel:channel + plane_coef_point.shape[1]*3] = torch.cat([plane_coef_point, plane_coef_point1, plane_coef_point2], dim=1)
            channel += plane_coef_point.shape[1]*3
        pts_feature = pts_feature.view(-1, samples.shape[1]*samples.shape[2])
        return pts_feature


class ConvNet(nn.Module):
    def __init__(self, base) -> None:
        super(ConvNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, base, 3, 2, 1),
            nn.GroupNorm(8, base),
            nn.LeakyReLU(),
            nn.Conv2d(base, base*2, 3, 2, 1),
            nn.LeakyReLU(8, base*2),
            nn.Softplus(),
            nn.Conv2d(base*2, base*4, 3, 1, 1),
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, base*4),
            nn.LeakyReLU(),
            nn.Conv2d(base*4, base*4, 3, 2, 1),
            nn.GroupNorm(8, base*4),
            nn.LeakyReLU(),
            nn.Conv2d(base*4, base*4, 3, 1, 1),
        )
    
    def forward(self, x):
        x_hr = self.conv(x)
        return self.conv2(x_hr), x_hr


class Tensor4D(nn.Module):
    def __init__(self, feature_type, lr_resolution, hr_resolution, image_guide=False, image_guide_interval=2, image_guide_base=16) -> None:
        super(Tensor4D, self).__init__()
        
        self.data_dims = 0
        self.feature_type = feature_type
        if feature_type == '3d':
            self.feature_plane = SpacePlane(lr_resolution, hr_resolution)
            self.data_dims = self.feature_plane.dims
        elif feature_type == '4d':
            self.feature_plane = TimeSpacePlane(lr_resolution, hr_resolution)
            self.data_dims = self.feature_plane.dims

        self.img_dims = 0
        self.image_guide = image_guide
        if image_guide:
            self.conv_net = ConvNet(image_guide_base)
            self.img_dims = image_guide_base*8*2
            self.ig_interval = image_guide_interval

        if feature_type == '4d':
            self.compress_network = CompressNetwork(self.data_dims, self.data_dims // 3)
            self.compress_list = [self.compress_network.compress1, self.compress_network.compress2, self.compress_network.compress3]

        self.dims = self.data_dims + self.img_dims
        self.matMode = torch.BoolTensor([[0, 1, 1], [1, 0, 1], [1, 1, 0]]).cuda()
        self.vecMode = torch.BoolTensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).cuda()
        
    def get_data_parameters(self):
        return list(self.feature_plane.parameters())
    
    def get_network_parameters(self):
        params = []
        if self.feature_type == '4d':
            params += list(self.compress_network.parameters())
        if self.image_guide:
            params += list(self.conv_net.parameters())
        return params

    def set_images(self, image, proj):
        step = self.ig_interval
        select_proj = torch.cat([proj[i*step:i*step+1] for i in range(proj.shape[0] // step)], dim=0)
        self.proj = select_proj
        self.img_shape = image.shape
        select_image = torch.cat([image[i*step:i*step+1] for i in range(image.shape[0] // step)], dim=0)
        self.image_feature, self.image_feature_hr = self.conv_net(F.interpolate(select_image.permute(0, 3, 1, 2), size=(1024, 1024)))

    def forward(self, xyz_sampled_ori, fid, time_emb):
        sigma_feature_list = [] 

        if self.image_guide:
            proj_pts = ((self.proj[:, :3, :3] @ xyz_sampled_ori.T.unsqueeze(0)) + self.proj[:, :3, 3:]).transpose(1, 2)
            proj_xy = proj_pts[:, :, :2] / (proj_pts[:, :, 2:] + 1e-6)
            B, H, W, C = self.img_shape
            proj_xy[:, :, 0] = (proj_xy[:, :, 0] - W / 2) / (W / 2)
            proj_xy[:, :, 1] = (proj_xy[:, :, 1] - H / 2) / (H / 2)
            N = self.image_feature.shape[0]
            img_feature = grid_sample(self.image_feature, proj_xy.reshape(N, -1, 1, 2)).reshape(N, -1, xyz_sampled_ori.shape[0])
            img_feature_cost = torch.sqrt(torch.sum((img_feature - torch.sum(img_feature, dim=0).unsqueeze(0) / N)**2, dim=0) / N + 1e-8)
            img_feature_max = torch.mean(img_feature, dim=0) + torch.max(img_feature, dim=0)[0]
            image_feature_hr = grid_sample(self.image_feature_hr, proj_xy.reshape(N, -1, 1, 2)).reshape(N, -1, xyz_sampled_ori.shape[0])
            image_feature_hr_cost = torch.sqrt(torch.sum((image_feature_hr - torch.sum(image_feature_hr, dim=0).unsqueeze(0) / N)**2, dim=0) / N + 1e-8)
            image_feature_hr_max = torch.mean(image_feature_hr, dim=0) + torch.max(image_feature_hr, dim=0)[0]
            sigma_feature_list = [img_feature_cost, img_feature_max, image_feature_hr_cost, image_feature_hr_max]
        
        xyz_sampled = xyz_sampled_ori
        scale = 1.0
        matMode = self.matMode
        coordinate_plane = torch.stack((xyz_sampled[..., matMode[0]] * scale, xyz_sampled[..., matMode[1]] * scale, xyz_sampled[..., matMode[2]] * scale)).view(3, -1, 1, 2)

        for idx_plane in range(3):
            sample_points = coordinate_plane[[idx_plane]]
            plane_coef_point = self.feature_plane.sample(sample_points, idx_plane, time_emb).view(-1, *xyz_sampled.shape[:1])
            if self.feature_type == '4d':
                plane_coef_point = self.compress_list[idx_plane](plane_coef_point.T).T
            sigma_feature_list.append(plane_coef_point)
            
        sigma_feature_list = torch.cat(sigma_feature_list, dim=0)
        # print(sigma_feature_list.shape)
        return sigma_feature_list.T