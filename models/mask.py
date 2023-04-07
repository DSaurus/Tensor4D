import torch
import torch.nn 
import torch.nn.functional as F
from tqdm import tqdm

class Mask3D:
    def __init__(self, mask_type, num_frames=None, mask_reso=None, device=None):
        self.mask_type = mask_type # 'bounding or visualhull'
        if mask_type == 'visualhull':
            self.R = mask_reso
            self.mask = torch.ones([num_frames, self.R, self.R, self.R]).float()
            self.device = device
        self.current_fid = -1
        self.current_mask = None

    def set_fid(self, fid):
        if fid != self.current_fid:
            self.current_fid = fid
            if self.mask_type == 'visualhull':
                self.current_mask = self.mask[fid.cpu()].to(self.device)
        
    def valid_input(self, pts, fid):
        with torch.no_grad():
            pts = pts.reshape(1, -1, 1, 1, 3)
            pts_max = torch.max(pts, dim=-1)[0]
            pts_min = torch.min(pts, dim=-1)[0]
            mask_max = (pts_max > 1).reshape(-1)
            mask_min = (pts_min < -1).reshape(-1)
            if self.mask_type == 'visualhull':
                R = self.R
                sigma = F.grid_sample(self.current_mask.view(1, 1, R, R, R), pts, mode='bilinear', padding_mode='border').reshape(-1)
                calc_mask = sigma < 0.05
            else:
                calc_mask = torch.ones_like(mask_max)
            calc_mask[mask_max] = 0
            calc_mask[mask_min] = 0
            return calc_mask

    def visualhull(self, pts_ori, projs, masks, g_nums):
        cam_nums = projs.shape[0]
        interval = 1
        pts_mask = torch.zeros(pts_ori.shape[0], g_nums)
        out_mask = torch.zeros(pts_ori.shape[0])
        N, H, W, C = masks.shape
        for gp in range(cam_nums // (g_nums*interval)):
            for j in range(g_nums):
                i = j + gp*(g_nums*interval)
                mask = masks[i, :, :, :1].permute(2, 0, 1).unsqueeze(0).clone()
                mask = torch.max_pool2d(mask, 7, 1, 3, 1)
                pts = torch.cat([pts_ori, torch.ones_like(pts_ori[:, :1])], dim=-1)
                pts = projs[i] @ pts.T
                pts = pts[:2] / pts[2:]
                pts[0] = pts[0] / W * 2 - 1
                pts[1] = pts[1] / H * 2 - 1
                pts = pts.T.reshape(1, -1, 1, 2)
                
                sample_mask = torch.nn.functional.grid_sample(mask, pts, mode='bilinear', padding_mode='zeros').reshape(-1)
                pts_mask[:, j] = sample_mask
            pts_mask_sum = torch.min(pts_mask, dim=1)[0]
            valid = pts_mask_sum > 0.1
            out_mask[valid] = -1
            if gp == 0:
                out_mask[~valid] = 1
        return out_mask

    def compute_image_mask(self, projs, masks, g_nums):
        N = 64
        R = self.R
        X = torch.linspace(-1, 1, R).split(N)
        Y = torch.linspace(-1, 1, R).split(N)
        Z = torch.linspace(-1, 1, R).split(N)
        cam_nums = projs.shape[0]
        
        self.mask = self.mask.to(self.device)
        for gp in tqdm(range(cam_nums // g_nums)):
        # for gp in range(1):
            with torch.no_grad():
                for xi, xs in enumerate(X):
                    for yi, ys in enumerate(Y):
                        for zi, zs in enumerate(Z):
                            xx, yy, zz = torch.meshgrid(xs, ys, zs)
                            pts = torch.cat([zz.reshape(-1, 1), yy.reshape(-1, 1), xx.reshape(-1, 1)], dim=-1).to(self.device)
                            val = self.visualhull(pts, projs[gp*g_nums:gp*g_nums+g_nums].to(self.device), masks[gp*g_nums:gp*g_nums+g_nums].to(self.device), g_nums).reshape(len(xs), len(ys), len(zs))
                            self.mask[gp, xi * N: xi * N + len(xs),yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
        self.mask = self.mask.unsqueeze(1)
        self.mask = -torch.max_pool3d(-self.mask, 7, 1, 3)
        self.mask[self.mask > -0.5] = 1
        self.mask = self.mask.detach().cpu()
        
    def compute_mask(self, fid, query_func, inv_s):
        N = 64
        R = 128
        X = torch.linspace(-1, 1, R).split(N)
        Y = torch.linspace(-1, 1, R).split(N)
        Z = torch.linspace(-1, 1, R).split(N)
        from .renderer import sigma_f
        mask = self.mask[fid].reshape(R, R, R).clone()
        self.triplane[0].flow(fid)
        with torch.no_grad():
            for xi, xs in enumerate(X):
                for yi, ys in enumerate(Y):
                    for zi, zs in enumerate(Z):
                        xx, yy, zz = torch.meshgrid(xs, ys, zs)
                        pts = torch.cat([zz.reshape(-1, 1), yy.reshape(-1, 1), xx.reshape(-1, 1)], dim=-1)
                        val = sigma_f(query_func(pts), inv_s).reshape(len(xs), len(ys), len(zs))
                        mask[xi * N: xi * N + len(xs),yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
        valid = mask > 0.02
        mask[valid] = 1
        mask[~valid] = -1
        mask = -torch.max_pool3d(mask.reshape(1, 1, 128, 128, 128), 7, 1, 3)
        self.mask[fid][mask[0] > -0.5] = 1