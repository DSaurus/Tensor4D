import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.mip_utils import cast_rays

def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    # cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples, device=cdf.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples])

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


class NeuSRenderer:
    def __init__(self,
                 sdf_network,
                 deviation_network,
                 color_network,
                 mask3d,
                 n_samples,
                 n_importance,
                 n_outside,
                 up_sample_steps,
                 perturb,
                 reg_l2=False,
                 mip_render=False,
                 flow_network=None):
                 
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        self.color_network = color_network
        self.mask3d = mask3d
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.n_outside = n_outside
        self.up_sample_steps = up_sample_steps
        self.perturb = perturb
        self.reg_l2 = reg_l2
        self.flow_network = flow_network
        self.mip_render = mip_render

    def mask_query_geometry(self, mean, cov, only_sdf=False):
        fid = self.fid
        time_emb = self.time_emb
        time_input = time_emb.expand(mean[:, :1].shape)
        space_time_input = torch.cat([mean, time_input], dim=-1)
        if not only_sdf:
            space_time_input.requires_grad_(True)
        inputs = space_time_input[:, :3]
        time_emb = space_time_input[:, 3:]
        N, _ = inputs.shape
        grads = torch.zeros((N, 4), device=inputs.device)
        sdf_nn = torch.zeros((N, 257), device=inputs.device)

        reg_l2 = torch.zeros((N, self.sdf_network.tensor4d.dims), device=inputs.device)
        grads[:, 0] = 1
        sdf_nn[:, 0] = -10

        mask = self.mask3d.valid_input(inputs, fid)
        if torch.sum(mask) == 0:
            results = {
                'sdf_nn': sdf_nn,
                'grads': grads[:, :3],
                'time_grads': grads[:, 3:],
                'pts_mask': mask,
                'reg_l2': reg_l2
            }
            return results
        mask_mean = inputs[mask, :]
        mask_time_emb = time_emb[mask, :]
        mask_cov = cov[mask, :]
        
        if self.flow_network is not None:
            mask_cov = torch.zeros_like(mask_mean) # flow mode, disable mip_render
            if fid != 0:
                pred_flow = self.flow_network(mask_mean, mask_cov, fid, mask_time_emb, reg_l2=False)
                mask_mean = mask_mean + pred_flow
        elif not self.mip_render:
            mask_cov = torch.zeros_like(mask_mean)

        if (not only_sdf) and self.reg_l2:
            pred_sdf_nn, pred_reg_l2 = self.sdf_network(mask_mean, mask_cov, fid, mask_time_emb, reg_l2=True)
            reg_l2[mask] = pred_reg_l2
        else:
            pred_sdf_nn = self.sdf_network(mask_mean, mask_cov, fid, mask_time_emb, reg_l2=False)

        if not only_sdf:
            pred_sdf = pred_sdf_nn[:, :1]
            d_output = torch.ones_like(pred_sdf, requires_grad=False, device=pred_sdf.device)
            gradients = torch.autograd.grad(
                outputs=pred_sdf,
                inputs=space_time_input,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
            grads[mask] = gradients.reshape(-1, 4)[mask]
            
        sdf_nn[mask] = pred_sdf_nn
        results = {
            'sdf_nn': sdf_nn,
            'grads': grads[:, :3],
            'time_grads': grads[:, 3:],
            'pts_mask': mask,
            'reg_l2': reg_l2
        }
        return results

    def mask_query_color(self, pts, mask, normals, view_dirs, features):
        N, _ = pts.shape
        out = torch.zeros((N, 3), device=pts.device)
        if torch.sum(mask) > 0:
            x = self.color_network(pts[mask], normals[mask], view_dirs[mask],  features[mask])
            out[mask] = x
            return out
        else:
            return torch.zeros((N, 3), device=pts.device)

    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s, pts_mask):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_mask, next_mask = pts_mask[:, :-1], pts_mask[:, 1:]
        mid_mask = torch.logical_and(prev_mask, next_mask)
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat([torch.zeros([batch_size, 1], device=sdf.device), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)

        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        alpha[~mid_mask] = 0
        alpha = alpha.clamp(0.0, 1.0)
        
        alpha = torch.cat([alpha, torch.zeros([batch_size, 1], device=alpha.device)], dim=-1)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1], device=alpha.device), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, pts_mask, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)
        if not last:
            new_sdf, new_pts_mask = self.sdf_network.sdf(pts.reshape(-1, 3), rt_mask=True)
            new_sdf = new_sdf.reshape(batch_size, n_importance)
            new_pts_mask = new_pts_mask.reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            pts_mask = torch.cat([pts_mask, new_pts_mask], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)
            pts_mask = pts_mask[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf, pts_mask

    def render_core(self,
                    rays_o,
                    rays_d,
                    rays_r,
                    z_vals,
                    sample_dist,
                    background_alpha=None,
                    background_sampled_color=None,
                    background_rgb=None,
                    cos_anneal_ratio=0.0):
        batch_size, n_samples = z_vals[:, :-1].shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        cat_dists = torch.cat([dists, torch.Tensor([sample_dist]).to(dists.device).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + cat_dists * 0.5

        cones = cast_rays(z_vals, rays_o, rays_d, rays_r, 'cone', diagonal=True)
        dirs = rays_d[:, None, :].expand(cones[0].shape)
        dirs = dirs.reshape(-1, 3)

        results = self.mask_query_geometry(cones[0].reshape(-1, 3), cones[1].reshape(-1, 3))
        sdf_nn_output, gradients, t_grads, pts_mask = results['sdf_nn'], results['grads'], results['time_grads'], results['pts_mask']
        sdf = sdf_nn_output[:, :1]
        feature_vector = sdf_nn_output[:, 1:]

        gradients = gradients.squeeze()
        sampled_color = self.mask_query_color(cones[0].reshape(-1, 3), pts_mask, gradients, dirs, feature_vector).reshape(batch_size, n_samples, 3)
        
        inv_s = self.deviation_network(torch.zeros([1, 3], device=sdf.device))[:, :1].clip(1e-6, 1e6)           # Single parameter
        inv_s = inv_s.expand(batch_size * n_samples, 1)

        true_cos = (dirs * gradients).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                    F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5))
        
        alpha[~pts_mask] = 0
        alpha = alpha.reshape(batch_size, n_samples).clip(0.0, 1.0)
       
        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1], device=alpha.device), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        weights_sum = weights.sum(dim=-1, keepdim=True)
        
        color = (sampled_color * weights[:, :, None]).sum(dim=1)
        if background_rgb is not None:    # Fixed background, usually black
            color = color + background_rgb * (1.0 - weights_sum)

        # Eikonal loss
        gradient_error = torch.mean((torch.linalg.norm(gradients.reshape(batch_size, n_samples, 3), ord=2,
                                            dim=-1) - 1.0) ** 2)
        time_grad_error = torch.mean(t_grads**2)
        return {
            'color': color,
            'sdf': sdf,
            'pts_mask': pts_mask,
            'dists': dists,
            'gradients': gradients.reshape(batch_size, n_samples, 3),
            's_val': 1.0 / inv_s,
            'mid_z_vals': mid_z_vals,
            'weights': weights,
            'gradient_error': gradient_error,
            'time_grad_error': time_grad_error,
            'reg_l2': results['reg_l2'].reshape(batch_size, n_samples, -1),
        }

    def render(self, rays_o, rays_d, rays_r, near, far, fid, time_emb, perturb_overwrite=-1, background_rgb=None, cos_anneal_ratio=0.0):
        self.fid = fid
        self.time_emb = time_emb
        self.mask3d.set_fid(fid)

        batch_size = len(rays_o)
        sample_dist = 2.0 / self.n_samples   # Assuming the region of interest is a unit sphere
        z_vals = torch.linspace(0.0, 1.0, self.n_samples, device=rays_o.device)
        z_vals = near + (far - near) * z_vals[None, :]

        z_vals_outside = None
        
        n_samples = self.n_samples
        perturb = self.perturb

        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1], device=z_vals.device) - 0.5)
            z_vals = z_vals + t_rand * 2.0 / self.n_samples

        background_alpha = None
        background_sampled_color = None

        # Up sample
        if self.n_importance > 0:
            with torch.no_grad():
                cast_z_vals = torch.cat([z_vals, z_vals[:, -1:]], dim=1)
                cones = cast_rays(cast_z_vals, rays_o, rays_d, rays_r, 'cone', diagonal=True)
                results = self.mask_query_geometry(cones[0].reshape(-1, 3), cones[1].reshape(-1, 3), only_sdf=True)
                sdf, pts_mask = results['sdf_nn'][:, :1], results['pts_mask']
                # sdf, pts_mask = self.sdf_network.sdf(pts.reshape(-1, 3), rt_mask=True)
                sdf = sdf.reshape(batch_size, self.n_samples)
                pts_mask = pts_mask.reshape(batch_size, self.n_samples)
                for i in range(self.up_sample_steps):
                    new_z_vals = self.up_sample(rays_o,
                                                rays_d,
                                                z_vals,
                                                sdf,
                                                self.n_importance // self.up_sample_steps + 1,
                                                64 * 2**i, pts_mask)
                    z_vals, sdf, pts_mask = self.cat_z_vals(rays_o,
                                                  rays_d,
                                                  z_vals,
                                                  new_z_vals,
                                                  sdf, pts_mask,
                                                  last=(i + 1 == self.up_sample_steps))

            n_samples = self.n_samples + self.n_importance

        background_alpha = None
        background_sampled_color = None
        sample_dist = 1e-2

        # Render core
        ret_fine = self.render_core(rays_o,
                                    rays_d,
                                    rays_r,
                                    z_vals,
                                    sample_dist,
                                    background_rgb=background_rgb,
                                    background_alpha=background_alpha,
                                    background_sampled_color=background_sampled_color,
                                    cos_anneal_ratio=cos_anneal_ratio)


        return {
            'color_fine': ret_fine['color'],
            's_val': ret_fine['s_val'].reshape(batch_size, n_samples).mean(dim=-1, keepdim=True),
            'mid_z_vals': ret_fine['mid_z_vals'],
            'weights': ret_fine['weights'],
            'weight_sum': ret_fine['weights'].sum(dim=-1, keepdim=True),
            'weight_max': torch.max(ret_fine['weights'], dim=-1, keepdim=True)[0],
            'gradients': ret_fine['gradients'],
            'gradient_error': ret_fine['gradient_error'],
            'time_grad_error': ret_fine['time_grad_error'],
            'reg_l2': ret_fine['reg_l2']
        }