from configparser import NoOptionError
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.embedder import get_embedder
from models.mip_utils import integrated_pos_enc
from tqdm import tqdm


class FieldNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 d_t4d,
                 min_emb,
                 max_emb,
                 n_layers,
                 t_emb=-1,
                 skip_in=(4,),
                 bias=0.5,
                 geometric_init=True,
                 weight_norm=True):
        super(FieldNetwork, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]
        dims[0] = d_in + (max_emb - min_emb)*3*2

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.min_emb = min_emb
        self.max_emb = max_emb
        self.t_emb = t_emb

        if t_emb > 0:
            embed_fn, time_input_ch = get_embedder(t_emb, input_dims=1)
            self.embed_fn = embed_fn
            dims[0] += time_input_ch

        for l in range(0, self.num_layers - 1):
            if l in self.skip_in:
                in_dim = dims[l] + dims[0] + d_t4d
            else:
                in_dim = dims[l]
            out_dim = dims[l+1]

            lin = nn.Linear(in_dim, out_dim)
            
            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] + d_t4d):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)

    def set_tensor4d(self, tensor4d):
        self.tensor4d = tensor4d

    def forward(self, mean, cov, fid, time_emb, reg_l2=False):
        cones_embedding = integrated_pos_enc((mean[:, None, :], cov[:, None, :]), self.min_emb, self.max_emb, diagonal=True).reshape(mean.shape[0], -1)
        inputs = mean
        tri_feat = self.tensor4d(inputs, fid, torch.mean(time_emb))

        if reg_l2:
            d_vec = F.normalize(torch.randn_like(inputs), dim=-1) * 1e-3
            d_tri_feat = self.tensor4d(inputs + d_vec, fid, torch.mean(time_emb))
            pred_reg_l2 = (d_tri_feat - tri_feat)**2
        
        xyz = inputs
        if self.t_emb > 0:
            time_input = self.embed_fn(time_emb)
            x = torch.cat([xyz, cones_embedding, time_input], 1)
        else:
            x = torch.cat([xyz, cones_embedding], 1)

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            
            if l in self.skip_in:
                if self.t_emb > 0:
                    x = torch.cat([x, tri_feat, xyz, cones_embedding, time_input], 1) / np.sqrt(2)
                else:
                    x = torch.cat([x, tri_feat, xyz, cones_embedding], 1) / np.sqrt(2)
            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        if reg_l2:
            return x, pred_reg_l2
        return x


# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class RenderingNetwork(nn.Module):
    def __init__(self,
                 d_feature,
                 mode,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 weight_norm=True,
                 multires_view=0,
                 squeeze_out=True):
        super().__init__()

        self.mode = mode
        self.squeeze_out = squeeze_out
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()

        self.mask = -torch.ones((1, 1, 256, 256, 256)).float().cuda()
    

    def forward(self, points, normals, view_dirs, feature_vectors):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)
            
        rendering_input = NoOptionError

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)

        x = rendering_input
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        if self.squeeze_out:
            x = torch.sigmoid(x)
        return x
        

class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        init_tensor = torch.zeros(120)
        init_tensor[:] = init_val
        self.register_parameter('variance', nn.Parameter(init_tensor))

    def forward(self, x):
        return torch.ones([len(x), 1]) * torch.exp(self.variance[0] * 10.0)