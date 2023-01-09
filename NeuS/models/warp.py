'''
Date: 2021-12-21 11:32:50 am
Author: dihuangdh
Descriptions: 
-----
LastEditTime: 2023-01-08 6:08:12 pm
LastEditors: dihuangdh
'''


import torch
import torch.nn as nn
from torch.autograd.functional import jacobian
from models.embedder import get_embedder
from functools import partial


# from FFJORD github code
def divergence_approx(input_points, offsets_of_inputs):  # , as_loss=True):
    # avoids explicitly computing the Jacobian
    e = torch.randn_like(
        offsets_of_inputs, device=offsets_of_inputs.get_device(), requires_grad=True)
    e_dydx = torch.autograd.grad(
        offsets_of_inputs, input_points, e, create_graph=True)[0]
    e_dydx_e = e_dydx * e
    approx_tr_dydx = e_dydx_e.view(offsets_of_inputs.shape[0], -1).sum(dim=1)
    return approx_tr_dydx


def compute_GM_error(squared_x, c):
    squared_x /= c ** 2
    return 2 * squared_x / (squared_x + 4)


init_func_dict = {
    'xavier_uniform': nn.init.xavier_uniform_,
    'uniform': partial(nn.init.uniform_, a=-1e-5, b=1e-5)
}


def skew(w):
    assert w.shape[-1] == 3
    s = torch.zeros((*w.shape[:-1], 3, 3), device=w.device, dtype=w.dtype)
    s[..., 0, 1] = -w[..., 2]
    s[..., 0, 2] = w[..., 1]
    s[..., 1, 2] = -w[..., 0]
    s = s - s.transpose(-1, -2)
    return s


def exp_so3(w, theta):
    W = skew(w)
    B = torch.prod(torch.tensor(theta.shape[:-1]))
    I = torch.eye(3, device=w.device, dtype=w.dtype).unsqueeze(
        0).expand(B, -1, -1).reshape(*theta.shape[:-1], 3, 3)
    theta = theta.unsqueeze(-1)
    return I + torch.sin(theta) * W + (1. - torch.cos(theta)) * W @ W


def exp_se3(S, theta):
    w, v = torch.split(S, [3, 3], dim=-1)
    B = torch.prod(torch.tensor(theta.shape[:-1]))
    W = skew(w)
    R = exp_so3(w, theta)
    I = torch.eye(3, device=w.device, dtype=w.dtype).unsqueeze(
        0).expand(B, -1, -1).reshape(*theta.shape[:-1], 3, 3)
    theta = theta.unsqueeze(-1)
    p = (theta * I + (1. - torch.cos(theta)) * W +
         (theta - torch.sin(theta)) * W @ W) @ v.unsqueeze(-1)
    return R, p.squeeze(-1)


class MLP(nn.Module):
    def __init__(self,
                 d_in,
                 d_hidden,
                 n_layers,
                 skip_in=(),
                 use_bias=True,
                 d_out=0,
                 out_activation=None,
                 out_init=None):
        super().__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)]

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim, bias=use_bias)

            nn.init.xavier_uniform_(lin.weight)
            nn.init.constant_(lin.bias, 0.0)

            setattr(self, 'lin' + str(l), lin)

        self.hidden_activation = nn.ReLU(True)

        self.d_out = d_out
        if self.d_out > 0:
            lin = nn.Linear(d_hidden, d_out, bias=use_bias)

            if out_init is not None:
                init_func_dict[out_init](lin.weight)
            nn.init.constant_(lin.bias, 0.0)
            setattr(self, 'lin_out', lin)

            self.out_activation = out_activation

    def forward(self, inputs):
        x = inputs

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, 'lin' + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1)

            x = lin(x)
            x = self.hidden_activation(x)

        if self.d_out > 0:
            lin = getattr(self, 'lin_out')
            x = lin(x)
            if self.out_activation is not None:
                x = self.out_activation(x)

        return x


class SE3Field(nn.Module):
    def __init__(self,
                 d_in=3,
                 d_vector=10,
                 trunk_depth=6,
                 trunk_width=128,
                 rotation_depth=0,
                 rotation_width=128,
                 pivot_depth=0,
                 pivot_width=128,
                 skip_in=(4,),
                 multires=6,
                 n_images=0,
                 enable=True,
                 ):
        super().__init__()
        self.enable = enable

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            self.input_ch = input_ch

        self.trunk = MLP(d_in=self.input_ch + d_vector,
                         d_hidden=trunk_width,
                         n_layers=trunk_depth,
                         skip_in=skip_in)

        self.w = MLP(d_in=trunk_width,
                     d_hidden=rotation_width,
                     n_layers=rotation_depth,
                     d_out=3,
                     out_init='uniform')

        self.v = MLP(d_in=trunk_width,
                     d_hidden=pivot_width,
                     n_layers=pivot_depth,
                     d_out=3,
                     out_init='uniform')
        
        # Latent-code
        self.lcode = torch.nn.Embedding(n_images, 10)
        torch.nn.init.zeros_(self.lcode.weight)
        # self.lcode.weight.requires_grad = False

        self.progress = torch.nn.Parameter(torch.tensor(0.)) # use Parameter so it could be checkpointed

    def warp(self, points, features, return_jacobian=False):
        points_embed = self.embed_fn(points, self.progress)

        points_embed = torch.cat([points_embed, features], dim=-1)
        trunk_output = self.trunk(points_embed)

        w = self.w(trunk_output)
        v = self.v(trunk_output)
        theta = torch.norm(w, dim=-1, keepdim=True)
        w = w / theta
        v = v / theta
        screw_aixs = torch.cat([w, v], dim=-1)
        R, p = exp_se3(screw_aixs, theta)

        warpped_points = R @ points.unsqueeze(-1) + p.unsqueeze(-1)
        warpped_points = warpped_points.squeeze(-1)

        return warpped_points, R 

    def forward(self, points, time, return_jacobian=False, return_div=False):
        lcode = self.lcode.weight[time]
        lcode = lcode[:, None, :].repeat(1, int(points.shape[0]/time.shape[0]), 1)
        features = lcode
        
        points = points.reshape(-1, 3)
        features = features.reshape(-1, 10)
        if self.enable:
            warpped_points, R = self.warp(points, features)
        else:
            warpped_points = points
            R = None
        out = {'warped_points': warpped_points, 'R': R, 'original_points': points}
        if return_div and self.enable:
            div = divergence_approx(points, warpped_points - points)
            out['div'] = div
        else:
            out['div'] = torch.zeros(0)
        return out