from functools import partial
from collections import OrderedDict as odict
import math
import joblib
import os.path as osp
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn

from core.net.sub.hmd_former.mapping import Mlp
from core.util.exe_util.util import make_instance


class HMD_former(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.N_vtx = 431
        self.D_vtx = 512
        self.all_mesh_types = ['full', 'sparse']  # cfg.NETWORK.METRO.mesh_upsampler_type == 'static'
        self.cfg = cfg

        # backbone
        self.backbone = make_instance(cfg.backbone)

        # duplex embed mapping
        self.D_up_down = make_instance(cfg.duplex_mapping)

        # attention decoder
        self.transformer = make_instance(cfg.transformer)

        # mesh_up
        self.mesh_up = make_instance(cfg.mesh_up)

        # cam head
        # cfg.NETWORK.METRO.cam_pos == 'img_feat'
        self.cam_proj_N = nn.Linear(7 * 7, 1)
        self.cam_proj_D = Mlp(2048, 64, 3)

    def main_params(self):
        params = list()
        params += list(self.backbone.parameters())
        params += list(self.transformer.parameters())
        if hasattr(self, 'D_up_down'):
            params += list(self.D_up_down.parameters())
        # cfg.NETWORK.METRO.kp2d_loss_type == 'proj'
        params += list(self.cam_proj_N.parameters())
        params += list(self.cam_proj_D.parameters())
        return params

    def mesh_up_params(self):
        params = list(self.mesh_up.parameters())
        return params

    def forward(self, img, mvm_mask=None, mpm_mask=None, mode='train', need_extra=False, total_itr=-1):
        B = img.shape[0]

        main_context = torch.no_grad if mode == 'train_mesh_up_only' else nullcontext
        with main_context():
            # extract feature
            img_feat = self.backbone(img)  # (B, 3, 224, 224) -> (B, C, H_f, W_f) = (B, 2048, 7, 7)

            # cam
            # cfg.NETWORK.METRO.kp2d_loss_type == 'proj' and cfg.NETWORK.METRO.cam_pos == 'img_feat'
            img_feat_last = img_feat[-1]  # (B, 2048, 7, 7)
            img_feat_last = img_feat_last.flatten(start_dim=2)  # (B, 2048, 49)
            cam = self.cam_proj_N(img_feat_last)  # (B, 2048, 1)
            cam = torch.transpose(cam, 1, 2)  # (B, 1, 2048)
            cam = self.cam_proj_D(cam)  # (B, 1, 3)
            cam = cam.squeeze(dim=1)  # (B, 3)

            ##### attention
            ### arange D_up (duplex mapping layer)
            D_up = self.D_up_down.forward_front
            if not self.cfg.train_D_up:
                D_up = partial(D_up, detach=True)
            ### forward attention
            multi_layer_tokens = self.transformer(img_feat, mvm_mask=mvm_mask, mpm_mask=mpm_mask, mode=mode,
                                                  D_up=D_up,
                                                  need_extra=need_extra, total_itr=total_itr)
            if need_extra and isinstance(multi_layer_tokens, tuple):
                multi_layer_tokens, extra = multi_layer_tokens
            else:
                extra = {}

        # mesh up
        D_down = self.D_up_down.forward_back
        if not self.cfg.train_D_down or mode == 'train_mesh_up_only':
            # train且 train_D_down == False时，传给meshup的D_down需要detach；此外，train_mesh_up_only时也需要
            D_down = partial(D_down, detach=True)
        # 对decoder最后的若干个阶段的tokens进行接下来的运算；auxiliary loss
        if mode in ['train', ]:
            num_layers = self.cfg.auxiliary_loss_layers
        else:
            num_layers = 1
        all_vtx3d = self.mesh_up(multi_layer_tokens[-num_layers:], D_down=D_down)

        outputs = dict()
        outputs['all_vtx3d'] = all_vtx3d  # N_smpl * (B, L, V, 3)
        outputs['cam'] = cam
        if need_extra:
            outputs['extra'] = extra

        return outputs
