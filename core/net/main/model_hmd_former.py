import os.path as osp
from collections import OrderedDict as odict

import torch.nn as nn
import torch.nn.functional as F
import joblib

from core.net.sub.hmd_former.hmd_former import HMD_former
from core.net.sub.smpl.smpl import SMPL, downsample_transforms
from core.net.tensor_util import *
from core.util.data_util.kp import cvt_full_to_kp


class Model(nn.Module):
    """ Main model.
        It contains the core model of hmd-former.
        When testing or inferring, it executes forward and manages the output.
        When training, it executes both forward and backward, so it manages the output
            and the loss calculation.

    """

    def __init__(self, cfg, mode='train'):
        super(Model, self).__init__()

        self.cfg = cfg
        self.mode = mode
        is_train = self.mode == 'train'

        #################### init network ####################
        # mesh generator
        self.hmd_former = HMD_former(cfg.hmd_former)

        # smpl
        self.smpl = SMPL(root=cfg.smpl.root)

        # J_reg
        self.J_types_full = cfg.hmd_former.J_types_full
        J_type_to_filename = {'Human36M': 'h36m_correct', 'MSCOCO': 'coco_hip_smpl'}
        for J_type in self.J_types_full:
            J_reg_path = osp.join(cfg.smpl.root, f'J_regressor_{J_type_to_filename[J_type]}.npy')
            J_reg = np.load(J_reg_path)
            J_reg = J_reg.astype(np.float32)[None, None, ...]  # (1, 1, J, V)
            J_reg = torch.from_numpy(J_reg)
            self.register_buffer(f'J_reg_{J_type}', J_reg)

    def models(self):
        """ 返回需要保存的子模块的 name 与 mode 的键值对
        """
        return {'hmd_former': self.hmd_former}

    def train(self, mode=True):
        self.hmd_former.train(mode)

    def reg_J_from_V(self, all_vtx3d, J_types, valid_reg_V_inds=None):

        all_kp3d_reg = odict()  # N_reg_smpl * (B, L, J0 + J1 + ..., 3)
        for J_type in J_types:
            J_reg = eval(f'self.J_reg_{J_type}')  # (1, J, V)
            # for vtx3d_ind, vtx3d in enumerate(all_vtx3d):
            for vtx3d_ind in valid_reg_V_inds or range(1):
                vtx3d = all_vtx3d[vtx3d_ind]  # (B, L, V, 3)
                # if vtx3d_ind == len(all_kp3d_reg):
                #     all_kp3d_reg.append(list())
                if vtx3d_ind not in all_kp3d_reg.keys():
                    all_kp3d_reg[vtx3d_ind] = list()

                kp3d_reg = J_reg @ vtx3d  # (B, L, J, 3)
                if J_type == 'MSCOCO':
                    """ MSCOCO 的最后一个节点人为给定 """
                    pelvis = (kp3d_reg[:, :, [11], :] + kp3d_reg[:, :, [12], :]) / 2  # (B, L, 1, 3)
                    kp3d_reg = torch.cat([kp3d_reg, pelvis], dim=2)  # (B, L, 17+1, 3)

                all_kp3d_reg[vtx3d_ind].append(kp3d_reg)
        all_kp3d_reg = [v for k, v in all_kp3d_reg.items()]
        for i in range(len(all_kp3d_reg)):
            all_kp3d_reg[i] = torch.cat(all_kp3d_reg[i], dim=2)  # (B, L, J0 + J1 + ..., 3)

        return all_kp3d_reg  # N_reg_smpl * (B, L, J0 + J1 + ..., 3)

    def forward(self, inputs, mode, **kwargs):
        if mode == 'train':
            assert False, f'training mode not implemented.'
        elif mode == 'test':
            need_extra = 'extra' in kwargs.keys() and kwargs['extra'] == True

            ##### prepare inputs need for METRO
            img = inputs['img']  # (B, C, H, W)

            ##### forward
            mesh_outputs = self.hmd_former(img, mode='test', need_extra=need_extra)
            ##### reg J from V
            all_kp3d_reg = self.reg_J_from_V(mesh_outputs['all_vtx3d'], ['Human36M'], valid_reg_V_inds=[0])

            outs = dict()
            """ root_ind, eval_inds 暂时用 h36m """
            root_ind = 0
            eval_inds = [1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16]

            ##### process gt
            ### kp3d_gt
            """ post process on kp3d_gt:
                    1. full-to-h36m
                    2. root-rel
                    3. to-eval
            """
            kp3d_gt = inputs['kp3d_cam']
            kp3d_gt = cvt_full_to_kp(kp3d_gt, J_type='Human36M', J_types=self.J_types_full)
            kp3d_gt = (kp3d_gt - kp3d_gt[:, [root_ind], :])[:, eval_inds, :]
            outs['kp3d_gt'] = kp3d_gt

            ##### process METRO outputs
            ### kp3d_pred: post-process pred kp (root-rel)
            """ post process on kp3d_reg_pred (h36m):
                    1. root-rel
                    2. to-eval
            """
            kp3d_reg_pred = all_kp3d_reg[0]  # (B, L, J_H36M_17, 3)
            root_smpl_reg_pred = kp3d_reg_pred[..., [root_ind], :]  # (B, L, 1, 3)
            kp3d_reg_pred = (kp3d_reg_pred - root_smpl_reg_pred)[..., eval_inds, :]  # (B, L, J_H36M_14, 3)
            outs['kp3d_pred'] = kp3d_reg_pred

            if 'eval' in kwargs and 'vtx3d_pred' in kwargs['eval']:
                ### vtx3d_pred
                vtx3d_full_pred = mesh_outputs['all_vtx3d'][0]  # (B, L, V, 3)
                vtx3d_full_pred = vtx3d_full_pred - root_smpl_reg_pred
                outs['vtx3d_pred'] = vtx3d_full_pred  # (B, L, V, 3)

                cam_param = mesh_outputs['cam']
                outs['cam'] = cam_param

            if need_extra:
                outs['extra'] = mesh_outputs['extra']

            return outs
