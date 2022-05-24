import torch
import torch.nn as nn

from core.net.sub.smpl.smpl import upsample_transforms
from core.net.sub.hmd_former.mapping import Mlp


class MeshUpsampler(nn.Module):
    def __init__(self,
                 D_vtx,
                 given_D_down=False,
                 mesh_up_type='static',
                 detach_after_sparse=False
                 ):
        super().__init__()

        self.D_vtx = D_vtx  # 512
        self.given_D_down = given_D_down  # true
        self.mesh_up_type = mesh_up_type  # static
        self.detach_after_sparse = detach_after_sparse

        if mesh_up_type == 'static':
            # mesh upsampling use fixed transform matrix
            if not given_D_down:
                self.fc_D = nn.Linear(self.D_vtx, 3)
            U_t = upsample_transforms[0] @ upsample_transforms[1]
            self.register_buffer('U_t', torch.from_numpy(U_t[None, ...]))  # (1, V_full, V_sparse)
        elif mesh_up_type == 'metro':
            # like MeshTransformer, use 2 FC to upsample the mesh
            self.mesh_upsampling = torch.nn.Linear(431, 1723)
            self.mesh_upsampling2 = torch.nn.Linear(1723, 6890)
        elif mesh_up_type == 'mlp':
            if detach_after_sparse:
                U_t = upsample_transforms[0] @ upsample_transforms[1]
                self.register_buffer('U_t', torch.from_numpy(U_t[None, ...]))  # (1, V_full, V_sparse)

            V_up = [431, 512, 1723, 2048, 6890]
            self.mesh_upsampling = Mlp(V_up[0], V_up[1], V_up[2])
            self.mesh_upsampling2 = Mlp(V_up[2], V_up[3], V_up[4])
        elif mesh_up_type == '2fc':
            if detach_after_sparse:
                U_t = upsample_transforms[0] @ upsample_transforms[1]
                self.register_buffer('U_t', torch.from_numpy(U_t[None, ...]))  # (1, V_full, V_sparse)
            self.mesh_upsampling = torch.nn.Linear(431, 1723)
            self.mesh_upsampling2 = torch.nn.Linear(1723, 6890)

        # D down
        if not given_D_down:
            # # 08152021, 如果没有 D_down，统一给一个可学习的 fc
            # self.fc_D = nn.Linear(self.D_vtx, 3)
            D_down = [512, 128, 3]
            if mesh_up_type == 'mlp':
                # todo: 1129, 这里默认 MeshUpsampler.share_final_layer == True
                # 为 vtx3d_sparse 专门给定一个 mlp_D
                self.mlp_D_sparse = Mlp(self.D_vtx, D_down[1], D_down[2])
            self.mlp_D = Mlp(self.D_vtx, D_down[1], D_down[2])

        self.reset_parameters()

    def reset_parameters(self):
        pass
        """ Linear 默认自带 Kaiming norm 初始化 
        """

    def forward(self, multi_layer_tokens, D_down=None):
        """
        :param multi_layer_tokens: L * (B, 431, D_vtx)
        :return:
        """
        assert not self.given_D_down or D_down, f'duplex_D_mapping strategy need D_down'

        L = len(multi_layer_tokens)
        B, _, _ = multi_layer_tokens[0].shape
        x = torch.stack(multi_layer_tokens, dim=1).flatten(start_dim=0, end_dim=1)  # (B*L, 431, D_vtx)

        all_vtx3d = list()
        if self.mesh_up_type == 'static':
            D_down = D_down or self.fc_D

            vtx3d_sparse = D_down(x)  # (B, V_sparse, 3)
            # fixed upsampler
            vtx3d_full = self.U_t @ vtx3d_sparse  # (B, V_full, 3)

            all_vtx3d = [vtx3d_full, vtx3d_sparse]
        elif self.mesh_up_type == 'metro':
            D_down = D_down or self.fc_D

            vtx3d_sparse = D_down(x)  # (B, V_sparse, 3)

            vtx3d_sparse_trans = vtx3d_sparse.transpose(1, 2)  # (B, 3, V_sparse)
            vtx3d_mediate_trans = self.mesh_upsampling(vtx3d_sparse_trans)  # (B, 3, V_mediate)
            vtx3d_full_trans = self.mesh_upsampling2(vtx3d_mediate_trans)  # (B, 3, V_full)
            vtx3d_mediate = vtx3d_mediate_trans.transpose(1, 2)  # (B, V_mediate, 3)
            vtx3d_full = vtx3d_full_trans.transpose(1, 2)  # (B, V_full, 3)

            all_vtx3d = [vtx3d_full, vtx3d_mediate, vtx3d_sparse]
        elif self.mesh_up_type in ['mlp', '2fc']:
            # arrange D_down for all vtx3d
            if self.given_D_down:
                D_down_sparse = D_down_all = D_down
            else:
                # D_down_sparse = self.mlp_D if cfg.NETWORK.MeshUpsampler.share_final_layer else self.mlp_D_sparse
                # D_down_all = self.mlp_D
                D_down_sparse = D_down_all = self.fc_D

            # sparse
            vtx3d_sparse = D_down_sparse(x)

            if self.detach_after_sparse:
                # detach tokens x
                x = x.detach()

            # mediate
            x_sparse_T = x.transpose(1, 2)  # (B, D_vtx, V_sparse)
            x_mediate_T = self.mesh_upsampling(x_sparse_T)
            x_mediate = x_mediate_T.transpose(1, 2)
            vtx3d_mediate = D_down_all(x_mediate)

            # full
            x_full_T = self.mesh_upsampling2(x_mediate_T)
            x_full = x_full_T.transpose(1, 2)
            vtx3d_full = D_down_all(x_full)

            if self.detach_after_sparse:
                # full_static
                vtx3d_full_static = self.U_t @ vtx3d_sparse  # (B, V_full, 3)

                all_vtx3d = [vtx3d_full_static, vtx3d_full, vtx3d_mediate, vtx3d_sparse]
            else:
                all_vtx3d = [vtx3d_full, vtx3d_mediate, vtx3d_sparse]

        for i in range(len(all_vtx3d)):
            all_vtx3d[i] = all_vtx3d[i].reshape(B, L, -1, 3)
        return all_vtx3d
