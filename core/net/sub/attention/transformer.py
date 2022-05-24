from functools import partial
import math
import joblib
import os.path as osp
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from core.net.sub.attention.block import TransformerDecoderBlock, sine_positional_embedding_img
from core.net.sub.smpl.smpl import load_canonical_sparse_smpl_vtx


class Transformer(nn.Module):

    def __init__(
            self,
            feature_map_size=[7],
            feature_map_depth=[2048],
            given_D_up=False,
            dec_layer_form='SA-CA-FFD',
            pre_norm=False,
            use_PE_enc=False,
            use_mvm=False,
    ):
        super().__init__()

        self.feature_map_size = feature_map_size
        self.feature_map_depth = feature_map_depth
        self.given_D_up = given_D_up
        self.dec_layer_form = dec_layer_form
        self.pre_norm = pre_norm
        self.use_PE_enc = use_PE_enc
        self.use_mvm = use_mvm

        N = 431
        D_enc = D_dec = 512
        num_layers = 6
        num_heads = 4

        self.N = self.N_dec = N
        self.D_enc = D_enc
        self.D_dec = D_dec

        # pre-linear
        if self.feature_map_depth[0] == self.D_enc:
            self.linear_down_0 = nn.Identity()
        else:
            self.linear_down_0 = nn.Linear(self.feature_map_depth[0], self.D_enc)

        # Encoder positional embedding
        PE_enc = sine_positional_embedding_img(feature_map_size[0], self.D_enc)  # (1, D_dec, 7, 7)
        PE_enc = PE_enc.flatten(start_dim=2).permute(0, 2, 1)  # (1, 49, D_dec)
        self.register_buffer('PE_enc', PE_enc)

        # Decoder positional embedding
        # smpl_V_sparse_canonic = smpl_mesh_sparse_canonic.v.astype(np.float32)  # (V_sparse, 3)
        smpl_V_sparse_canonic = load_canonical_sparse_smpl_vtx()
        self.register_buffer('smpl_V_sparse_canonic', torch.from_numpy(smpl_V_sparse_canonic))

        # D_up
        if not self.given_D_up:
            self.linear_PE_dec = nn.Linear(3, self.D_dec)

        ##### attention
        drop = 0.0
        dec_add_SA_to_first_layer = False
        # attention decoder
        self.decoder = TransformerDecoderBlock(
            D=self.D_dec, D_mlp_hidden=4 * self.D_dec, num_layers=num_layers, num_heads=num_heads,
            qkv_bias=True, qk_scale=None,
            pre_norm=self.pre_norm, norm_end=True,
            drop=drop, attn_drop=drop,
            layer_form=dec_layer_form, add_SA_to_first_layer=dec_add_SA_to_first_layer
        )

    def forward(self, img_feat, mode='train', mvm_mask=None, mpm_mask=None, D_up=None, need_extra=False,
                *args, **kwargs):
        """
        :param img_feat:    (B, 2048, 7, 7)
        :param mvm_mask:
        :param mode:
        :return:
        """
        assert not self.given_D_up or D_up, f'duplex_D_mapping strategy need D_up'

        B = img_feat[0].shape[0]

        # featmap proj, [512, 1024, 2048]->512
        if len(self.feature_map_size) == 1:
            # TODO: v0.2.14.6 以及之前，只考虑单尺度 img_feat；暂时保留这一层的命名规则，否则会和之后的冲突
            img_feat = img_feat[0].reshape(B, self.feature_map_depth[0], -1).transpose(1, 2)  # (B, 49, 2048)
            tokens_enc = self.linear_down_0(img_feat)  # (B, 49, 512)
        else:
            tokens_enc = []
            for s in range(len(self.feature_map_size)):
                img_feat_at_scale = img_feat[s].reshape(B, self.feature_map_depth[s], -1).transpose(1, 2)  # (B,H*W,C)
                tokens_enc_at_scale = eval(f'self.featmap_proj_{s}')(img_feat_at_scale)
                tokens_enc.append(tokens_enc_at_scale)
            tokens_enc = torch.cat(tokens_enc, dim=1)  # (B, 28*28+14*14+7*7=1029, 512)

        # positional embedding for Encoder
        PE_enc = self.PE_enc.to(tokens_enc.device).repeat(B, 1, 1)  # (B, 49or1029, 1024)

        # positional embedding for Decoder
        smpl_pos = self.smpl_V_sparse_canonic[None, ...]  # (1, 431, 3)
        D_up_func = D_up if self.given_D_up else self.linear_PE_dec
        PE_dec = D_up_func(smpl_pos)  # (1, 431, 1024)
        PE_dec = PE_dec.repeat(B, 1, 1)  # (B, 431, 1024)

        # encoder
        output_enc = [tokens_enc]
        if need_extra:
            extra_enc = dict()

        # decoder
        tokens_dec = PE_dec
        if not self.use_PE_enc:
            PE_enc = None
        mvm_mask = mvm_mask if self.use_mvm else None
        tokens_enc = output_enc[-1]
        output_dec = self.decoder(tokens_enc, tokens_dec, PE_enc, PE_dec,
                                  every_layer=True, need_extra=need_extra,
                                  attn_mask=None, mvm_mask=mvm_mask)
        if need_extra:
            output_dec, extra_dec = output_dec

        # after arange extra, return the results
        if need_extra:
            extra = dict()
            for k, v in extra_enc.items():
                extra[f'enc.{k}'] = v
            for k, v in extra_dec.items():
                extra[f'dec.{k}'] = v
            return output_dec, extra
        else:
            return output_dec
