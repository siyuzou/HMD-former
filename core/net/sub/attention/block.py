from collections import OrderedDict
import math

import torch
import torch.nn as nn
from timm.models.layers import DropPath

from core.net.tensor_util import dropout_layer

dropout = nn.Dropout


# dropout = dropout_layer


class Mlp(nn.Module):
    """ Borrowed from PoseFormer, further modified.
    """

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            drop=0.
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        # x = self.drop(x)  # DETR中这里没有做drop
        return x


class MultiheadAttention(nn.Module):
    """ Borrowed from PoseFormer, further modified.
        Multi-head Self Attention Layer
    """

    def __init__(
            self,
            D,
            num_heads=4,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.,
            proj_drop=0.,
            hard_attn=False,
    ):
        assert D % num_heads == 0
        super().__init__()

        self.D = D
        self.num_heads = num_heads
        self.hard_attn = hard_attn

        D_head = D // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or D_head ** -0.5

        if not self.hard_attn:
            self.linear_q = nn.Linear(D, D, bias=qkv_bias)
            self.linear_k = nn.Linear(D, D, bias=qkv_bias)
        self.linear_v = nn.Linear(D, D, bias=qkv_bias)

        self.proj = nn.Linear(D, D)

        self.attn_drop = dropout(attn_drop) if attn_drop > 0 else nn.Identity()
        self.proj_drop = dropout(proj_drop) if proj_drop > 0 else nn.Identity()

    def forward(self, q, k, v, need_extra=False, attn_mask=None, soft_attn_mask=False):
        """ soft_attn_mask == False时，attn_mask是二值的，1表示被丢弃，0表示保留
            soft_attn_mask == True时，attn_mask是连续的，越靠近1表示重要性越弱，越靠近0表示重要性越强
        """
        if not self.hard_attn:
            B, N_q, D_q = q.shape
            _, N_k, D_k = k.shape
            q = self.linear_q(q).reshape(B, N_q, self.num_heads, self.D // self.num_heads).permute(0, 2, 1, 3)
            k = self.linear_k(k).reshape(B, N_k, self.num_heads, self.D // self.num_heads).permute(0, 2, 1, 3)

            attn_raw = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N_q, N_k)
            if attn_mask is not None:
                if soft_attn_mask:
                    attn = attn_raw + torch.log(1 - attn_mask)
                else:
                    attn = attn_raw.masked_fill(attn_mask, float('-inf'))
            else:
                attn = attn_raw.clone()
            attn = attn.softmax(dim=-1)  # (B, H, N_q, N_k)
            attn_dropped = self.attn_drop(attn)
            # above: (B, H, N, N)
        else:
            attn_raw = torch.log(1 - attn_mask)  # -inf ~ 0
            attn_dropped = attn = attn_raw.softmax(dim=-1)

        B, N_v, D_v = v.shape
        _, _, N_q, N_k = attn.shape
        v = self.linear_v(v).reshape(B, N_v, self.num_heads, self.D // self.num_heads).permute(0, 2, 1, 3)
        # (B, H, N_q, D//H) -> (B, N_q, H, D//H) -> (B, N, D)
        x = (attn_dropped @ v).transpose(1, 2).reshape(B, N_q, self.D)
        x = self.proj(x)
        x = self.proj_drop(x)

        if need_extra:
            extra = {
                'attn_raw': attn_raw,
                'attn': attn
            }
            if attn_mask is not None:
                extra['attn_mask'] = attn_mask
            return x, extra
        else:
            return x


class TransformerEncoderLayer(nn.Module):
    """ Borrowed from PoseFormer, further modified.
        Transformer Encoder Block Layer
    """

    def __init__(
            self,
            D,
            D_mlp_hidden=None,
            num_heads=4,
            qkv_bias=True,
            qk_scale=None,
            pre_norm=True,
            drop=0.,
            attn_drop=0.,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()

        self.pre_norm = pre_norm

        self.norm1 = norm_layer(D)
        self.self_attn = MultiheadAttention(
            D=D, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = norm_layer(D)
        D_mlp_hidden = D * 4 if D_mlp_hidden is None else D_mlp_hidden
        self.mlp = Mlp(in_features=D, hidden_features=D_mlp_hidden, drop=drop)

        self.dropout = nn.Dropout(drop)

    def tokens_with_PE(self, tokens, PE):
        return tokens if PE is None else tokens + PE

    def forward(self, tokens, PE_enc=None, need_extra=False):
        """
        :param tokens:      (B, N, D)
        :param PE_enc:      (B, N, D)
        :return:
        """

        if self.pre_norm:
            return self.forward_pre_norm(tokens, PE_enc, need_extra=need_extra)
        else:
            return self.forward_post_norm(tokens, PE_enc, need_extra=need_extra)

    def forward_pre_norm(self, tokens, PE_enc, need_extra=False):
        """
        :param tokens:  (B, N, D)
        :param PE_enc:  (B, N, D)
        :return:
        """

        tokens2 = self.norm1(tokens)
        q = k = self.tokens_with_PE(tokens2, PE_enc)
        v = tokens2
        tokens2 = self.self_attn(q, k, v, need_extra=need_extra)
        if need_extra:
            tokens2, extra_SA = tokens2
        tokens = tokens + self.dropout(tokens2)

        tokens2 = self.norm2(tokens)
        tokens2 = self.mlp(tokens2)
        tokens = tokens + self.dropout(tokens2)

        if need_extra:
            extra = dict()
            for k, v in extra_SA.items():
                extra[f'SA.{k}'] = v
            return tokens, extra
        else:
            return tokens

    def forward_post_norm(self, tokens, PE_enc, need_extra=False):
        """
        :param tokens:  (B, N, D)
        :param PE_enc:  (B, N, D)
        :return:
        """

        q = k = self.tokens_with_PE(tokens, PE_enc)
        v = tokens
        tokens2 = self.self_attn(q, k, v, need_extra=need_extra)
        if need_extra:
            tokens2, extra_SA = tokens2
        tokens = tokens + self.dropout(tokens2)
        tokens = self.norm1(tokens)

        tokens2 = self.mlp(tokens)
        tokens = tokens + self.dropout(tokens2)
        tokens = self.norm2(tokens)

        if need_extra:
            extra = dict()
            for k, v in extra_SA.items():
                extra[f'SA.{k}'] = v
            return tokens, extra
        else:
            return tokens


class TransformerEncoderBlock(nn.Module):

    def __init__(
            self,
            D,
            D_mlp_hidden=None,
            num_layers=4,
            num_heads=4,
            qkv_bias=True,
            qk_scale=None,
            pre_norm=True,
            norm_end=False,
            drop=0.,
            attn_drop=0.,
    ):
        super().__init__()

        D_mlp_hidden = D * 4 if D_mlp_hidden is None else D_mlp_hidden

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = TransformerEncoderLayer(
                D=D, D_mlp_hidden=D_mlp_hidden, num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                pre_norm=pre_norm,
                drop=drop, attn_drop=attn_drop
            )
            self.layers.append(layer)

        if pre_norm and norm_end:
            self.norm_end = nn.LayerNorm(D)
        else:
            self.norm_end = None

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, tokens_enc, PE_enc=None, every_layer=False, need_extra=False):
        results = []
        if need_extra:
            extra_all = dict()

        for lid, layer in enumerate(self.layers):
            tokens_enc = layer(tokens_enc, PE_enc, need_extra=need_extra)
            if need_extra:
                tokens_enc, extra = tokens_enc
                for k, v in extra.items():
                    extra_all[f'{lid}.{k}'] = v

            if every_layer and lid != len(self.layers) - 1:
                results.append(tokens_enc)

        if self.norm_end:
            tokens_enc = self.norm_end(tokens_enc)

        results.append(tokens_enc)

        if need_extra:
            return results, extra_all
        else:
            return results


class TransformerDecoderLayer(nn.Module):
    """ Borrowed from PoseFormer, further modified.
        Transformer Encoder Block Layer
    """

    comp_2_name = {'SA': 'self_attn', 'CA': 'cross_attn', 'FFD': 'mlp'}

    def __init__(
            self,
            D,
            D_mlp_hidden=None,
            num_heads=4,
            qkv_bias=True,
            qk_scale=None,
            pre_norm=True,
            drop=0.,
            attn_drop=0.,
            norm_layer=nn.LayerNorm,
            CA_hard_attn=False,
            form='SA-CA-FFD',
    ):
        components = form.split('-')
        assert all(comp in ['SA', 'CA', 'FFD'] for comp in components), f'error components: {components}'

        super().__init__()

        self.pre_norm = pre_norm
        self.CA_hard_attn = CA_hard_attn
        self.components = components

        # self.  # 为了能够load历史模型，名字要求统一

        for i, comp_str in enumerate(self.components):
            if comp_str == 'SA':
                comp = MultiheadAttention(
                    D=D, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            elif comp_str == 'CA':
                comp = MultiheadAttention(
                    D=D, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
                    hard_attn=CA_hard_attn)
            elif comp_str == 'FFD':
                D_mlp_hidden = D * 4 if D_mlp_hidden is None else D_mlp_hidden
                comp = Mlp(in_features=D, hidden_features=D_mlp_hidden, drop=drop)

            # register curr comp
            comp_name = self.comp_name(i)
            self.add_module(comp_name, comp)

            # register norm layer
            norm = norm_layer(D)
            norm_name = self.norm_name(i)
            self.add_module(norm_name, norm)

        self.dropout = nn.Dropout(drop)

    def comp_name(self, i):
        """
            e.g.,   components = ['SA', 'CA, 'SA', 'FFD'], i = 2
                    output = self_attn_2
        """
        comp_str = self.components[i]
        curr_comp_count = self.components[0:i].count(comp_str)  # 查询历史这种部件出现的次数，确定此部件的id
        comp_name = f'{self.comp_2_name[comp_str]}{"" if curr_comp_count == 0 else "_" + str(curr_comp_count + 1)}'  # exp: self_attn_2
        return comp_name

    def norm_name(self, i):
        norm_name = f'norm{i + 1}'  # norm1, norm2, norm3, 和历史定义吻合
        if self.components == ['CA', 'SA', 'FFD']:
            # TODO: 0830, swap_attn 历史实现时，norm1-SA & norm2-CA 强制配对，因此这里也需要这么操作
            norm_name = {0: 'norm2', 1: 'norm1', 2: 'norm3'}[i]
        return norm_name

    def tokens_with_PE(self, tokens, PE):
        return tokens if PE is None else tokens + PE

    def forward(self, tokens_enc, tokens_dec, PE_enc=None, PE_dec=None, need_extra=False,
                self_attn_mask=None, cross_attn_mask=None, soft_cross_attn_mask=False,
                mvm_mask=None):
        """
        :param tokens_enc:  (B, N_k, D)
        :param tokens_dec:  (B, N_q, D)
        :param PE_enc:      (B, N_k, D)
        :param PE_dec:      (B, N_q, D)
        :param self_attn_mask:   (H, N_q, N_q)
        :param mvm_mask:    (B, V, 1)
        :return:
        """

        if self.pre_norm:
            return self.forward_pre_norm(tokens_enc, tokens_dec, PE_enc, PE_dec,
                                         need_extra=need_extra,
                                         self_attn_mask=self_attn_mask,
                                         cross_attn_mask=cross_attn_mask, soft_cross_attn_mask=soft_cross_attn_mask,
                                         mvm_mask=mvm_mask)
        else:
            return self.forward_post_norm(tokens_enc, tokens_dec, PE_enc, PE_dec,
                                          need_extra=need_extra,
                                          self_attn_mask=self_attn_mask,
                                          cross_attn_mask=cross_attn_mask, soft_cross_attn_mask=soft_cross_attn_mask,
                                          mvm_mask=mvm_mask)

    def forward_pre_norm(self, tokens_enc, tokens_dec, PE_enc=None, PE_dec=None, need_extra=False,
                         self_attn_mask=None, cross_attn_mask=None, soft_cross_attn_mask=False,
                         mvm_mask=None, swap_attn=False):
        """
        :param tokens_enc:  (B, N_k, D)
        :param tokens_dec:  (B, N_q, D)
        :param PE_enc:      (B, N_k, D)
        :param PE_dec:      (B, N_q, D)
        :param mvm_mask:    (B, N_q, 1), values: {0, 1}, 0 means masked, 1 means retained
        :return:
        """

        if need_extra:
            extra = dict()

        for i, comp_str in enumerate(self.components):
            # get curr comp
            comp_name = self.comp_name(i)
            comp = eval(f'self.{comp_name}')

            # get curr norm
            norm_name = self.norm_name(i)
            norm = eval(f'self.{norm_name}')

            if comp_str == 'SA':
                ##### self-attn
                tokens_dec2 = norm(tokens_dec)
                q = k = self.tokens_with_PE(tokens_dec2, PE_dec)
                v = tokens_dec2
                tokens_dec2 = comp(q, k, v, need_extra=need_extra,
                                   attn_mask=self_attn_mask)
                if need_extra:
                    tokens_dec2, curr_extra = tokens_dec2
                    for k, v in curr_extra.items():
                        extra[f'SA{i + 1}.{k}'] = v

                tokens_dec = tokens_dec + self.dropout(tokens_dec2)

            elif comp_str == 'CA':
                ##### cross-attn
                tokens_dec2 = norm(tokens_dec)
                k = self.tokens_with_PE(tokens_enc, PE_enc)
                q = self.tokens_with_PE(tokens_dec2, PE_dec)
                v = tokens_enc
                tokens_dec2 = comp(q, k, v, need_extra=need_extra,
                                   attn_mask=cross_attn_mask, soft_attn_mask=soft_cross_attn_mask)
                if need_extra:
                    tokens_dec2, curr_extra = tokens_dec2
                    for k, v in curr_extra.items():
                        extra[f'CA{i + 1}.{k}'] = v
                if mvm_mask is not None:
                    constant_tensor = torch.ones_like(tokens_dec2) * 0.01
                    tokens_dec2 = tokens_dec2 * mvm_mask + constant_tensor * (1 - mvm_mask)
                tokens_dec = tokens_dec + self.dropout(tokens_dec2)

            elif comp_str == 'FFD':
                ##### FFD
                tokens_dec2 = norm(tokens_dec)
                tokens_dec2 = comp(tokens_dec2)
                tokens_dec = tokens_dec + self.dropout(tokens_dec2)

        if need_extra:
            return tokens_dec, extra
        else:
            return tokens_dec

    def forward_post_norm(self, tokens_enc, tokens_dec, PE_enc=None, PE_dec=None, need_extra=False,
                          self_attn_mask=None, cross_attn_mask=None, soft_cross_attn_mask=False,
                          mvm_mask=None):
        """
        :param tokens_enc:  (B, N_k, D)
        :param tokens_dec:  (B, N_q, D)
        :param PE_enc:      (B, N_k, D)
        :param PE_dec:      (B, N_q, D)
        :param mvm_mask:    (B, V, 1)
        :return:
        """

        if need_extra:
            extra = dict()

        for i, comp_str in enumerate(self.components):
            # get curr comp
            comp_name = self.comp_name(i)
            comp = eval(f'self.{comp_name}')

            # get curr norm
            norm_name = self.norm_name(i)
            norm = eval(f'self.{norm_name}')

            if comp_str == 'SA':
                ##### self-attn
                q = k = self.tokens_with_PE(tokens_dec, PE_dec)
                v = tokens_dec
                tokens_dec2 = comp(q, k, v, need_extra=need_extra,
                                   attn_mask=self_attn_mask)
                if need_extra:
                    tokens_dec2, curr_extra = tokens_dec2
                    for k, v in curr_extra.items():
                        extra[f'SA{i + 1}.{k}'] = v
                tokens_dec = tokens_dec + self.dropout(tokens_dec2)
                tokens_dec = norm(tokens_dec)

            elif comp_str == 'CA':
                ##### cross-attn
                k = self.tokens_with_PE(tokens_enc, PE_enc)
                q = self.tokens_with_PE(tokens_dec, PE_dec)
                v = tokens_enc
                tokens_dec2 = comp(q, k, v, need_extra=need_extra,
                                   attn_mask=cross_attn_mask, soft_attn_mask=soft_cross_attn_mask)
                if need_extra:
                    tokens_dec2, curr_extra = tokens_dec2
                    for k, v in curr_extra.items():
                        extra[f'CA{i + 1}.{k}'] = v
                if mvm_mask is not None:
                    constant_tensor = torch.ones_like(tokens_dec2) * 0.01
                    tokens_dec2 = tokens_dec2 * mvm_mask + constant_tensor * (1 - mvm_mask)
                tokens_dec = tokens_dec + self.dropout(tokens_dec2)
                tokens_dec = norm(tokens_dec)

            elif comp_str == 'FFD':
                ##### FFD
                tokens_dec2 = comp(tokens_dec)
                tokens_dec = tokens_dec + self.dropout(tokens_dec2)
                tokens_dec = norm(tokens_dec)

        if need_extra:
            return tokens_dec, extra
        else:
            return tokens_dec


class TransformerDecoderBlock(nn.Module):

    def __init__(
            self,
            D,
            D_mlp_hidden=None,
            num_layers=4,
            num_heads=4,
            qkv_bias=True,
            qk_scale=None,
            pre_norm=True,
            norm_end=False,
            drop=0.,
            attn_drop=0.,
            layer_form='SA-CA-FFD',
            add_SA_to_first_layer=False
    ):
        super().__init__()

        D_mlp_hidden = D * 4 if D_mlp_hidden is None else D_mlp_hidden
        self.layer_form = layer_form

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            curr_layer_form = 'SA-' + self.layer_form if i == 0 and add_SA_to_first_layer else self.layer_form
            layer = TransformerDecoderLayer(
                D=D, D_mlp_hidden=D_mlp_hidden, num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                pre_norm=pre_norm,
                drop=drop, attn_drop=attn_drop,
                form=curr_layer_form
            )
            self.layers.append(layer)

        if pre_norm and norm_end:
            self.norm_end = nn.LayerNorm(D)
        else:
            self.norm_end = None

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, tokens_enc, tokens_dec, PE_enc=None, PE_dec=None,
                every_layer=False, need_extra=False, attn_mask=None, mvm_mask=None):
        results = []
        if need_extra:
            extra_all = dict()

        for lid, layer in enumerate(self.layers):
            attn_mask_at_l = None if attn_mask is None else attn_mask[lid]  # (H, N, N)
            tokens_dec = layer(tokens_enc, tokens_dec, PE_enc, PE_dec, need_extra=need_extra,
                               self_attn_mask=attn_mask_at_l,
                               mvm_mask=mvm_mask)
            if need_extra:
                tokens_dec, extra = tokens_dec
                for k, v in extra.items():
                    extra_all[f'{lid}.{k}'] = v

            if every_layer and lid != len(self.layers) - 1:
                results.append(tokens_dec)

        if self.norm_end:
            tokens_dec = self.norm_end(tokens_dec)

        results.append(tokens_dec)

        if need_extra:
            return results, extra_all
        else:
            return results


def sine_positional_embedding_img(width, D, temperature=10000, normalize=False, scale=None):
    assert D % 2 == 0
    assert scale is None or normalize == True, "normalize should be True if scale is passed"

    D_pos = D // 2
    if scale is None:
        scale = 2 * math.pi

    single_axis = torch.arange(start=0, end=width, step=1, dtype=torch.float32) + 1
    x_embed = single_axis.unsqueeze(0).unsqueeze(0).repeat(1, width, 1)  # (1, 7, 7)
    y_embed = single_axis.unsqueeze(-1).unsqueeze(0).repeat(1, 1, width)

    if normalize:
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

    dim_t = torch.arange(D_pos, dtype=torch.float32)  # (128, )
    dim_t = temperature ** (2 * (dim_t // 2) / D_pos)

    pos_x = x_embed[:, :, :, None] / dim_t  # (B, 7, 7, 128)
    pos_y = y_embed[:, :, :, None] / dim_t  # (B, 7, 7, 128)
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)  # (B, 7, 7, 64*2)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)  # (B, 7, 7, 64*2)
    pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # (B, 256, 7, 7)
    return pos
