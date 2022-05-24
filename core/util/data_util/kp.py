import numpy as np
import torch


# from core.config import cfg


def get_common_joint_names():
    return [
        "rankle",  # 0  "lankle",    # 0
        "rknee",  # 1  "lknee",     # 1
        "rhip",  # 2  "lhip",      # 2
        "lhip",  # 3  "rhip",      # 3
        "lknee",  # 4  "rknee",     # 4
        "lankle",  # 5  "rankle",    # 5
        "rwrist",  # 6  "lwrist",    # 6
        "relbow",  # 7  "lelbow",    # 7
        "rshoulder",  # 8  "lshoulder", # 8
        "lshoulder",  # 9  "rshoulder", # 9
        "lelbow",  # 10  "relbow",    # 10
        "lwrist",  # 11  "rwrist",    # 11
        "neck",  # 12  "neck",      # 12
        "headtop",  # 13  "headtop",   # 13
    ]


def get_h36m_joint_names():
    """
    I2L 与 Vibe 对 H36M 的节点定义有一些区别
    :return:
    """
    return [
        'hip',  # 0
        'rhip',  # 1
        'rknee',  # 2
        'rankle',  # 3
        'lhip',  # 4
        'lknee',  # 5
        'lankle',  # 6
        'Spine (H36M)',  # 7
        'neck',  # 8
        'Head (H36M)',  # 9
        'headtop',  # 10
        'lshoulder',  # 11
        'lelbow',  # 12
        'lwrist',  # 13
        'rshoulder',  # 14
        'relbow',  # 15
        'rwrist',  # 16
    ]


def get_h36m_pairs():
    return (
        (0, 7), (7, 8), (8, 9), (9, 10),
        (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16),
        (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6)
    )


def get_coco_joint_names():
    """
    这些节点的真实语义（对应spin中的哪个部分）还需要进一步测试
    暂时只能把它当成中间变量（coco_gt -> psuedo_spin_49_gt -> coco_gt **kp_loss** coco_pred <--J_reg-- smpl_pred）
    """
    return [
        'nose',  # 0
        'leye',  # 1
        'reye',  # 2
        'lear',  # 3
        'rear',  # 4
        'lshoulder',  # 5
        'rshoulder',  # 6
        'lelbow',  # 7
        'relbow',  # 8
        'lwrist',  # 9
        'rwrist',  # 10
        'lhip',  # 11
        'rhip',  # 12
        'lknee',  # 13
        'rknee',  # 14
        'lankle',  # 15
        'rankle',  # 16
        'hip'  # 17
    ]


def gene_cvt_map(src, dst):
    """
    fill -1 for joints that doesn't exist
    :param src:
    :param dst:
    :return:
    """
    if src == 'spin' and dst == 'h36m':
        pass
        # print(1)
    if src == 'h36m' and dst == 'spin':
        pass
        # print(1)

    src_names = eval(f'get_{src}_joint_names')()
    dst_names = eval(f'get_{dst}_joint_names')()

    cvt_map = -np.ones((len(dst_names),), dtype=np.int64)

    for idx, jn in enumerate(dst_names):
        if jn in src_names:
            cvt_map[idx] = src_names.index(jn)

    return cvt_map


KP_STYLES = ['h36m', 'coco', ]
KP_MAP = dict()
for src in KP_STYLES:
    for tgt in KP_STYLES:
        if src != tgt:
            KP_MAP[f'{src}_to_{tgt}'] = gene_cvt_map(src, tgt)


####################################

def cvt_kp_old(joints, src, tgt):
    """
    deprecated.
    """
    src_names = eval(f'get_{src}_joint_names')()
    dst_names = eval(f'get_{tgt}_joint_names')()

    B, J_src, C = joints.shape
    out_joints2d = np.zeros((B, len(dst_names), C))

    for idx, jn in enumerate(dst_names):
        if jn in src_names:
            out_joints2d[:, idx] = joints[:, src_names.index(jn)]

    return out_joints2d


def cvt_kp(joints, src, tgt, default=0):
    cvt_map = KP_MAP[f'{src}_to_{tgt}']
    return cvt_kp_from_map(joints, cvt_map, default=default)


def cvt_kp_from_map(src_kp, src_to_tgt_map, default=0):
    """

    :param src_kp:  (..., J_src, C)
    :param src_to_tgt_map: (J_tgt, )
    :param default:
    :return: (..., J_tgt, C)
    """
    J_src, C = src_kp.shape[-2:]
    J_tgt = src_to_tgt_map.shape[0]

    if J_src >= J_tgt:
        return src_kp[..., src_to_tgt_map, :]
    else:
        if isinstance(src_kp, np.ndarray):
            # np.ndarray
            src_kp_with_0 = np.concatenate([src_kp, default * np.ones_like(src_kp[..., [0], :])],
                                           axis=-2)  # (..., J_src + 1, C)
        else:
            # torch.tensor
            src_kp_with_0 = torch.cat([src_kp, default * torch.ones_like(src_kp[..., [0], :])],
                                      dim=-2)  # (..., J_src + 1, C)
        return src_kp_with_0[..., src_to_tgt_map, :]


J_types = ['Human36M', 'MSCOCO']
num_joints = {'Human36M': 17, 'MSCOCO': 18}
joint_names_dict = {
    'Human36M': ['Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle',
                 'Torso', 'Neck', 'Nose', 'Head',  # 7 ~ 10
                 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist']
}
eval_inds_dict = {
    'Human36M': [1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16]
}
metrics_dict = {
    'Human36M': ['mpjpe', 'pa_mpjpe']
}


def cvt_kp_to_full(kp, J_type):
    """
    每次训练前开始，会根据使用到的J_type定义一个新的节点列表
    """
    kp_shape = list(kp.shape)

    # J_types = cfg.DATASETS.Common.J_types
    num_Js = [num_joints[name] for name in J_types]
    cumsum_Js = np.cumsum(num_Js).tolist()
    cumsum_Js = [0] + cumsum_Js
    J_ind = J_types.index(J_type)
    J_start_ind = cumsum_Js[J_ind]

    kp_full_shape = kp_shape.copy()
    kp_full_shape[-2] = cumsum_Js[-1]
    kp_full = np.zeros(kp_full_shape, dtype=np.float32)
    kp_full[J_start_ind:J_start_ind + num_Js[J_ind], :] = kp
    return kp_full


def cvt_full_to_kp(kp, J_type, J_types):
    """
    np.ndarray & torch.tensor
    """

    # J_types = cfg.DATASETS.Common.J_types
    num_Js = [num_joints[name] for name in J_types]
    cumsum_Js = np.cumsum(num_Js).tolist()
    cumsum_Js = [0] + cumsum_Js
    J_ind = J_types.index(J_type)
    J_start_ind = cumsum_Js[J_ind]

    kp_curr = kp[..., J_start_ind:J_start_ind + num_Js[J_ind], :]
    return kp_curr


if __name__ == '__main__':
    h36m_to_common_map = gene_cvt_map('h36m', 'common')
    common_to_h36m_map = gene_cvt_map('common', 'h36m')

    common_kp3d = np.array([i // 3 for i in range(14 * 3)], dtype=np.int64).reshape(14, 3)
    h36m_kp3d = np.array([i // 3 for i in range(17 * 3)], dtype=np.int64).reshape(17, 3)
    common_kp3d_tensor = torch.from_numpy(common_kp3d)
    h36m_kp3d_tensor = torch.from_numpy(h36m_kp3d)

    # common_gene_kp3d = cvt_kp_from_map(h36m_kp3d, h36m_to_common_map)
    # print(common_gene_kp3d)
    #
    # h36m_gene_kp3d = cvt_kp_from_map(common_kp3d, common_to_h36m_map, default=-20)
    # print(h36m_gene_kp3d)
    #
    # common_gene_kp3d_tensor = cvt_kp_from_map(h36m_kp3d_tensor, h36m_to_common_map)
    # print(common_gene_kp3d_tensor)
    #
    # h36m_gene_kp3d_tensor = cvt_kp_from_map(common_kp3d_tensor, common_to_h36m_map, default=-30)
    # print(h36m_gene_kp3d_tensor)

    print(cvt_kp(common_kp3d, 'common', 'h36m'))
    print(cvt_kp(common_kp3d_tensor, 'common', 'h36m'))
    print(cvt_kp(h36m_kp3d, 'h36m', 'common'))
    print(cvt_kp(h36m_kp3d_tensor, 'h36m', 'common'))
