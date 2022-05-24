import torch
import numpy as np

# from core.config import cfg


def cam2pixel(cam_coord, f, c):
    """ P_img = Z^-1 @ K @ P_cam
        此处没有乘 Z^-1，因此 z 值不是1
    """

    # K = np.array([
    #     [float(f[0]), 0, float(c[0])],
    #     [0, float(f[1]), float(c[1])],
    #     [0, 0, 1.],
    # ])
    # img_coord = K @ cam_coord.T
    # return img_coord

    x = cam_coord[:, 0] / cam_coord[:, 2] * f[0] + c[0]
    y = cam_coord[:, 1] / cam_coord[:, 2] * f[1] + c[1]
    z = cam_coord[:, 2]
    return np.stack((x, y, z), 1)


def pixel2cam(pixel_coord, f, c):
    x = (pixel_coord[..., 0] - c[..., [0]]) / f[..., [0]] * pixel_coord[..., 2]
    y = (pixel_coord[..., 1] - c[..., [1]]) / f[..., [1]] * pixel_coord[..., 2]
    z = pixel_coord[..., 2]
    if isinstance(pixel_coord, np.ndarray):
        return np.stack((x, y, z), axis=-1)
    elif isinstance(pixel_coord, torch.Tensor):
        return torch.stack((x, y, z), dim=-1)
    else:
        assert 0, 'type error'



def world2cam(world_coord, R, t):
    # I2L-MeshNet: R @ P + t
    # cam_coord = np.dot(R, world_coord.transpose(1, 0)).transpose(1, 0) + t.reshape(1, 3)

    # raw: R @ (P - t)
    cam_coord = (world_coord - t) @ R.T
    return cam_coord


def cam2world(cam_coord, R, t):
    world_coord = np.dot(np.linalg.inv(R), (cam_coord - t.reshape(1, 3)).transpose(1, 0)).transpose(1, 0)
    return world_coord


def rigid_transform_3D(A, B):
    n, dim = A.shape
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    H = np.dot(np.transpose(A - centroid_A), B - centroid_B) / n
    U, s, V = np.linalg.svd(H)
    R = np.dot(np.transpose(V), np.transpose(U))
    if np.linalg.det(R) < 0:
        s[-1] = -s[-1]
        V[2] = -V[2]
        R = np.dot(np.transpose(V), np.transpose(U))

    varP = np.var(A, axis=0).sum()
    c = 1 / varP * np.sum(s)

    t = -np.dot(R, np.transpose(centroid_A)) + np.transpose(centroid_B)
    return c, R, t


def rigid_align(A, B):
    c, R, t = rigid_transform_3D(A, B)
    A2 = np.transpose(np.dot(c * R, np.transpose(A))) + t
    return A2


def transform_joint_to_other_db(src_joint, src_name, dst_name):
    """
    将 {h36m, mscoco, pw3d} 的 joints 转换成 smpl 的 joints
    依据是前者与后者的相同语义节点
    前者有、后者没有的，置0
    """
    src_joint_num = len(src_name)
    dst_joint_num = len(dst_name)

    new_joint = np.zeros(((dst_joint_num,) + src_joint.shape[1:]), dtype=np.float32)
    for src_idx in range(len(src_name)):
        name = src_name[src_idx]
        if name in dst_name:
            dst_idx = dst_name.index(name)
            new_joint[dst_idx] = src_joint[src_idx]

    return new_joint


h36m_joints_name = (
    'Pelvis',
    'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle',
    'Torso', 'Neck', 'Nose', 'Head_top',
    'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist'
)
smpl_joints_name = (
    'Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee',
    'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest',
    'L_Toe', 'R_Toe', 'Neck', 'L_Thorax', 'R_Thorax',
    'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow',
    'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand', 'Nose',
    'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear'
)


def transform_smpl_to_h36m(src_joint):
    src_joint_num = len(smpl_joints_name)
    dst_joint_num = len(h36m_joints_name)

    new_joint = np.zeros(((dst_joint_num,) + src_joint.shape[1:]), dtype=np.float32)
    for src_idx in range(len(smpl_joints_name)):
        name = smpl_joints_name[src_idx]

        # 暂时将 smpl 的 Head 与 h36m 的 Head_top 对应
        # todo: 这样的处理正确吗？
        if name == 'Head':
            name = 'Head_top'

        if name in h36m_joints_name:
            dst_idx = h36m_joints_name.index(name)
            new_joint[dst_idx] = src_joint[src_idx]

    return new_joint


def iimg2ohm(coord_input_img, z_axis=True):
    """
    x, y-axis:
        input img space --> output heatmap space
        (0, 256) --> (0, 64)
    z-axis:
        voxelize
        (0, 1) --> (0, 64)
    :param coord_input_img:     (J, 3) or (B, J, 3)
    :return coord_output_hm:    (J, 3) or (B, J, 3)
    """
    coord_output_hm = np.zeros_like(coord_input_img)
    coord_output_hm[..., 0] = coord_input_img[..., 0] \
                              / cfg.DATASETS.common.input_img_shape[1] * cfg.DATASETS.common.output_hm_shape[2]
    coord_output_hm[..., 1] = coord_input_img[..., 1] \
                              / cfg.DATASETS.common.input_img_shape[0] * cfg.DATASETS.common.output_hm_shape[1]
    if z_axis:
        coord_output_hm[..., 2] = coord_input_img[..., 2] * cfg.DATASETS.common.output_hm_shape[0]
    return coord_output_hm


def ohm2iimg(coord_output_hm):
    """
    x, y-axis:
        output heatmap space --> input img space
        (0, 64) --> (0,256)
    z-axis:
        devoxelize
        (0, 64) --> (0, 1)
    :param coord_output_hm:     (J, 3) or (B, J, 3)
    :return coord_input_img:    (J, 3) or (B, J, 3)
    """
    coord_input_img = np.zeros_like(coord_output_hm)
    coord_input_img[..., 0] = coord_output_hm[..., 0] \
                              * cfg.DATASETS.common.input_img_shape[1] / cfg.DATASETS.common.output_hm_shape[2]
    coord_input_img[..., 1] = coord_output_hm[..., 1] \
                              * cfg.DATASETS.common.input_img_shape[0] / cfg.DATASETS.common.output_hm_shape[1]
    coord_input_img[..., 2] = coord_output_hm[..., 2] / cfg.DATASETS.common.output_hm_shape[0]
    return coord_input_img


def img2ndc(coord_input_img, z_axis=True, z_in_meter=False):
    """
    img space -> ndc space
        x, y-axis:  (0, 256) -> (-1, 1)
        z-axis:     (-1m, 1m) -> (-bbox_3d_size/2, bbox_3d_size/2) -> (-1, 1)
    :param coord_input_img:     (J, 3) or (B, J, 3)
    :return coord_output_ndc:    (J, 3) or (B, J, 3)
    """
    coord_output_ndc = np.zeros_like(coord_input_img)
    coord_output_ndc[..., 0] = 2 * coord_input_img[..., 0] / cfg.DATASETS.common.input_img_shape[1] - 1.0
    coord_output_ndc[..., 1] = 2 * coord_input_img[..., 1] / cfg.DATASETS.common.input_img_shape[0] - 1.0
    if z_axis:
        # coord_output_ndc[..., 2] = 2 * coord_input_img[..., 2] / cfg.DATASETS.common.bbox_3d_size
        if z_in_meter:
            # already as (-1, 1)
            pass
        else:
            # (-1000, 1000) -> (-1, 1)
            coord_output_ndc[..., 2] = coord_input_img[..., 2] / 1000
    return coord_output_ndc


def ndc2img(coord_output_ndc):
    """
    ndc space -> img space
        x, y-axis:  (-1, 1) -> (0, 224)
        z-axis:     (-1, 1) -> (-bbox_3d_size/2, bbox_3d_size/2)
    :param coord_output_ndc:     (J, 3) or (B, J, 3)
    :return coord_input_img:    (J, 3) or (B, J, 3)
    """
    if isinstance(coord_output_ndc, np.ndarray):
        coord_input_img = np.zeros_like(coord_output_ndc)
    elif isinstance(coord_output_ndc, torch.Tensor):
        coord_input_img = torch.zeros_like(coord_output_ndc)
    else:
        assert 0, f'type error'

    coord_input_img[..., 0] = (coord_output_ndc[..., 0] + 1.0) / 2 * cfg.DATASETS.common.input_img_shape[1]
    coord_input_img[..., 1] = (coord_output_ndc[..., 1] + 1.0) / 2 * cfg.DATASETS.common.input_img_shape[0]
    coord_input_img[..., 2] = coord_output_ndc[..., 2] / 2 * cfg.DATASETS.common.bbox_3d_size
    return coord_input_img
