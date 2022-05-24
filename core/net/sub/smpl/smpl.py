# This script is borrowed and extended from https://github.com/nkolot/SPIN/blob/master/models/hmr.py
# Adhere to their licence to use this script

import joblib

import torch
import numpy as np
import os.path as osp
from smplx import SMPL as _SMPL
from smplx.lbs import vertices2joints

# from psbody.mesh import Mesh

# from core.config import cfg

SMPL_JNAME_24 = [
    # 0~23, 24kps from original SMPL
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2',  # 0~6
    'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot', 'neck',  # 7~12
    'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',  # 13~17
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hand', 'right_hand',  # 18~23
]

SMPL_FLIP_PAIRS_24 = (
    (1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23))

SMPL_JNAME_29 = SMPL_JNAME_24 + [
    # 24~28, 5 face joints, picked from vertices. read smplx.vertex_joint_selector
    'nose', 'right_eye', 'left_eye', 'right_ear', 'left_ear',
]

SMPL_FLIP_PAIRS_29 = (
    (1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23), (25, 26), (27, 28))

SMPL_SKELETON_29 = (
    (0, 1), (1, 4), (4, 7), (7, 10), (0, 2), (2, 5), (5, 8), (8, 11), (0, 3), (3, 6), (6, 9), (9, 14), (14, 17),
    (17, 19), (19, 21), (21, 23), (9, 13), (13, 16), (16, 18), (18, 20), (20, 22), (9, 12), (12, 24), (24, 15),
    (24, 25), (24, 26), (25, 27),
    (26, 28)
)

SMPL_JNAME_45 = SMPL_JNAME_24 + [
    # 24~28, 5kps from face
    'nose', 'right_eye', 'left_eye', 'right_ear', 'left_ear',
    # 29~34, 6kps from feet
    'left_bigtoe', 'left_smalltoe', 'left_heel', 'right_bigtoe', 'right_smalltoe', 'right_heel',
    # 35~44, 10kps from hands
    'left_thumb', 'left_index', 'left_middle', 'left_ring', 'left_pinky',
    'right_thumb', 'right_index', 'right_middle', 'right_ring', 'right_pinky',
]

SMPL_JNAME_54 = SMPL_JNAME_45 + [
    # 45~53, 9kps from VIBE's J_regressor_extra.npy
    'right_hip_ext', 'left_hip_ext', 'neck_ext', 'head_top_ext', 'pelvis_ext',
    'thorax_ext', 'spine_ext', 'jaw_ext', 'head_ext'
]

SPIN_JNAME_49 = [
    'OP Nose', 'OP Neck', 'OP RShoulder',
    'OP RElbow', 'OP RWrist', 'OP LShoulder',
    'OP LElbow', 'OP LWrist', 'OP MidHip',  # no.8
    'OP RHip', 'OP RKnee', 'OP RAnkle',
    'OP LHip', 'OP LKnee', 'OP LAnkle',  # no.14
    'OP REye', 'OP LEye', 'OP REar',
    'OP LEar', 'OP LBigToe', 'OP LSmallToe',
    'OP LHeel', 'OP RBigToe', 'OP RSmallToe', 'OP RHeel',  # no.24
    'Right Ankle', 'Right Knee', 'Right Hip',
    'Left Hip', 'Left Knee', 'Left Ankle',  # no.30
    'Right Wrist', 'Right Elbow', 'Right Shoulder',
    'Left Shoulder', 'Left Elbow', 'Left Wrist',  # no.36
    'Neck (LSP)', 'Top of Head (LSP)',  # no.38
    'Pelvis (MPII)', 'Thorax (MPII)',
    'Spine (H36M)', 'Jaw (H36M)',  # no.42
    'Head (H36M)', 'Nose', 'Left Eye',
    'Right Eye', 'Left Ear', 'Right Ear'
]

SMPL_FLIP_PAIRS_49 = (
    (2, 5), (3, 6), (4, 7), (9, 12), (10, 13), (11, 14),
    (15, 16), (17, 18), (19, 22), (20, 23), (21, 24),
    (25, 30), (26, 29), (27, 28), (31, 36), (32, 35), (33, 34),
    (45, 46), (47, 48),
)

SPIN_JNAME_49_to_SMPL_JID_54 = {
    'OP Nose': 24, 'OP Neck': 12, 'OP RShoulder': 17,
    'OP RElbow': 19, 'OP RWrist': 21, 'OP LShoulder': 16,
    'OP LElbow': 18, 'OP LWrist': 20, 'OP MidHip': 0,
    'OP RHip': 2, 'OP RKnee': 5, 'OP RAnkle': 8,
    'OP LHip': 1, 'OP LKnee': 4, 'OP LAnkle': 7,
    'OP REye': 25, 'OP LEye': 26, 'OP REar': 27,
    'OP LEar': 28, 'OP LBigToe': 29, 'OP LSmallToe': 30,
    'OP LHeel': 31, 'OP RBigToe': 32, 'OP RSmallToe': 33, 'OP RHeel': 34,
    'Right Ankle': 8, 'Right Knee': 5, 'Right Hip': 45,
    'Left Hip': 46, 'Left Knee': 4, 'Left Ankle': 7,
    'Right Wrist': 21, 'Right Elbow': 19, 'Right Shoulder': 17,
    'Left Shoulder': 16, 'Left Elbow': 18, 'Left Wrist': 20,
    'Neck (LSP)': 47, 'Top of Head (LSP)': 48,
    'Pelvis (MPII)': 49, 'Thorax (MPII)': 50,
    'Spine (H36M)': 51, 'Jaw (H36M)': 52,
    'Head (H36M)': 53, 'Nose': 24, 'Left Eye': 26,
    'Right Eye': 25, 'Left Ear': 28, 'Right Ear': 27
}

JOINTS_ID = {i for i in range(len(SMPL_JNAME_29))}
JOINTS_NAME_2_ID = {SMPL_JNAME_29[i]: i for i in JOINTS_ID}
JOINTS_ID_2_NAME = {i: SMPL_JNAME_29[i] for i in JOINTS_ID}


# SMPL_MEAN_PARAMS = osp.join(cfg.NETWORK.SMPL.root_dir, 'smpl_mean_params.npz')


class SMPL(torch.nn.Module):
    """ Extension of the official SMPL implementation to support more joints
        smpl原生返回0~23节点
        额外地：
            脸部返回5个节点：24~28
            脚上返回6个节点:29~34
            手上10个：35:44
        共45个

        J_regressor_extra额外9个
        共54个

        spin从这54个节点中挑选了49个

        ********************************
        smplx==0.1.13 与 torch==1.6.0+cu101 有冲突，转换csc_matrix->tensor时会导致其不contiguous而view报错
        现在使用smplx==0.1.26
    """
    JOINT_TYPES = ('SMPL_24', 'SMPL_29', 'SPIN_49',)
    GENDER_ID_2_NAME = {0: 'neutral', 1: 'male', 2: 'female'}

    def __init__(self,
                 root,
                 only_neutral=False,
                 joint_type='SMPL_24',
                 **kwargs):
        """
        :param joint_type: 支持 SMPL_29 & SPIN_49
        """
        assert joint_type in self.JOINT_TYPES, f'{joint_type} not in legal JOINT_TYPES: {self.JOINT_TYPES}'
        super().__init__()

        self.root = root
        self.joint_type = joint_type
        self.only_neutral = only_neutral

        if self.joint_type in ['SMPL_24', 'SMPL_29']:
            use_hands = False
            use_feet_keypoints = False
        else:
            use_hands = True
            use_feet_keypoints = True

        # 载入不同性别的模型
        self.neutral = _SMPL(model_path=self.root,
                             use_hands=use_hands,
                             use_feet_keypoints=use_feet_keypoints,
                             create_transl=False,
                             gender='neutral',
                             **kwargs)
        if not self.only_neutral:
            self.male = _SMPL(model_path=self.root,
                              use_hands=use_hands,
                              use_feet_keypoints=use_feet_keypoints,
                              create_transl=False,
                              gender='male',
                              **kwargs)
            # self.add_module('male', male)

            self.female = _SMPL(model_path=self.root,
                                use_hands=use_hands,
                                use_feet_keypoints=use_feet_keypoints,
                                create_transl=False,
                                gender='female',
                                **kwargs)
            # self.add_module('female', female)

        if self.joint_type == 'SPIN_49':
            J_regressor_extra = np.load(
                osp.join(self.root, 'J_regressor_extra.npy'))  # 9
            self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra, dtype=torch.float32))
            self.smpl_54_to_spin_49_map = torch.tensor([SPIN_JNAME_49_to_SMPL_JID_54[i] for i in SPIN_JNAME_49],
                                                       dtype=torch.long)

    @property
    def faces(self):
        return self.neutral.faces

    def forward(self, betas, thetas, is_rotmat=True, gender='neutral', **kwargs):
        """
        :param betas:   (B, 10)
        :param thetas:  (B, 24, 3, 3) if is_rotmat == True or (B, 72) if is_rotmat == False
        :return:
        """

        if isinstance(gender, torch.Tensor):
            assert betas.shape[0] == thetas.shape[0] == gender.shape[0]

        if is_rotmat:
            body_pose = thetas[:, 1:]
            global_orient = thetas[:, 0:1]
        else:
            body_pose = thetas[:, 3:]
            global_orient = thetas[:, 0:3]

        if isinstance(gender, str):
            smpl_layer = eval(f'self.{gender}')
            smpl_output = smpl_layer(
                betas=betas.contiguous(),
                body_pose=body_pose.contiguous(),
                global_orient=global_orient.contiguous(),
                pose2rot=not is_rotmat,
                **kwargs
            )
            smpl_vtx = smpl_output.vertices  # (B, 6980, 3)
            smpl_kp3d = smpl_output.joints  # (B, 29or45, 3)
        elif isinstance(gender, torch.Tensor):
            all_inds = []
            all_smpl_vtx = []
            all_smpl_kp3d = []

            for g_id, g_name in self.GENDER_ID_2_NAME.items():
                hit_inds = torch.where(gender == g_id)[0]

                if len(hit_inds) > 0:
                    smpl_layer = eval(f'self.{g_name}')
                    smpl_output = smpl_layer(
                        betas=betas[hit_inds].contiguous(),
                        body_pose=body_pose[hit_inds].contiguous(),
                        global_orient=global_orient[hit_inds].contiguous(),
                        pose2rot=not is_rotmat,
                        **kwargs
                    )
                    smpl_vtx = smpl_output.vertices  # (B_hit, 6980, 3)
                    smpl_kp3d = smpl_output.joints  # (B_hit, 29or45, 3)

                    all_inds.append(hit_inds)
                    all_smpl_vtx.append(smpl_vtx)
                    all_smpl_kp3d.append(smpl_kp3d)

            # 将3种性别的内容直接接在一起
            all_inds = torch.cat(all_inds, dim=0)
            all_smpl_vtx = torch.cat(all_smpl_vtx, dim=0)
            all_smpl_kp3d = torch.cat(all_smpl_kp3d, dim=0)

            # 根据all_inds的排序结果，重新还原排序
            sorted_inds = all_inds.sort().indices
            smpl_vtx = all_smpl_vtx[sorted_inds]
            smpl_kp3d = all_smpl_kp3d[sorted_inds]

        if self.joint_type == 'SPIN_49':
            extra_kp3d_9 = vertices2joints(self.J_regressor_extra, smpl_vtx)  # (B, 9, 3)
            smpl_kp3d = torch.cat([smpl_kp3d, extra_kp3d_9], dim=1)  # (B, 45+9=54, 3)
            smpl_kp3d = smpl_kp3d[:, self.smpl_54_to_spin_49_map, :]  # (B, 49, 3)

        if self.joint_type == 'SMPL_24':
            smpl_kp3d = smpl_kp3d[:, 0:24, :]

        return (
            smpl_vtx,  # (B, J, 3)
            smpl_kp3d,  # (B, V, 3)
        )


# todo: 12112021, make below methods to SMPL class

def get_smpl_faces():
    smpl = SMPL(cfg.NETWORK.SMPL.root_dir, batch_size=1, create_transl=False)
    return smpl.faces


# def load_canonical_sparse_smpl_old():
#     smpl_sparse_obj_path = osp.join(cfg.NETWORK.SMPL.root_dir, cfg.NETWORK.SMPL.smpl_sparse_obj_filename)
#     smpl_431 = Mesh(filename=smpl_sparse_obj_path)
#     return smpl_431


def load_canonical_sparse_smpl_vtx():
    smpl_sparse_vtx_path = osp.join('/home/siyu/windness/proj/HPE/MeshTransformer_windness/data/model/smpl',
                                    'smpl_431_vtx.npy')
    smpl_431_vtx = joblib.load(smpl_sparse_vtx_path)  # (431, 3), in np.float32
    return smpl_431_vtx


def load_upsample_transform():
    smpl_trans_matrix_path = osp.join('/home/siyu/windness/proj/HPE/MeshTransformer_windness/data/model/smpl',
                                      'smpl_6890_to_431_sparse.pt')
    mesh_sampler = joblib.load(smpl_trans_matrix_path)

    # merge 2 sparse matrix to 1 dense matrix
    # todo: load torch sparse matrix
    U_t_sparse_431_to_1723 = mesh_sampler['U_t_sparse_431_to_1723']
    U_t_sparse_1723_to_6890 = mesh_sampler['U_t_sparse_1723_to_6890']
    U_t_sparse_to_mediate = U_t_sparse_431_to_1723.todense().astype(np.float32)
    U_t_mediate_to_full = U_t_sparse_1723_to_6890.todense().astype(np.float32)

    return U_t_mediate_to_full, U_t_sparse_to_mediate


upsample_transforms = load_upsample_transform()


def load_downsample_transform():
    smpl_trans_matrix_path = osp.join('/home/siyu/windness/proj/HPE/MeshTransformer_windness/data/model/smpl',
                                      'smpl_6890_to_431_sparse.pt')
    mesh_sampler = joblib.load(smpl_trans_matrix_path)

    # merge 2 sparse matrix to 1 dense matrix
    # todo: load torch sparse matrix
    D_t_sparse_6890_to_1723 = mesh_sampler['D_t_sparse_6890_to_1723']
    D_t_sparse_1723_to_431 = mesh_sampler['D_t_sparse_1723_to_431']
    D_t_full_to_mediate = D_t_sparse_6890_to_1723.todense().astype(np.float32)
    D_t_mediate_to_sparse = D_t_sparse_1723_to_431.todense().astype(np.float32)

    return D_t_full_to_mediate, D_t_mediate_to_sparse


downsample_transforms = load_downsample_transform()

# global_smpl = SMPL(gender='male') # todo: MALE!!!
global_smpl = SMPL(root='/home/siyu/windness/proj/HPE/MeshTransformer_windness/data/model/smpl')

if __name__ == '__main__':
    smpl = SMPL(joint_type='SMPL_29')
    smpl_output = smpl(
        torch.ones((16, 10), dtype=torch.float32),
        torch.ones((16, 24, 3, 3), dtype=torch.float32),
        gender=torch.from_numpy(np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 0, 1, 2], dtype=np.int64))
    )

    print(1)
