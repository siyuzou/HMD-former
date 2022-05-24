import os.path as osp
import joblib

from easydict import EasyDict as edict
import numpy as np
import cv2
import torch
from core.util.data_util.preprocessing import img_numpy2torch, kp_img2norm

from core.util.exe_util.logger import logger
import core.data.util.util as util
import core.data.util.improc as improc
from core.data.util.augmentation.appearance import augment_appearance
from core.net.sub.smpl.smpl import global_smpl, downsample_transforms
from core.util.data_util.preprocessing import generate_patch_image, blockwise_mask
from core.util.data_util.transforms import world2cam, cam2pixel
from core.util.data_util.kp import cvt_kp_to_full
import transforms3d

H36M_J17_FLIP_PAIRS = ((1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13))

def load_h36m_anno(root, data_split):
    anno_path = f'{root}/h36m_test_frontal.anno'
    assert osp.exists(anno_path), f'{anno_path} not exsists.'
    anno_dict = joblib.load(anno_path)

    print('re-arrange h36m kp order.')
    kp3d_w = anno_dict['kp3d_w']  # (B, 17, 3)
    kp3d_w = kp3d_w[:, [16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], :]
    anno_dict['kp3d_w'] = kp3d_w

    return anno_dict


def get_single_sample(anno, root_dir, data_split):
    img_path = anno['img_path'] # "./data/dataset/Human3.6M/S9/Images_downscaled/..."
    img_path = osp.join(root_dir, osp.relpath(img_path, 'data/dataset/Human3.6M'))
    bbox = anno['bbox']  # x0, y0, w, h. w != h
    # pre-process bbox to make sure that w == h
    bbox_center = bbox[0:2] + bbox[2:4] / 2
    bbox[2:4] = bbox_side = max(bbox[2], bbox[3])
    bbox[0:2] = bbox_center - bbox_side / 2
    kp3d_wld = anno['kp3d_wld']
    cam = anno['cam']
    R, t, f, c = cam['R'], cam['t'], cam['f'], cam['c']

    ##### prepare param.
    # common
    img_side = 224
    img_preproc = 'imagenet'
    root_ind = 0
    scale = 1.0
    rot = 0
    do_flip = False

    ##### img
    img_ori = improc.imread_jpeg(img_path)
    img_ori_shape = img_ori.shape[0:2]
    img, img2bb_trans, bb2img_trans, img2bb_trans_wo_rot = \
        generate_patch_image(img_ori, bbox, scale, rot, do_flip, (img_side, img_side))
    img = img.astype(np.uint8)
    # np -> torch
    img = img_numpy2torch(img, img_preproc)

    ##### kp3d & kp2d
    ### init kp3d_cam & kp2d_img
    kp3d_cam = world2cam(kp3d_wld, R, t)  # rot
    kp2d_img = cam2pixel(kp3d_cam, f, c)[:, 0:2]
    ### kp2d: img-to-bbox trans, considering: rot (already considered in img2bb_trans, so no need to consider rot_aug) + scale
    """ x, y-axis: img-to-bbox affine trans """
    kp2d_img_xy1 = np.concatenate((kp2d_img[:, :2], np.ones_like(kp2d_img[:, :1])), 1)
    kp2d_img = (img2bb_trans @ kp2d_img_xy1.T).T  # (17, 2) = ((2, 3) @ (3, 17)).T
    ### kp3d: rot
    rot_mat = transforms3d.euler.euler2mat(0, 0, rot * np.pi / 180, 'ryxz').T.astype(np.float32)
    kp3d_cam = kp3d_cam @ rot_mat.T

    ### kp post-proc
    kp2d_img = kp_img2norm(kp2d_img, img_side=224)  # (0, 224) -> (-1, 1)
    kp3d_cam = kp3d_cam / 1000.  # milimeters to meters
    # kp3d root-rel
    kp3d_cam = kp3d_cam - kp3d_cam[[root_ind]]

    ##### kp_valid
    kp_valid = ~np.any(np.isnan(kp3d_cam), axis=-1, keepdims=True)  # (17, 1)

    ##### cvt kp to full
    kp3d_cam = cvt_kp_to_full(kp3d_cam, J_type='Human36M')
    kp2d_img = cvt_kp_to_full(kp2d_img, J_type='Human36M')
    kp_valid = cvt_kp_to_full(kp_valid, J_type='Human36M')

    sample = {
        'img': img,
        'kp3d_cam': kp3d_cam,  # (J, 3)
        'kp2d_img': kp2d_img,  # (J, 2)
        'kp_valid': kp_valid,  # (J, 1)
        'gender': 0  # 0 for neutral
    }

    return sample


class Human36M_test(torch.utils.data.Dataset):
    def __init__(self, root, split='valid', *args, **kwargs):

        assert split == 'valid'

        self.root_dir = root
        self.data_split = split

        self._args = args
        self._kwargs = kwargs

        self.anno_dict = load_h36m_anno(root=root, data_split=split)
        logger.info(f'{len(self.anno_dict["img_path"])} samples loaded for h36m.')

    def __len__(self) -> int:
        return len(self.anno_dict['img_path'])

    def __getitem__(self, index):
        img_path = self.anno_dict['img_path'][index]
        bbox = self.anno_dict['bbox'][index]
        kp3d_wld = self.anno_dict['kp3d_w'][index]
        cam = {
            'R': self.anno_dict['camera.R'][index],
            't': self.anno_dict['camera.t'][index],
            'f': self.anno_dict['camera.intrinsic_matrix'][index][[0, 1], [0, 1]],
            'c': self.anno_dict['camera.intrinsic_matrix'][index][0:2, 2],
        }

        anno = {
            'img_path': img_path,
            'bbox': bbox,
            'kp3d_wld': kp3d_wld,
            'cam': cam,
        }

        sample = get_single_sample(anno, self.root_dir, self.data_split)
        sample['id'] = np.int64(index)

        return sample
