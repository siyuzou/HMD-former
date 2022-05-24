import numpy as np
import torchvision.transforms as transforms
import cv2
import random
# from config import cfg
# from core.config import cfg
import math


def load_img(path, order='RGB'):
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError("Fail to read %s" % path)

    if order == 'RGB':
        img = img[:, :, ::-1].copy()

    img = img.astype(np.float32)
    return img


def get_bbox(joint_img, joint_valid):
    x_img, y_img = joint_img[:, 0], joint_img[:, 1]
    x_img = x_img[joint_valid == 1];
    y_img = y_img[joint_valid == 1];
    xmin = min(x_img);
    ymin = min(y_img);
    xmax = max(x_img);
    ymax = max(y_img);

    x_center = (xmin + xmax) / 2.;
    width = xmax - xmin;
    xmin = x_center - 0.5 * width * 1.2
    xmax = x_center + 0.5 * width * 1.2

    y_center = (ymin + ymax) / 2.;
    height = ymax - ymin;
    ymin = y_center - 0.5 * height * 1.2
    ymax = y_center + 0.5 * height * 1.2

    bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32)
    return bbox


def process_bbox(bbox, img_width, img_height):
    # sanitize bboxes
    x, y, w, h = bbox
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((img_width - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((img_height - 1, y1 + np.max((0, h - 1))))
    if w * h > 0 and x2 >= x1 and y2 >= y1:
        bbox = np.array([x1, y1, x2 - x1, y2 - y1])
    else:
        return None

    # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w / 2.
    c_y = bbox[1] + h / 2.
    # aspect_ratio = cfg.input_img_shape[1]/cfg.input_img_shape[0]
    aspect_ratio = 1
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w * 1.25
    bbox[3] = h * 1.25
    bbox[0] = c_x - bbox[2] / 2.
    bbox[1] = c_y - bbox[3] / 2.

    return bbox


def get_aug_config(exclude_flip):
    scale_factor = 0.25
    rot_factor = 30
    color_factor = 0.2

    scale = np.clip(np.random.randn(), -1.0, 1.0) * scale_factor + 1.0
    rot = np.clip(np.random.randn(), -2.0,
                  2.0) * rot_factor if random.random() <= 0.6 else 0
    c_up = 1.0 + color_factor
    c_low = 1.0 - color_factor
    color_scale = np.array([random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)])
    if exclude_flip:
        do_flip = False
    else:
        do_flip = random.random() <= 0.5

    """ scale:                  -0.75 ~ 0.75
        rot:            40%     0
                        60%     -60 ~ 60
        color_scale:            (0.8 ~ 1.2) * 3
        do_flip:        50%     True
                        50%     False
    """
    return scale, rot, color_scale, do_flip


def augmentation(img, target_shape, bbox, do_aug=True, exclude_flip=False, force=False):
    if do_aug:
        scale, rot, color_scale, do_flip = get_aug_config(exclude_flip)
    else:
        scale, rot, color_scale, do_flip = 1.0, 0.0, np.array([1, 1, 1]), False

    if force:
        scale, rot, color_scale, do_flip = 1.0, 0.0, np.array([1, 1, 1]), False

    img, trans, inv_trans, _ = generate_patch_image(img, bbox, scale, rot, do_flip, target_shape)
    img = np.clip(img * color_scale[None, None, :], 0, 255)
    return img, trans, inv_trans, rot, do_flip


def generate_patch_image(cvimg, bbox, scale, rot, do_flip, out_shape, bbox_type='x0-y0-w-h'):
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape

    if bbox_type == 'x0-y0-w-h':
        bb_c_x = float(bbox[0] + 0.5 * bbox[2])
        bb_c_y = float(bbox[1] + 0.5 * bbox[3])
        bb_width = float(bbox[2])
        bb_height = float(bbox[3])
    elif bbox_type == 'cx-cy-w-h':
        bb_c_x = float(bbox[0])
        bb_c_y = float(bbox[1])
        bb_width = float(bbox[2])
        bb_height = float(bbox[3])

    if do_flip:
        img = img[:, ::-1, :]
        bb_c_x = img_width - bb_c_x - 1

    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot)
    img_patch = cv2.warpAffine(img, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR)
    img_patch = img_patch.astype(np.float32)
    inv_trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot,
                                        inv=True)

    # img2bb_trans that don't consider rot
    img2bb_trans_wo_rot = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0],
                                                  scale, 0)

    return img_patch, trans, inv_trans, img2bb_trans_wo_rot


def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    trans = trans.astype(np.float32)
    return trans


def img_numpy2torch(img, img_preproc):
    if img_preproc in ['0to1', '-1to1']:
        transform = transforms.ToTensor()
    elif img_preproc == 'imagenet':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        assert 0, f'unknown img preproc type: {img_preproc}'

    img = transform(img)

    if img_preproc == '-1to1':
        img = img * 2 - 1

    return img


def img_torch2numpy(img, img_preproc):
    img = img.detach().cpu()
    if img_preproc == 'imagenet':
        inv_transform = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        )

        if len(img.shape) == 3:
            img = inv_transform(img)
        else:
            if False:
                # on rmt2xrtx3090, this is correct
                img = inv_transform(img)
            else:
                # on PC, transforms can only be applied to CxHxW, so do it one-by-one
                for img_ in img:
                    img_[...] = inv_transform(img_)
    elif img_preproc == '-1to1':
        img = (img + 1) / 2
    img = img.clamp(0., 1.)
    img = (img.numpy() * 255.).astype(np.uint8)
    if len(img.shape) == 3:
        img = np.transpose(img, (1, 2, 0))
    elif len(img.shape) == 4:
        img = np.transpose(img, (0, 2, 3, 1))
    return img


def kp_img2norm(kp, img_side=224):
    # (0, 224) -> (-1, 1)
    return kp / img_side * 2. - 1


def kp_norm2img(kp, img_side=224):
    # (-1, 1) -> (0, 224)
    return (kp + 1) / 2 * img_side


def blockwise_mask(num_P, mask_shape=(7, 7), rng=None, do_print=False):
    mask = np.zeros(mask_shape, dtype=np.int64)
    if rng is None:
        rng = np.random.RandomState()
    counter = 0
    while True:
        mask_sum = mask.sum()

        # min_attempt = 16
        min_attempt = np.ceil(np.sqrt(num_P - mask_sum)) ** 2
        if min_attempt == 1:
            a = b = 1
        else:
            curr_num_P = min_attempt if num_P - mask_sum < min_attempt \
                else int(rng.uniform(min_attempt, num_P - mask_sum))
            aspect_ratio = rng.uniform(0.3, 1.0 / 0.3)
            a = int(np.sqrt(curr_num_P * aspect_ratio))
            b = int(np.sqrt(curr_num_P / aspect_ratio))
        t = int(rng.uniform(0, mask_shape[0] - a + 1))
        l = int(rng.uniform(0, mask_shape[1] - b + 1))
        mask[t:t + a, l:l + b] = 1

        counter += 1
        if mask.sum() >= num_P:
            break

    if do_print:
        print(mask)
        print(f'tries: {counter}, sum: {mask.sum()}')
    return mask
