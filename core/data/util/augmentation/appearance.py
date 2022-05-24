import numpy as np

# from core.config import cfg
import core.data.util.augmentation.background as aug_background
import core.data.util.augmentation.color as aug_color
import core.data.util.augmentation.voc_loader as aug_voc_loader
import core.data.util.improc as improc
import core.data.util.util as util


def augment_appearance(im, rng=None, do_color=True, do_occlude=True):
    if rng is None:
        rng = np.random.RandomState(seed=np.random.randint(2 ** 32))
    occlusion_rng = util.new_rng(rng)
    color_rng = util.new_rng(rng)

    occlude_aug_prob = 0.7  # 以0.7的概率做aug
    occlude_types = ['objects', 'random-erase']

    if occlude_aug_prob > 0:
        # todo: 还没有下载 VOC 数据集，没有办法将它作为 occlusion 的来源
        # occlude_type = str(occlusion_rng.choice(['objects', 'random-erase']))
        # occlude_type = str(occlusion_rng.choice(['random-erase']))
        occlude_type = str(occlusion_rng.choice(occlude_types))
    else:
        occlude_type = None

    if occlude_type == 'objects':
        # For object occlusion augmentation, do the occlusion first, then the filtering,
        # so that the occluder blends into the image better.
        if occlusion_rng.uniform(0.0, 1.0) < occlude_aug_prob:
            im = object_occlude(im, occlusion_rng, inplace=True)
        if do_color:
            im = aug_color.augment_color(im, color_rng)
    elif occlude_type == 'random-erase':
        # For random erasing, do color aug first, to keep the random block distributed
        # uniformly in 0-255, as in the Random Erasing paper
        if do_color:
            im = aug_color.augment_color(im, color_rng)
        if occlude_type and occlusion_rng.uniform(0.0, 1.0) < occlude_aug_prob:
            im = random_erase(im, 0, 1 / 3, 0.3, 1.0 / 0.3, occlusion_rng, inplace=True)
    else:
        if do_color:
            im = aug_color.augment_color(im, color_rng)

    return im


def object_occlude(im, rng, inplace=True):
    # Following [Sárándi et al., arxiv:1808.09316, arxiv:1809.04987]

    occlude_aug_scale = 0.8

    factor = im.shape[0] / 224
    count = rng.randint(1, 8)
    occluders = aug_voc_loader.load_occluders()

    for i in range(count):
        occluder, occ_mask = util.choice(occluders, rng)
        rescale_factor = rng.uniform(0.2, 1.0) * factor * occlude_aug_scale

        occ_mask = improc.resize_by_factor(occ_mask, rescale_factor)
        occluder = improc.resize_by_factor(occluder, rescale_factor)

        center = rng.uniform(0, im.shape[0], size=2)
        im = improc.paste_over(occluder, im, alpha=occ_mask, center=center, inplace=inplace)

    return im


def random_erase(im, area_factor_low, area_factor_high, aspect_low, aspect_high, rng, inplace=True):
    # Following the random erasing paper [Zhong et al., arxiv:1708.04896]

    img_side = 224
    occlude_aug_scale = 0.8

    image_area = img_side ** 2

    while True:
        occluder_area = (
                rng.uniform(area_factor_low, area_factor_high) *
                image_area * occlude_aug_scale)
        aspect_ratio = rng.uniform(aspect_low, aspect_high)
        height = (occluder_area * aspect_ratio) ** 0.5
        width = (occluder_area / aspect_ratio) ** 0.5
        pt1 = rng.uniform(0, img_side, size=2)
        pt2 = pt1 + np.array([width, height])
        if np.all(pt2 < img_side):
            pt1 = pt1.astype(int)
            pt2 = pt2.astype(int)
            if not inplace:
                im = im.copy()
            im[pt1[1]:pt2[1], pt1[0]:pt2[0]] = rng.randint(
                0, 255, size=(pt2[1] - pt1[1], pt2[0] - pt1[0], 3), dtype=im.dtype)
            return im
