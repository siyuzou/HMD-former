from functools import reduce, wraps

import numpy as np
import torch


#################### Tensor Sanity Check ####################
def direct_check_tensor_nan_inf(x):
    x = x.detach()
    with torch.no_grad():
        return x.isnan().sum() > 0


def check_result_nan_inf(f):
    """
        check whether the result of a function f contains nan or inf value.
        only for tensor result or tuple/list result (whose element is tensor)
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        result = f(*args, **kwargs)

        # todo: implement a global flag container
        if True:
            return result

        def check_single_tensor_nan(x):
            x = x.detach()
            with torch.no_grad():
                return x.isnan().sum() > 0

        def check_single_tensor_inf(x):
            x = x.detach()
            with torch.no_grad():
                return x.isinf().sum() > 0

        if isinstance(result, torch.Tensor):
            if check_single_tensor_nan(result):
                print(f'{f.__name__}(): returns nan value.')
            if check_single_tensor_inf(result):
                print(f'{f.__name__}(): returns inf value.')
        if isinstance(result, (list, tuple)):
            for i, item in enumerate(result):
                if check_single_tensor_nan(item):
                    print(f'{f.__name__}(): returns nan value at position {i}.')
                if check_single_tensor_inf(item):
                    print(f'{f.__name__}(): returns inf value at position {i}.')
        return result

    return wrapper


#################### Tensor Sanity Check ####################


@check_result_nan_inf
def weighted_L1_loss(kp_gt, kp_pred, weighing, scale=1.0, eps=1e-5):
    """
    :param kp_gt:   (B, J, C)
    :param kp_pred: (B, J, C)
    :param weighing:(B, J)
    :return:
    """
    loss = weighted_mean((kp_gt - kp_pred).abs(), weighing, eps=eps) * scale
    return loss


@check_result_nan_inf
def unsqueeze_axis(input_tensor, axis):
    if not isinstance(axis, (tuple, list)):
        axis = [axis]
    for ax in axis:
        input_tensor = input_tensor.unsqueeze(ax)
    return input_tensor


@check_result_nan_inf
def broadcast_like(input_tensor, target_tensor):
    return input_tensor + torch.zeros_like(target_tensor, dtype=input_tensor.dtype)


@check_result_nan_inf
def weighted_mean(x, weighing, axis=None, keepdim=False, eps=1e-5):
    n_new_dims = len(x.shape) - len(weighing.shape)
    weighing = unsqueeze_axis(weighing, [-1] * n_new_dims)
    weighing = broadcast_like(weighing, x)

    if axis is None:
        mean = (x * weighing).sum() / (weighing.sum() + eps)
    else:
        mean = (x * weighing).sum(dim=axis, keepdim=keepdim) / (weighing.sum(dim=axis, keepdim=keepdim) + eps)

    return mean


@check_result_nan_inf
def weighted_sum(x, weighing, axis=None, keepdim=False):
    n_new_dims = len(x.shape) - len(weighing.shape)
    weighing = unsqueeze_axis(weighing, [-1] * n_new_dims)
    weighing = broadcast_like(weighing, x)

    sum = (x * weighing).sum(dim=axis, keepdim=keepdim)
    return sum


@check_result_nan_inf
def weighted_mean_stdev(input_tensor, weighing, items_axis, dimensions_axis, fixed_ref=None, eps=1e-5):
    """
    :param input_tensor:    (B, J, C)
    :param weighing:        (B, J)
    :param items_axis:      1
    :param dimensions_axis: 2
    """
    n_new_dims = len(input_tensor.shape) - len(weighing.shape)  # 1
    weighing = unsqueeze_axis(weighing, [-1] * n_new_dims)  # (B, J, 1)
    weighing_b = broadcast_like(weighing, input_tensor)  # (B, J, C)

    if fixed_ref is not None:
        mean = fixed_ref
    else:
        mean = weighted_mean(input_tensor, weighing_b, axis=items_axis, keepdim=True)  # (B, 1, C)
    centered = input_tensor - mean  # (B, J, C)

    denominator = weighing.sum(dim=items_axis, keepdim=True)  # (B, 1, 1)
    numerator = weighted_sum(centered ** 2, weighing_b, axis=[items_axis, dimensions_axis], keepdim=True)  # (B, 1, 1)

    stdev = torch.sqrt(numerator / (denominator + eps))
    return mean, stdev


@check_result_nan_inf
def align_2d_skeletons(kp2d_gt, kp3d_pred_xy, kp_valid, eps=1e-5):
    """
    :param kp2d_gt:         (B, J, 2)
    :param kp3d_pred_xy:    (B, J, 2)
    :param kp_valid:        (B, J, 1)
    :return:
    """

    mean_pred, stdev_pred = weighted_mean_stdev(
        kp3d_pred_xy, kp_valid[:, :, 0], items_axis=1, dimensions_axis=2)  # (B, 1, 2), (B, 1, 1)
    mean_gt, stdev_gt = weighted_mean_stdev(
        kp2d_gt, kp_valid[:, :, 0], items_axis=1, dimensions_axis=2)
    kp2d_pred_result = (kp3d_pred_xy - mean_pred) / (stdev_pred + eps) * stdev_gt + mean_gt

    return kp2d_pred_result, (mean_pred, stdev_pred, mean_gt, stdev_gt)


@check_result_nan_inf
def amax(x, axis=None, keepdim=True):
    """ max on multiple axis, only returns value """
    if axis is None:
        return x.max()

    ndims = len(x.shape)
    axis = [d if d >= 0 else d + ndims for d in axis]
    axis.sort()

    # todo: permute the target axis to the end

    # flatten the axis
    axis_depth_total = reduce(lambda x, y: x * y, (x.shape[ax] for ax in axis))
    flat_shape = [*x.shape[0:ndims - len(axis)], axis_depth_total]
    x = x.flatten().reshape(flat_shape)

    # calculate max on the last dim
    max = x.max(dim=-1, keepdim=keepdim).values
    return max


@check_result_nan_inf
def softmax(target, axis=-1):
    """ only works when axes are at the end """
    # todo: 取了exp后，有的值变成了inf，比如一个e^95，如何限制？？？
    with torch.no_grad():
        target_max = amax(target, axis=axis)
        n_new_dims = len(target.shape) - len(target_max.shape)
        target_max = unsqueeze_axis(target_max, [-1] * n_new_dims)

    exponentiated = torch.exp(target - target_max)
    normalizer_denominator = torch.sum(exponentiated, dim=axis, keepdim=True)
    result = exponentiated / normalizer_denominator

    return result


@check_result_nan_inf
def soft_argmax(x, axis):
    """ borrowed from MeTRabs tfu.py
    """

    input_shape = x.shape
    ndims = len(input_shape)
    dtype = x.dtype
    device = x.device

    softmaxed = softmax(x, axis=axis)

    def relative_coords_along_axis(ax):
        grid_shape = [1] * ndims
        grid_shape[ax] = input_shape[ax]
        grid = torch.linspace(0.0, 1.0, input_shape[ax]).reshape(grid_shape)
        return grid.to(dtype).to(device)

    ##### Single axis
    if not isinstance(axis, (tuple, list)):
        return (relative_coords_along_axis(axis) * softmaxed).sum(dim=axis)

    ##### Multiple axes
    # Convert negative axes to the corresponding positive index (e.g. -1 means last axis)
    heatmap_axes = [ax if ax >= 0 else ndims + ax + 1 for ax in axis]
    other_axes = [other_ax for other_ax in range(ndims) if other_ax not in heatmap_axes]
    result = []
    for ax in heatmap_axes:
        other_heatmap_axes = [other_ax for other_ax in heatmap_axes if other_ax != ax]
        summed_over_other_axes = torch.sum(softmaxed, dim=other_heatmap_axes, keepdim=True)  # (B, J, W, 1, 1)
        coords = relative_coords_along_axis(ax)  # (1, 1, W, 1, 1)
        decoded = torch.sum(coords * summed_over_other_axes, dim=ax, keepdim=True)  # (B, J, 1, 1, 1)
        result.append(decoded.reshape([input_shape[i] for i in other_axes]))  # (B, J)

    result = torch.stack(result, dim=-1)  # (B, J, 3)

    return result


class dropout_layer(torch.nn.Module):
    def __init__(self, p):
        super().__init__()
        assert 0 <= p <= 1
        self.p = p

    def forward(self, x):
        if not self.training:
            return x

        if self.p == 1:
            return torch.zeros_like(x)
        if self.p == 0:
            return x
        mask = (torch.rand(x.shape, device=x.device) > self.p).float()
        return mask * x / (1.0 - self.p)


def orthographic_projection(X, camera):
    """Perform orthographic projection of 3D points X using the camera parameters, ( + t) * s
    Args:
        X: size = [B, N, 3]
        camera: size = [B, 3]
    Returns:
        Projected 2D points -- size = [B, N, 2]
    """
    camera = camera.reshape(-1, 1, 3)
    X_trans = X[:, :, :2] + camera[:, :, 1:]
    shape = X_trans.shape  # (B, N, 2)
    X_2d = (camera[:, :, 0] * X_trans.reshape(shape[0], -1)).reshape(shape)
    return X_2d


def make_inv_bilinear_mask_pt(points, hm_side=7):
    """ 使用参考 demo/MSInc_attn_mask_demo/MSInc_attn_mask_demo.py
    :param points:  (B, N, 2)
    :param hm_side: int
    :return:
    """

    B, N, _ = points.shape
    inter = 1 / (hm_side - 1).to(torch.float32)
    axis = torch.linspace(-inter, 1 + inter, hm_side + 2, dtype=torch.float32, device=points.device)
    output_mask = torch.zeros((B, N, hm_side + 2, hm_side + 2), dtype=torch.float32, device=points.device)

    mask = (points[..., 0] >= -inter) & (points[..., 0] < 1 + inter) & \
           (points[..., 1] >= -inter) & (points[..., 1] < 1 + inter)  # (B, N)

    x0y0 = torch.floor(points * (hm_side - 1)).to(torch.int64)  # (B, N, 2)
    # x0, y0 = torch.split(x0y0.permute(0, 2, 1).reshape(B, -1), 2, dim=-1)  # 2 * (B, N)
    x0, y0 = torch.split(x0y0.permute(0, 2, 1).reshape(B, -1), N, dim=-1)  # 2 * (B, N)
    x1, y1 = x0 + 1, y0 + 1

    # 过一遍 mask，将超出范围的点的索引设为0，防止越界
    x0[mask == 0] = 0
    x1[mask == 0] = 0
    y0[mask == 0] = 0
    y1[mask == 0] = 0

    # 确定N个点在2D空间中上下左右四个关键点的值
    axis_x0 = axis[x0.reshape(-1) + 1].reshape(B, N)  # (B, N)
    axis_x1 = axis[x1.reshape(-1) + 1].reshape(B, N)
    axis_y0 = axis[y0.reshape(-1) + 1].reshape(B, N)
    axis_y1 = axis[y1.reshape(-1) + 1].reshape(B, N)

    # 将反双线性插值的结果填入 output_mask
    batch_axis = torch.arange(0, B, dtype=torch.int64, device=points.device)[..., None]  # (B, 1)
    point_axis = torch.arange(0, N, dtype=torch.int64, device=points.device)[None, ...]  # (1, N)
    output_mask[batch_axis, point_axis, y0 + 1, x0 + 1] = \
        mask * torch.abs((points[..., 0] - axis_x1) * (points[..., 1] - axis_y1)) / inter ** 2
    output_mask[batch_axis, point_axis, y1 + 1, x0 + 1] = \
        mask * torch.abs((points[..., 0] - axis_x1) * (points[..., 1] - axis_y0)) / inter ** 2
    output_mask[batch_axis, point_axis, y0 + 1, x1 + 1] = \
        mask * torch.abs((points[..., 0] - axis_x0) * (points[..., 1] - axis_y1)) / inter ** 2
    output_mask[batch_axis, point_axis, y1 + 1, x1 + 1] = \
        mask * torch.abs((points[..., 0] - axis_x0) * (points[..., 1] - axis_y0)) / inter ** 2

    # output_mask[:, list(range(N)), y0 + 1, x0 + 1] = \
    #     mask * torch.abs((points[..., 0] - axis_x1) * (points[..., 1] - axis_y1)) / inter ** 2
    # output_mask[:, list(range(N)), y1 + 1, x0 + 1] = \
    #     mask * torch.abs((points[..., 0] - axis_x1) * (points[..., 1] - axis_y0)) / inter ** 2
    # output_mask[:, list(range(N)), y0 + 1, x1 + 1] = \
    #     mask * torch.abs((points[..., 0] - axis_x0) * (points[..., 1] - axis_y1)) / inter ** 2
    # output_mask[:, list(range(N)), y1 + 1, x1 + 1] = \
    #     mask * torch.abs((points[..., 0] - axis_x0) * (points[..., 1] - axis_y0)) / inter ** 2

    return output_mask[:, :, 1:-1, 1:-1]
