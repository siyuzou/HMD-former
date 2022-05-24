import numpy as np
import torch


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def avg(self):
        return self.sum / self.count


class MetricsMeter(object):

    def __init__(self, preserve_raw_data=False):
        self.metrics = dict()
        self.preserve_raw_data = preserve_raw_data
        if preserve_raw_data:
            self.metrics_raw = dict()

    def update(self, new_results, batchsize):
        for k, v in new_results.items():
            if k.endswith('_raw'):
                continue
            if k not in self.metrics:
                self.metrics[k] = AverageMeter()
            self.metrics[k].update(v, batchsize)

        if self.preserve_raw_data:
            for k, v in new_results.items():
                if not k.endswith('_raw'):
                    continue
                if k not in self.metrics_raw:
                    self.metrics_raw[k] = list()
                self.metrics_raw[k].append(v)

    def get_results(self):
        results = dict()
        for k, v in self.metrics.items():
            v_mean = v.avg()
            if isinstance(v_mean, torch.Tensor):
                results[k] = v_mean.detach().cpu().numpy()
            else:
                results[k] = v_mean
        return results

    def get_raw_results(self):
        assert self.preserve_raw_data

        results = dict()
        for k, v in self.metrics_raw.items():
            if isinstance(v[0], torch.Tensor):
                v = torch.cat(v, dim=0)
                results[k] = v.detach().cpu().numpy()
            else:
                v = np.concatenate(v, axis=0)
                results[k] = v
        return results

def evaluate_one_batch(outs, dataset_name='Human36M', preserve_raw=False):
    metrics = ['mpjpe', 'pa_mpjpe']
    results = dict()
    batchsize = outs[list(outs.keys())[0]].shape[0]

    for metric in metrics:
        if 'mpjpe' == metric:
            kp3d_gt_rootrel_eval = outs['kp3d_gt'].unsqueeze(dim=1)  # (B, 1, J, 3)
            kp3d_pred_rootrel_eval = outs['kp3d_pred']  # (B, L, J, 3)
            L = kp3d_pred_rootrel_eval.shape[1]

            mpjpe_raw = torch.sqrt(((kp3d_pred_rootrel_eval - kp3d_gt_rootrel_eval) ** 2).sum(dim=-1))  # (B, L, J)
            mpjpe_raw *= 1000  # meters to millimeters
            mpjpe = mpjpe_raw.mean(dim=0)  # (L, J)

            results[metric] = mpjpe  # (L, J_eval, )
            results[f'{metric}_mean'] = mpjpe.mean(dim=-1)  # (L, )
            if preserve_raw:
                results[f'{metric}_raw'] = mpjpe_raw.mean(dim=-1)  # (B, L)

        if 'pa_mpjpe' == metric:
            kp3d_pred_cam_eval_A = batch_compute_similarity_transform_torch(
                kp3d_pred_rootrel_eval, kp3d_gt_rootrel_eval.expand(-1, L, -1, -1))
            pa_mpjpe_raw = torch.sqrt(((kp3d_pred_cam_eval_A - kp3d_gt_rootrel_eval) ** 2).sum(dim=-1))  # (B, L, J)
            pa_mpjpe_raw *= 1000  # meters to millimeters
            pa_mpjpe = pa_mpjpe_raw.mean(dim=0)  # (L, J)

            results[metric] = pa_mpjpe  # (L, J)
            results[f'{metric}_mean'] = pa_mpjpe.mean(dim=-1)  # (L, )
            if preserve_raw:
                results[f'{metric}_raw'] = pa_mpjpe_raw.mean(dim=-1)  # (B, L)

        if 'mpve' == metric:
            vtx3d_gt_rootrel_eval = outs['vtx3d_gt'].unsqueeze(dim=1)  # (B, 1, V, 3)
            vtx3d_pred_rootrel_eval = outs['vtx3d_pred']  # (B, L, V, 3)

            mpve_raw = torch.sqrt(((vtx3d_pred_rootrel_eval - vtx3d_gt_rootrel_eval) ** 2).sum(dim=-1)).mean(2)
            mpve_raw *= 1000  # meters to millimeters, (B, L)
            mpve = mpve_raw.mean(0)  # (L, )
            results[f'{metric}_mean'] = results[metric] = mpve  # (L, )
            if preserve_raw:
                results[f'{metric}_raw'] = mpve_raw  # (B, L)

    return results, batchsize

def batch_compute_similarity_transform_torch(S1, S2):
    """
    borrowed from VIBE
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.

    :arg S1: (B, J, C) or (B, C, J)
    :arg S2: (B, J, C) or (B, C, J)
    C = 2 or 3
    J > 3
    """

    if len(S1.shape) == 4:
        pre_flatten = True
        B, L, J, C = S1.shape
        S1 = S1.flatten(0, 1)
        S2 = S2.flatten(0, 1)
    else:
        pre_flatten = False

    transposed = False
    if S1.shape[1] != 3 and S1.shape[1] != 2:
        S1 = S1.permute(0, 2, 1)
        S2 = S2.permute(0, 2, 1)
        transposed = True
    assert (S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1 ** 2, dim=1).sum(dim=1)

    # 3. The outer product of X1 and X2.
    K = X1.bmm(X2.permute(0, 2, 1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0], 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0, 2, 1))))

    # Construct R.
    R = V.bmm(Z.bmm(U.permute(0, 2, 1)))

    # 5. Recover scale.
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

    # 6. Recover translation.
    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

    # 7. Error:
    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

    if transposed:
        S1_hat = S1_hat.permute(0, 2, 1)

    if pre_flatten:
        S1_hat = S1_hat.reshape(B, L, J, C)

    return S1_hat


def generate_results_txt(results, dataset_name='Human36M'):
    from core.util.data_util.kp import joint_names_dict, eval_inds_dict, metrics_dict

    L = results[list(results.keys())[0]].shape[0]
    txt = []
    for l in range(L):
        results_curr_layer = {k: v[l] for k, v in results.items()}
        joint_names = np.array(joint_names_dict[dataset_name])
        joint_eval_names = joint_names[np.array(eval_inds_dict[dataset_name])].tolist()
        all_metrics = metrics_dict[dataset_name]
        txt_results_curr_layer = get_per_joint_metrics_txt(results=results_curr_layer,
                                                           metric_names=all_metrics,
                                                           joint_names=joint_eval_names)

        txt.append(f'results for layer {l}:')
        txt.append(txt_results_curr_layer)
    return '\n'.join(txt)


def get_per_joint_metrics_txt(results, metric_names, joint_names=None):
    num_J = results[metric_names[0]].shape[0]
    if joint_names is None:
        joint_names = [f'J_{j_id}' for j_id in range(num_J)]

    # txt args
    j_id_width = 4
    j_name_width = 14
    metric_width = 10
    lines = []

    # print title
    line_fmt = f'{{:>{j_id_width}s}} {{:<{j_name_width}s}} ' + f'{{:>{metric_width}s}}' * len(metric_names)
    line_len = j_id_width + 1 + j_name_width + 1 + metric_width * len(metric_names)
    line_args = ['No.J', 'J_name'] + [metric_name.upper() for metric_name in metric_names]
    lines.append(line_fmt.format(*line_args))
    lines.append('-' * line_len)

    # print metric result per joint
    line_fmt = f'{{:>{j_id_width}s}} {{:<{j_name_width}s}} ' + f'{{:>{metric_width}.4f}}' * len(metric_names)
    for j_id in range(num_J):
        line_args = [str(j_id), str(joint_names[j_id])]
        for name in metric_names:
            metric = results[name]
            if len(metric.shape) == 0:
                line_args.append(float('nan'))
            else:
                line_args.append(metric[j_id])
        lines.append(line_fmt.format(*line_args))
    lines.append('-' * line_len)

    # print mean metric
    line_args = ['', 'mean']
    # for name, metric in results.items():
    for name in metric_names:
        metric = results[name]
        if len(metric.shape) == 0:
            line_args.append(metric)
        else:
            line_args.append(metric.mean())
    lines.append(line_fmt.format(*line_args))

    txt = '\n'.join(lines)
    return txt
