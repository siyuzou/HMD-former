import time
import argparse
import os
import os.path as osp
import random
import shutil
import json

from omegaconf import OmegaConf
import numpy as np
import torch

from core.util.exe_util.logger import init_logger, logger


def parse_args_and_init(mode='train'):
    # create conf
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.load(f'./core/cfg/{mode}/base.yaml')
    if 'cfg' in cli_cfg:
        user_cfg = OmegaConf.load(cli_cfg.cfg)
        cfg = OmegaConf.merge(cfg, user_cfg)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # get time
    init_time = time.strftime('%m%d%Y.%H.%M.%S', time.localtime(time.time()))
    cfg.exe.init_time = init_time
    mode_and_time = f'{mode}_{init_time}'

    # 初始化输出文件
    cfg.exe.model_dir = osp.join(cfg.exe.output_dir, 'net')
    cfg.exe.log_dir = osp.join(cfg.exe.output_dir, 'log')
    cfg.exe.summary_dir = osp.join(cfg.exe.output_dir, 'summary', mode_and_time)
    cfg.exe.vis_dir = osp.join(cfg.exe.output_dir, 'vis', mode_and_time)
    cfg.exe.code_dir = osp.join(cfg.exe.output_dir, 'code', mode_and_time)
    os.makedirs(cfg.exe.model_dir, exist_ok=True)
    os.makedirs(cfg.exe.log_dir, exist_ok=True)
    os.makedirs(cfg.exe.summary_dir, exist_ok=True)
    os.makedirs(cfg.exe.vis_dir, exist_ok=True)
    os.makedirs(cfg.exe.code_dir, exist_ok=True)

    # init logger
    log_path = osp.join(cfg.exe.log_dir, f'{mode_and_time}.txt')
    init_logger(log_path=log_path)
    logger.info(f'logger created. log file path: {log_path}')

    # backup cfg
    # cfg_str = json.dumps(cfg, indent=4)
    cfg_str = OmegaConf.to_yaml(cfg)
    logger.info(f'cfg:\n------------------------------\n{cfg_str}\n------------------------------')
    cfg_output_path = osp.join(cfg.exe.code_dir, 'config.yaml')
    # with open(cfg_output_path, 'w') as f:
    #     f.write(cfg_str)
    OmegaConf.save(config=cfg, f=cfg_output_path)

    # backup the source code
    code_input_path = './core'
    code_output_path = osp.join(cfg.exe.code_dir, 'core')
    # os.makedirs(code_output_path, exist_ok=False)
    shutil.copytree(code_input_path, code_output_path)
    logger.info(f'save source code to: {code_output_path}')

    # set seed
    setup_seed(cfg.exe.seed, cfg.exe.deterministic)

    # gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(gpu) for gpu in cfg.exe.gpu_ids])
    logger.info(f'>>> Using GPU: {cfg.exe.gpu_ids}')

    return cfg


def setup_seed(seed, deterministic):
    # cfg.GLOBAL.rng_cpu = torch.Generator()
    if seed >= 0:
        # pytorch determinism: https://pytorch.org/docs/stable/notes/randomness.html
        logger.info(f'setting seed: {seed}')
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        # cfg.GLOBAL.rng_cpu.manual_seed(2 ^ 31 + seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        logger.info(f'exe on determinisitic mode')
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        logger.info(f'exe on non-determinisitic mode')
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def cal_total_params(model, only_total=False, cal_str=False):
    if cal_str:
        results = ['calculate net params...']

    named_params = list(model.named_parameters())
    name_len = 90
    shape_len = 25
    param_len = 10
    total_params = 0
    for name, param in named_params:
        param_numel = param.numel()
        total_params += param_numel
        if not only_total and cal_str:
            results.append(f'{name:<{name_len}} | '
                           f'{str(np.array(list(param.shape)).tolist()):>{shape_len}} | '
                           f'{param_numel:>{param_len}}')

    if not only_total and cal_str:
        results.append('-' * 84)

    if cal_str:
        results.append(f'{"total":<{name_len + 1}}|{" " * (shape_len + 2)}|{total_params:>{param_len}}')
        return total_params, '\n'.join(results)
    else:
        return total_params
