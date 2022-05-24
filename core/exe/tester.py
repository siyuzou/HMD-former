import os
import os.path as osp

import tqdm
import torch
from torch.nn.parallel.data_parallel import DataParallel

from core.exe.base import Base
from core.util.exe_util.logger import logger
from core.data.dataset.make_dataset import create_valid_dataloader

from core.metric.eval import evaluate_one_batch, MetricsMeter, generate_results_txt
from core.util.exe_util.util import get_cls


class Tester(Base):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

    def init(self):
        self.prepare_model()
        self.prepare_data()

    def prepare_data(self):
        # make data
        logger.info(f'loading validset.')
        self.valid_set, self.valid_loader = create_valid_dataloader(self.cfg)
        logger.info(f'validset loaded.')

    def prepare_model(self):

        logger.info(f'making net.')
        model = get_cls(self.cfg.net.cls)(self.cfg.net, mode='test')

        assert len(self.cfg.net.model_paths) > 0
        for model_name, model_path in self.cfg.net.model_paths.items():
            self._load_model(getattr(model, model_name), model_path)
            logger.info(f'<{model_name}> loaded from <{model_path}>')

        # DataParallel
        if len(self.cfg.exe.gpu_ids) > 0:
            model = DataParallel(model).cuda()
        model.eval()

        self.model = model

        logger.info('net constructed.')

    @property
    def raw_model(self):
        if isinstance(self.model, DataParallel):
            return self.model.module
        else:
            return self.model

    def _load_model(self, model, model_path):
        logger.info(f'loading net from: {osp.basename(model_path)}')
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['network'], strict=False)

    def test_once(self):

        dataset_name = 'Human36M'
        save_raw_metrics = False

        self.model.eval()

        all_results = dict()
        all_metrics = MetricsMeter(preserve_raw_data=save_raw_metrics)

        for itr, inputs in enumerate(tqdm.tqdm(self.valid_loader)):
            # forward
            with torch.no_grad():
                outs = self.model(inputs, 'test')

            # evaluate
            results, batchsize = evaluate_one_batch(outs, dataset_name=dataset_name, preserve_raw=save_raw_metrics)
            all_metrics.update(results, batchsize)

        logger.info(f'merging...')
        # merge
        results = all_metrics.get_results()
        txt = generate_results_txt(results, dataset_name)
        logger.info(f'test result on {dataset_name} split:\n{txt}')
        all_results[dataset_name] = results

        return all_results
