from torch.utils.data import DataLoader
from core.util.exe_util.util import get_cls, make_instance

def create_valid_dataloader(cfg):
    # extract cls name
    dataset = make_instance(cfg.dataset.valid)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.valid.batch_size * len(cfg.exe.gpu_ids),
        shuffle=False,
        num_workers=cfg.exe.workers_test,
        drop_last=False,
    )

    return dataset, dataloader
