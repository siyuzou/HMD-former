import importlib


def get_cls(name):
    mod_name, cls_name = name.rsplit('.', maxsplit=1)
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name)
    return cls


def make_instance(cfg, kwargs=None):
    cfg = cfg.copy()
    cls = cfg.pop('cls')
    cfg = dict(cfg)
    if kwargs is not None and len(kwargs) > 0:
        cfg.update(kwargs)
    instance = get_cls(cls)(**cfg)
    return instance
