import abc
import os
import os.path as osp

from core.util.exe_util.timer import Timer


class Base:
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.cur_epoch = 0

        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()