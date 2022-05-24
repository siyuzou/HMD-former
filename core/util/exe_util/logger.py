import logging
import os
import os.path as osp

OK = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
END = '\033[0m'

PINK = '\033[95m'
BLUE = '\033[94m'
GREEN = OK
RED = FAIL
WHITE = END
YELLOW = WARNING


class colorlogger():
    def __init__(self):
        pass

    def init_logger(self, log_path=None):
        if log_path is not None:
            log_dir, log_name = osp.split(log_path)
        else:
            log_name = 'log'

        # set log
        self._logger = logging.getLogger(log_name)
        self._logger.setLevel(logging.INFO)
        console_log = logging.StreamHandler()
        console_log.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            "{}%(asctime)s{} %(message)s".format(GREEN, END),
            "%m-%d %H:%M:%S")
        console_log.setFormatter(console_formatter)
        self._logger.addHandler(console_log)

        if log_path is not None:
            os.makedirs(log_dir, exist_ok=True)
            file_log = logging.FileHandler(log_path, mode='a')
            file_log.setLevel(logging.INFO)
            file_formatter = logging.Formatter(
                "%(asctime)s %(message)s",
                "%m-%d %H:%M:%S")
            file_log.setFormatter(file_formatter)
            self._logger.addHandler(file_log)

    def debug(self, msg):
        if hasattr(self, '_logger'):
            self._logger.debug(str(msg))
        else:
            print(msg)

    def info(self, msg):
        if hasattr(self, '_logger'):
            self._logger.info(str(msg))
        else:
            print(msg)

    def warning(self, msg):
        if hasattr(self, '_logger'):
            self._logger.warning(WARNING + 'WRN: ' + str(msg) + END)
        else:
            print(msg)

    def critical(self, msg):
        if hasattr(self, '_logger'):
            self._logger.critical(RED + 'CRI: ' + str(msg) + END)
        else:
            print(msg)

    def error(self, msg):
        if hasattr(self, '_logger'):
            self._logger.error(RED + 'ERR: ' + str(msg) + END)
        else:
            print(msg)


logger = colorlogger()


def init_logger(log_path):
    global logger
    logger.init_logger(log_path=log_path)
