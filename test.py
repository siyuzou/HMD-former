from core.util.exe_util.init import parse_args_and_init, cal_total_params
from core.util.exe_util.logger import logger
from core.exe.tester import Tester


def main():
    cfg = parse_args_and_init(mode='test')

    tester = Tester(cfg)
    tester.init()

    # print network parameters
    _, result_str = cal_total_params(tester.raw_model, only_total=False, cal_str=True)
    logger.info(result_str)
    if cfg.exe.dry_run:
        logger.info('Dry run, exit.')
        exit(0)

    # test
    tester.test_once()


if __name__ == '__main__':
    main()
