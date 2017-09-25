def use_cpu():
    from caffe import set_mode_cpu
    set_mode_cpu()


def use_gpu(device):
    from caffe import set_mode_gpu, set_device
    set_mode_gpu()
    set_device(device)


class Phase:
    from caffe import TRAIN, TEST


def set_log_level(level):
    from caffe import init_log
    init_log(level)

