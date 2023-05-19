import torch


def create_device_name(device_id):
    if isinstance(device_id, (list, tuple)):
        assert len(device_id) > 0
        device_id = device_id[0]
    if device_id < 0:
        device_name = "cpu"
    else:
        if torch.cuda.is_available():
            device_name = 'cuda:%d' % device_id
        elif torch.backends.mps.is_available():
            device_name = 'mps:%d' % device_id
        else:
            raise ValueError("No cuda/mps available. Use `--gpu -1` for CPU.")
    return device_name


def create_device(device_id):
    return torch.device(create_device_name(device_id))


def device_is(device, name):
    if isinstance(device, torch.device):
        return device.type == name
    else:
        return name in str(device)


def device_is_mps(device):
    return device_is(device, "mps")


def device_is_cpu(device):
    return device_is(device, "cpu")


def device_is_cuda(device):
    return device_is(device, "cuda")


def autocast(device, dtype=None, enabled=True):
    if device_is_cpu(device):
        # autocast on cpu is extremely slow for unknown reasons
        # disabled
        amp_device_type = "cpu"
        amp_dtype = torch.bfloat16
        if enabled:
            enabled = False
    elif device_is_mps(device):
        # currently pytorch does not support mps autocast
        # disabled
        amp_device_type = "cpu"
        amp_dtype = torch.bfloat16
        if enabled:
            enabled = False
    elif device_is_cuda(device):
        amp_device_type = device.type
        amp_dtype = dtype
        if False:
            # TODO: I think better to do this, but leave it to the user (use --disable-amp option)
            cuda_capability = torch.cuda.get_device_capability(device)
            if enabled and cuda_capability < (7, 0):
                enabled = False
    else:
        # Unknown device
        amp_device_type = device.type
        amp_dtype = dtype

    return torch.autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=enabled)
