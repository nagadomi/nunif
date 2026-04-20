import torch
from .ui_utils import create_parser, set_state_args, waifu2x_main
from . import models # noqa
from nunif.logger import logger
from nunif.device import device_is_cuda
from nunif.utils.video import pyav_init_cuda_primary_context


def main():
    parser = create_parser()
    args = parser.parse_args()
    set_state_args(args)
    waifu2x_main(args)

    if device_is_cuda(args.state["device"]):
        max_vram_mb = int(torch.cuda.max_memory_allocated(args.state["device"]) / (1024 * 1024))
        logger.debug(f"GPU Max Memory Allocated {max_vram_mb}MB")


if __name__ == "__main__":
    pyav_init_cuda_primary_context()
    main()
