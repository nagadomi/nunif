from nunif.utils.video import pyav_init_cuda_primary_context

from .cli import main

if __name__ == "__main__":
    pyav_init_cuda_primary_context()
    main()
