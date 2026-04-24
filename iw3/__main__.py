from .cli import main
from nunif.utils.video import pyav_init_cuda_primary_context

if __name__ == "__main__":
    pyav_init_cuda_primary_context()
    main()
