from av.codec.hwaccel import HWAccel, hwdevices_available
from typing import List, Set, Optional


def get_supported_hwdevices() -> List[str]:
    supported_options: List[str] = []
    devices: Set[str] = set(hwdevices_available())
    if "cuda" in devices:
        # TODO: direct cuda support
        supported_options.append("cuda")
        supported_options.append("cuda_hwdownload")

    # TODO: Support "qsv"
    for device in ("d3d12va", "d3d11va", "dxva2", "vaapi", "amf", "videotoolbox"):
        if device in devices:
            supported_options.append(device)

    return supported_options


HW_DEVICES = get_supported_hwdevices()


def create_hwaccel(
    device: Optional[str],
    device_id: Optional[int] = 0,
    disable_software_fallback: bool = False,
) -> Optional[HWAccel]:
    if device is None:
        return None

    assert device in HW_DEVICES

    allow_software_fallback = not disable_software_fallback

    if device == "cuda":
        if device_id is None:
            device_id = 0

        return HWAccel(
            device_type="cuda",
            device=device_id,
            is_hw_owned=True,
            options={"primary_ctx": "1"},
            allow_software_fallback=allow_software_fallback,
        )
    elif device == "cuda_hwdownload":
        if device_id is None:
            device_id = 0

        return HWAccel(
            device_type="cuda",
            device=device_id,
            is_hw_owned=False,
            options={"primary_ctx": "1"},
            allow_software_fallback=allow_software_fallback,
        )
    else:
        return HWAccel(
            device_type=device,
            is_hw_owned=False,
            options={"primary_ctx": "1"},
            allow_software_fallback=allow_software_fallback,
        )
