import sys
import threading

PROCESS_PER_MONITOR_DPI_AWARE = 2
PROCESS_POWER_THROTTLING_CURRENT_VERSION = 1
PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1
PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION = 0x4
PROCESS_POWER_THROTTLING_CLASS = 4

_S_OK = 0
_E_ACCESSDENIED = -2147024891  # 0x80070005 as signed 32-bit HRESULT

_init_lock = threading.Lock()
_dpi_initialized = False
_dpi_result = False
_power_throttling_initialized = False
_power_throttling_result = False


def enable_win32_dpi_awareness() -> bool:
    global _dpi_initialized, _dpi_result
    if sys.platform != "win32":
        return False

    with _init_lock:
        if _dpi_initialized:
            return _dpi_result

        import ctypes

        try:
            # Use an isolated WinDLL instance to avoid mutating global ctypes.windll prototypes.
            shcore = ctypes.WinDLL("shcore", use_last_error=True)
            set_dpi_awareness = shcore.SetProcessDpiAwareness
            set_dpi_awareness.argtypes = [ctypes.c_int]
            set_dpi_awareness.restype = ctypes.c_long  # HRESULT
            hr = int(set_dpi_awareness(PROCESS_PER_MONITOR_DPI_AWARE))
            _dpi_result = hr in (_S_OK, _E_ACCESSDENIED)
        except Exception:
            _dpi_result = False

        _dpi_initialized = True
        return _dpi_result


def disable_win32_power_throttling() -> bool:
    global _power_throttling_initialized, _power_throttling_result
    if sys.platform != "win32":
        return False

    with _init_lock:
        if _power_throttling_initialized:
            return _power_throttling_result

        import ctypes
        from ctypes import wintypes

        class PROCESS_POWER_THROTTLING_STATE(ctypes.Structure):
            _fields_ = [
                ("Version", wintypes.ULONG),
                ("ControlMask", wintypes.ULONG),
                ("StateMask", wintypes.ULONG),
            ]

        try:
            # Opt out of EcoQoS power throttling so background processing is not moved to E-cores.
            # Use an isolated WinDLL instance to avoid mutating global ctypes.windll prototypes.
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            get_current_process = kernel32.GetCurrentProcess
            get_current_process.argtypes = []
            get_current_process.restype = wintypes.HANDLE

            set_process_information = kernel32.SetProcessInformation
            set_process_information.argtypes = [
                wintypes.HANDLE,
                wintypes.DWORD,
                wintypes.LPVOID,
                wintypes.DWORD,
            ]
            set_process_information.restype = wintypes.BOOL

            handle = get_current_process()
            # Try both flags first (Windows 11+), then fall back to EXECUTION_SPEED only
            # when IGNORE_TIMER_RESOLUTION is rejected (e.g. Windows 10).
            ok = False
            for mask in (
                PROCESS_POWER_THROTTLING_EXECUTION_SPEED | PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION,
                PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
            ):
                state = PROCESS_POWER_THROTTLING_STATE(
                    PROCESS_POWER_THROTTLING_CURRENT_VERSION,
                    mask,
                    0,
                )
                if set_process_information(
                    handle,
                    PROCESS_POWER_THROTTLING_CLASS,
                    ctypes.byref(state),
                    ctypes.sizeof(state),
                ):
                    ok = True
                    break
            _power_throttling_result = ok
        except Exception:
            _power_throttling_result = False

        _power_throttling_initialized = True
        return _power_throttling_result
