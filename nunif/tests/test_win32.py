import ctypes
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from nunif.utils import win32


def _reset_win32_state() -> None:
    win32._dpi_initialized = False
    win32._dpi_result = False
    win32._power_throttling_initialized = False
    win32._power_throttling_result = False


@pytest.fixture(autouse=True)
def reset_state():
    _reset_win32_state()
    yield
    _reset_win32_state()


class _FakeFunc:
    def __init__(self, impl: Any) -> None:
        self._impl = impl
        self.argtypes: list[Any] | None = None
        self.restype: Any = None
        self.calls: list[tuple[Any, ...]] = []

    def __call__(self, *args: Any) -> Any:
        self.calls.append(args)
        return self._impl(*args)


def _extract_throttling_state(state_ptr: Any) -> tuple[int, int, int]:
    raw = ctypes.cast(state_ptr, ctypes.POINTER(ctypes.c_ulong * 3)).contents
    return int(raw[0]), int(raw[1]), int(raw[2])


def test_power_throttling_full_mask_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[tuple[int, int, int, int, int]] = []

    def fake_set_info(handle: Any, info_class: int, state_ptr: Any, size: int) -> int:
        version, control_mask, state_mask = _extract_throttling_state(state_ptr)
        observed.append((info_class, size, version, control_mask, state_mask))
        return 1

    fake_kernel32 = SimpleNamespace(
        GetCurrentProcess=_FakeFunc(lambda: 0x1234),
        SetProcessInformation=_FakeFunc(fake_set_info),
    )

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "WinDLL", lambda name, **kwargs: fake_kernel32, raising=False)

    assert win32.disable_win32_power_throttling() is True
    assert observed == [
        (
            win32.PROCESS_POWER_THROTTLING_CLASS,
            ctypes.sizeof(ctypes.c_ulong) * 3,
            win32.PROCESS_POWER_THROTTLING_CURRENT_VERSION,
            win32.PROCESS_POWER_THROTTLING_EXECUTION_SPEED | win32.PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION,
            0,
        )
    ]
    assert fake_kernel32.GetCurrentProcess.argtypes == []
    assert fake_kernel32.SetProcessInformation.argtypes is not None

    # Idempotent: second call does not invoke Win32 API again.
    assert win32.disable_win32_power_throttling() is True
    assert len(observed) == 1


def test_power_throttling_fallback_to_execution_speed(monkeypatch: pytest.MonkeyPatch) -> None:
    masks: list[int] = []

    def fake_set_info(handle: Any, info_class: int, state_ptr: Any, size: int) -> int:
        _, control_mask, state_mask = _extract_throttling_state(state_ptr)
        assert state_mask == 0
        masks.append(control_mask)
        return 1 if control_mask == win32.PROCESS_POWER_THROTTLING_EXECUTION_SPEED else 0

    fake_kernel32 = SimpleNamespace(
        GetCurrentProcess=_FakeFunc(lambda: 0x1234),
        SetProcessInformation=_FakeFunc(fake_set_info),
    )

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "WinDLL", lambda name, **kwargs: fake_kernel32, raising=False)

    assert win32.disable_win32_power_throttling() is True
    assert masks == [
        win32.PROCESS_POWER_THROTTLING_EXECUTION_SPEED | win32.PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION,
        win32.PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
    ]


def test_enable_win32_dpi_awareness_idempotent_and_accepts_access_denied(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_shcore = SimpleNamespace(
        SetProcessDpiAwareness=_FakeFunc(lambda awareness: win32._E_ACCESSDENIED),
    )

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "WinDLL", lambda name, **kwargs: fake_shcore, raising=False)

    assert win32.enable_win32_dpi_awareness() is True
    assert win32.enable_win32_dpi_awareness() is True
    assert fake_shcore.SetProcessDpiAwareness.calls == [(win32.PROCESS_PER_MONITOR_DPI_AWARE,)]


def test_non_windows_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def fail_windll(*args: Any, **kwargs: Any) -> Any:
        nonlocal called
        called = True
        raise AssertionError("WinDLL should not be called on non-Windows platforms")

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(ctypes, "WinDLL", fail_windll, raising=False)

    assert win32.enable_win32_dpi_awareness() is False
    assert win32.disable_win32_power_throttling() is False
    assert called is False


def test_enable_win32_dpi_awareness_failure_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_shcore = SimpleNamespace(
        SetProcessDpiAwareness=_FakeFunc(lambda awareness: -2147024809),  # E_INVALIDARG
    )

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "WinDLL", lambda name, **kwargs: fake_shcore, raising=False)

    assert win32.enable_win32_dpi_awareness() is False
    assert win32.enable_win32_dpi_awareness() is False
    assert len(fake_shcore.SetProcessDpiAwareness.calls) == 1


def test_power_throttling_failure_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_kernel32 = SimpleNamespace(
        GetCurrentProcess=_FakeFunc(lambda: 0x1234),
        SetProcessInformation=_FakeFunc(lambda *args: 0),
    )

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "WinDLL", lambda name, **kwargs: fake_kernel32, raising=False)

    assert win32.disable_win32_power_throttling() is False
    assert len(fake_kernel32.SetProcessInformation.calls) == 2
    assert win32.disable_win32_power_throttling() is False
    assert len(fake_kernel32.SetProcessInformation.calls) == 2


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only smoke test")
def test_windows_real_calls_do_not_mutate_global_windll() -> None:
    if sys.platform != "win32":
        return
    before_restype = ctypes.windll.kernel32.GetCurrentProcess.restype
    before_argtypes = ctypes.windll.kernel32.GetCurrentProcess.argtypes
    before_set_info_argtypes = ctypes.windll.kernel32.SetProcessInformation.argtypes
    before_dpi_argtypes = ctypes.windll.shcore.SetProcessDpiAwareness.argtypes

    assert win32.enable_win32_dpi_awareness() is True
    assert win32.disable_win32_power_throttling() is True
    # Repeat calls should succeed via idempotent cache.
    assert win32.enable_win32_dpi_awareness() is True
    assert win32.disable_win32_power_throttling() is True

    assert ctypes.windll.kernel32.GetCurrentProcess.restype == before_restype
    assert ctypes.windll.kernel32.GetCurrentProcess.argtypes == before_argtypes
    assert ctypes.windll.kernel32.SetProcessInformation.argtypes == before_set_info_argtypes
    assert ctypes.windll.shcore.SetProcessDpiAwareness.argtypes == before_dpi_argtypes
