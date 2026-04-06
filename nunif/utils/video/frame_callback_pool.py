import torch
from concurrent.futures import ThreadPoolExecutor
from .color_transform import TensorFrame
import numpy as np
import av


RGB_8BIT = "rgb24"
RGB_16BIT = "gbrp16le"


# TODO: from* to* utilities will be deleted later


def from_ndarray(x):
    if x.dtype == np.uint8:
        format = RGB_8BIT
    elif x.dtype == np.uint16:
        format = RGB_16BIT
    else:
        raise ValueError(f"unsupported dtype {x.dtype}")

    return av.video.frame.VideoFrame.from_ndarray(x, format=format)


def from_image(im):
    return av.video.frame.VideoFrame.from_image(im)


def from_tensor(x, use_16bit=False):
    if use_16bit:
        dtype = torch.uint16
        value_scale = 65535.0
    else:
        dtype = torch.uint8
        value_scale = 255.0

    x = (
        (x.permute(1, 2, 0).contiguous() * value_scale)
        .round_()
        .to(dtype)
        .detach()
        .cpu()
        .numpy()
    )
    return from_ndarray(x)


def to_tensor(frame, device=None):
    if isinstance(frame, TensorFrame):
        x = frame.to_chw()
        if device is not None:
            x = x.to(device)
        return x
    elif isinstance(frame, av.VideoFrame):
        x = torch.from_numpy(to_ndarray(frame))
        if device is not None:
            x = x.to(device)
        # CHW float32
        return x.permute(2, 0, 1).contiguous() / torch.iinfo(x.dtype).max
    else:
        raise ValueError(f"{type(frame)} not supported")


def to_frame(x, use_16bit=False):
    if torch.is_tensor(x):
        # float CHW
        return from_tensor(x, use_16bit=use_16bit)
    elif isinstance(x, np.ndarray):
        # uint8/uint16 HWC
        return from_ndarray(x)
    elif isinstance(x, av.VideoFrame):
        return x
    else:
        return from_image(x)


def to_ndarray(frame):
    use_16bit = frame.format.components[0].bits > 8
    if use_16bit:
        format = RGB_16BIT
    else:
        format = RGB_8BIT
    return frame.to_ndarray(format=format)


def get_source_dtype(frame):
    if isinstance(frame, TensorFrame):
        use_16bit = frame.use_16bit
    elif isinstance(frame, av.VideoFrame):
        use_16bit = frame.format.components[0].bits > 8

    if use_16bit:
        return torch.uint16
    else:
        return torch.uint8


class _DummyFuture:
    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result

    def done(self):
        return True


class _DummyThreadPool:
    def __init__(self):
        pass

    def submit(self, func, *args, **kwargs):
        result = func(*args, **kwargs)
        return _DummyFuture(result)

    def shutdown(self):
        pass


class FrameCallbackPool:
    """
    thread pool callback wrapper
    """

    def __init__(
        self,
        frame_callback,
        batch_size,
        device,
        max_workers=1,
        max_batch_queue=2,
        require_pts=False,
        skip_pts=-1,
        require_flush=False,
        preprocess_callback=None,
        postprocess_callback=None,
        use_16bit=False,
    ):
        if max_workers > 0:
            self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        else:
            self.thread_pool = _DummyThreadPool()
        self.require_pts = require_pts
        self.require_flush = require_flush
        self.skip_pts = skip_pts
        self.use_16bit = use_16bit
        self.frame_callback = frame_callback
        self.preprocess_callback = preprocess_callback
        self.postprocess_callback = postprocess_callback
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.max_batch_queue = max_batch_queue
        self.devices = device if isinstance(device, (tuple, list)) else [device]
        self.round_robin_index = 0
        self.frame_queue = []
        self.pts_queue = []
        self.batch_queue = []
        self.pts_batch_queue = []
        self.futures = []

    def make_args(self, batch, pts_batch, flush):
        if self.require_pts and self.require_flush:
            return (batch, pts_batch, flush)
        elif self.require_pts:
            return (batch, pts_batch)
        elif self.require_flush:
            return (batch, flush)
        else:
            return (batch,)

    def get_results(self, future):
        frames = future.result()
        if self.postprocess_callback is not None:
            frames = self.postprocess_callback(frames)
        return [frame for frame in frames] if frames is not None else []

    def submit(self, *args):
        if self.preprocess_callback is not None:
            args = self.preprocess_callback(*args)
            future = self.thread_pool.submit(self.frame_callback, args)
        else:
            future = self.thread_pool.submit(self.frame_callback, *args)

        return future

    def __call__(self, frame):
        if False:
            # for debug
            print(
                "\n__call__",
                "frame_queue",
                len(self.frame_queue),
                "batch_queue",
                len(self.batch_queue),
                "pts_queue",
                len(self.pts_queue),
                "pts_batch_queue",
                len(self.pts_batch_queue),
                "futures",
                len(self.futures),
            )

        if frame is None:
            return self.finish()
        if frame.pts <= self.skip_pts:
            return None

        self.pts_queue.append(frame.pts)
        device = self.devices[self.round_robin_index % len(self.devices)]
        frame = to_tensor(frame, device=device)

        self.frame_queue.append(frame)
        if len(self.frame_queue) == self.batch_size:
            batch = torch.stack(self.frame_queue)
            self.batch_queue.append(batch)
            self.frame_queue.clear()
            self.pts_batch_queue.append(list(self.pts_queue))
            self.pts_queue.clear()
            self.round_robin_index += 1

        if self.batch_queue:
            if len(self.futures) < self.max_workers or self.max_workers <= 0:
                batch = self.batch_queue.pop(0)
                pts_batch = self.pts_batch_queue.pop(0)
                future = self.submit(*self.make_args(batch, pts_batch, False))
                self.futures.append(future)
            if len(self.batch_queue) >= self.max_batch_queue and self.futures:
                future = self.futures.pop(0)
                return self.get_results(future)
        if self.futures:
            if self.futures[0].done():
                future = self.futures.pop(0)
                return self.get_results(future)

        return None

    def finish(self):
        if self.frame_queue:
            batch = torch.stack(self.frame_queue)
            self.batch_queue.append(batch)
            self.frame_queue.clear()
            self.pts_batch_queue.append(list(self.pts_queue))
            self.pts_queue.clear()

        frame_remains = []
        while len(self.batch_queue) > 0:
            if len(self.futures) < self.max_workers or self.max_workers <= 0:
                batch = self.batch_queue.pop(0)
                pts_batch = self.pts_batch_queue.pop(0)
                future = self.submit(*self.make_args(batch, pts_batch, False))
                self.futures.append(future)
            else:
                future = self.futures.pop(0)
                frame_remains += self.get_results(future)
        while len(self.futures) > 0:
            future = self.futures.pop(0)
            frame_remains += self.get_results(future)

        if self.require_flush:
            future = self.submit(*self.make_args(None, None, True))
            frame_remains += self.get_results(future)

        return frame_remains

    def shutdown(self):
        pool = self.thread_pool
        self.thread_pool = None
        if pool is not None:
            pool.shutdown()

    def __del__(self):
        self.shutdown()
