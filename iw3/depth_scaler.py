import torch


def minmax_normalize(frame, min_value, max_value):
    if torch.is_tensor(min_value):
        min_value = min_value.to(frame.device)
        max_value = max_value.to(frame.device)

    scale = (max_value - min_value)
    if scale > 0:
        frame = (frame - min_value) / scale
        frame = frame.clamp(0.0, 1.0)
    else:
        # all zero
        frame = frame.clamp(0.0, 1.0)

    return frame


def max_normalize(frame, min_value, max_value):
    if torch.is_tensor(max_value):
        max_value = max_value.to(frame.device)
    if max_value > 0:
        frame = frame / max_value
        frame = frame.clamp(0.0, 1.0)
    else:
        # all zero
        frame = frame.clamp(0.0, 1.0)

    return frame


class MinMaxBuffer():
    def __init__(self, size, dtype, device):
        assert size > 0
        self.count = 0
        self.size = size * 2
        self.data = torch.zeros(self.size, dtype=dtype).to(device)

    def _add(self, value):
        index = self.count % self.size
        self.data[index] = value
        self.count += 1

    def _fill(self, min_value, max_value):
        self.data[0::2] = min_value
        self.data[1::2] = max_value

    def add(self, min_value, max_value):
        if self.count == 0:
            self._fill(min_value, max_value)
            self.count = 2
        else:
            self._add(min_value)
            self._add(max_value)

    def is_filled(self):
        return self.count >= self.size

    def get_minmax(self):
        return self.data.amin(), self.data.amax()


class EMAMinMaxScaler():
    #   SimpleMinMaxScaler: decay=0, buffer_size=1
    # IncrementalEMAScaler: decay=0.75, buffer_size=1
    #      WindowEMAScaler: decay=0.9, buffer_size=30
    def __init__(self, decay=0, buffer_size=1, mode="minmax"):
        assert mode in {"minmax", "max"}
        self.normalize = {"minmax": minmax_normalize, "max": max_normalize}[mode]
        self.frame_queue = []
        assert buffer_size > 0
        self.reset(decay=decay, buffer_size=buffer_size)

    def reset(self, decay=None, buffer_size=None, **kwargs):
        # assert len(self.frame_queue) == 0  # need flush

        if decay is not None:
            self.decay = float(decay)
        if buffer_size is not None:
            self.buffer_size = int(buffer_size)
        self.min_value = None
        self.max_value = None
        self.frame_queue = []
        self.minmax_buffer = None

    def get_minmax(self):
        assert self.minmax_buffer is not None and self.minmax_buffer.is_filled()
        return self.minmax_buffer.get_minmax()

    def __call__(self, frame, return_minmax=False):
        return self.update(frame, return_minmax=return_minmax)

    def update(self, frame, return_minmax=False):
        if self.minmax_buffer is None:
            self.minmax_buffer = MinMaxBuffer(self.buffer_size, dtype=frame.dtype, device=frame.device)
        self.frame_queue.append(frame)
        self.minmax_buffer.add(frame.amin(), frame.amax())
        if not self.minmax_buffer.is_filled():
            # queued
            if return_minmax:
                return None, None, None
            else:
                return None

        min_value, max_value = self.get_minmax()
        if self.min_value is None:
            self.min_value = min_value
            self.max_value = max_value
        else:
            self.min_value = self.decay * self.min_value + (1. - self.decay) * min_value
            self.max_value = self.decay * self.max_value + (1. - self.decay) * max_value

        frame = self.frame_queue.pop(0)
        frame = self.normalize(frame, self.min_value, self.max_value)

        if return_minmax:
            return (frame, self.min_value, self.max_value)
        else:
            return frame

    def flush(self, return_minmax=False):
        if not self.frame_queue:
            self.reset()
            return []

        if self.min_value is None:
            min_value, max_value = self.minmax_buffer.get_minmax()
        else:
            min_value, max_value = self.min_value, self.max_value

        if return_minmax:
            frames = [(self.normalize(frame, min_value, max_value),
                       min_value, max_value)
                      for frame in self.frame_queue]
            self.reset()
            return frames
        else:
            frames = [self.normalize(frame, min_value, max_value)
                      for frame in self.frame_queue]
            self.reset()
            return frames


def _test():
    import matplotlib.pyplot as plt
    x = [float(i) for i in range(100)]
    zeros = [0 for i in range(100)]

    x = torch.tensor(zeros + x + list(reversed(x)) + zeros, dtype=torch.float32)
    x = torch.stack([x, x + 10]).permute(1, 0).contiguous()

    scaler = EMAMinMaxScaler(decay=0.9, buffer_size=22)
    min_values = []
    max_values = []
    for frame in x:
        frame, min_value, max_value = scaler.update(frame, return_minmax=True)
        if min_value is not None:
            min_values.append(min_value)
            max_values.append(max_value)
    for frame, min_value, max_value in scaler.flush(return_minmax=True):
        min_values.append(min_value)
        max_values.append(max_value)

    min_values = torch.tensor(min_values)
    max_values = torch.tensor(max_values)

    x = torch.stack([
        x.permute(1, 0)[0],
        x.permute(1, 0)[1],
        min_values,
        max_values,
    ]).permute(1, 0)
    plt.plot(x)
    plt.show()


if __name__ == "__main__":
    _test()
