import torch


class FrameQueue():
    def __init__(
            self,
            synthetic_view, seq, height, width, dtype, device,
            mask_height=None, mask_width=None
    ):
        if mask_width is None:
            mask_width = width
        if mask_height is None:
            mask_height = height

        self.left_eye = torch.zeros((seq, 3, height, width), dtype=dtype, device=device)
        self.right_eye = torch.zeros((seq, 3, height, width), dtype=dtype, device=device)
        if synthetic_view == "both":
            self.left_mask = torch.zeros((seq, 1, mask_height, mask_width), dtype=dtype, device=device)
            self.right_mask = torch.zeros((seq, 1, mask_height, mask_width), dtype=dtype, device=device)
        elif synthetic_view == "right":
            self.right_mask = torch.zeros((seq, 1, mask_height, mask_width), dtype=dtype, device=device)
            self.left_mask = None
        elif synthetic_view == "left":
            self.left_mask = torch.zeros((seq, 1, mask_height, mask_width), dtype=dtype, device=device)
            self.right_mask = None

        self.synthetic_view = synthetic_view
        self.index = 0
        self.max_index = seq

    def full(self):
        return self.index == self.max_index

    def empty(self):
        return self.index == 0

    def add(self, left_eye, right_eye, left_mask=None, right_mask=None):
        self.left_eye[self.index] = left_eye
        self.right_eye[self.index] = right_eye
        if left_mask is not None:
            self.left_mask[self.index] = left_mask
        if right_mask is not None:
            self.right_mask[self.index] = right_mask

        self.index += 1

    def fill(self):
        if self.full():
            return 0

        pad = 0
        i = self.index - 1
        if self.synthetic_view == "both":
            frame = dict(left_eye=self.left_eye[i].clone(),
                         right_eye=self.right_eye[i].clone(),
                         left_mask=self.left_mask[i].clone(),
                         right_mask=self.right_mask[i].clone())
        elif self.synthetic_view == "right":
            frame = dict(left_eye=self.left_eye[i].clone(),
                         right_eye=self.right_eye[i].clone(),
                         right_mask=self.right_mask[i].clone())
        elif self.synthetic_view == "left":
            frame = dict(left_eye=self.left_eye[i].clone(),
                         right_eye=self.right_eye[i].clone(),
                         left_mask=self.left_mask[i].clone())
        while not self.full():
            pad += 1
            self.add(**frame)

        return pad

    def remove(self, n):
        if n > 0 and n < self.max_index:
            for i in range(n):
                self.left_eye[i] = self.left_eye[i + n]
                self.right_eye[i] = self.right_eye[i + n]
                if self.right_mask is not None:
                    self.right_mask[i] = self.right_mask[i + n]
                if self.left_mask is not None:
                    self.left_mask[i] = self.left_mask[i + n]

        self.index -= n
        assert self.index >= 0

    def get(self):
        if self.synthetic_view == "both":
            return self.left_eye, self.right_eye, self.left_mask, self.right_mask
        elif self.synthetic_view == "left":
            return self.left_eye, self.right_eye, self.left_mask
        elif self.synthetic_view == "right":
            return self.left_eye, self.right_eye, self.right_mask

    def clear(self):
        self.index = 0
