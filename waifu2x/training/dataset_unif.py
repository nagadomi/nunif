import random
import torch
from nunif.training.sampler import HardExampleSampler
from .dataset import Waifu2xDatasetBase, Waifu2xDataset


# 1x 2x 4x unified dataset
class MultiBatchSampler():
    def __init__(self, samplers, batch_size, num_samples=None, shuffle=True):
        self.samplers = samplers
        self.batch_size = batch_size
        self.index_range = []
        self.shuffle = shuffle
        self.num_samples = num_samples
        ei = 0
        for sampler in samplers:
            sampler_len = len(sampler.weights) if isinstance(sampler, HardExampleSampler) else len(sampler)
            self.index_range.append((ei, ei + sampler_len))
            ei += sampler_len
        self.data_len = ei

    def split_index(self, batch_sampler_index):
        sampler_index = -1
        start_index = -1
        for i, (si, ei) in enumerate(self.index_range):
            if batch_sampler_index < ei:
                sampler_index = i
                start_index = si
                break
        assert sampler_index >= 0 and start_index >= 0
        data_index = batch_sampler_index - start_index
        return sampler_index, data_index

    def marge_index(self, sampler_index, data_index):
        index_start, index_end = self.index_range[sampler_index]
        assert index_start + data_index < index_end
        return index_start + data_index

    def _min_index_end(self):
        min_index_end = min([
            len(sampler.weights) if isinstance(sampler, HardExampleSampler) else len(sampler)
            for sampler in self.samplers])
        return min_index_end

    def __iter__(self):
        if not self.shuffle:
            min_index_end = self._min_index_end()
            index_iter = iter(range(min_index_end))
            i = 0
            while True:
                try:
                    sampler_index = i % len(self.samplers)
                    batch = [self.marge_index(sampler_index, next(index_iter))
                             for _ in range(self.batch_size)]
                    yield batch
                    i += 1
                except StopIteration:
                    break
        else:
            num_samples = 0
            sampler_iters = [iter(sampler) for sampler in self.samplers]
            while True:
                try:
                    sampler_index = random.randint(0, len(self.samplers) - 1)
                    batch = [self.marge_index(sampler_index, next(sampler_iters[sampler_index]))
                             for _ in range(self.batch_size)]
                    num_samples += self.batch_size
                    yield batch
                    if self.num_samples <= num_samples:
                        break
                except StopIteration:
                    break

    def __len__(self):
        if self.num_samples is not None:
            return self.num_samples // self.batch_size
        else:
            return self._min_index_end() // self.batch_size


class Waifu2xUnifiedDataset(Waifu2xDatasetBase):
    """
    Dataset that randomly returns multiple scale factor settings each minibatch
    """
    def __init__(self, input_dir,
                 model_offsets,
                 scale_factors,
                 tile_size,
                 batch_size,
                 num_samples=None,
                 da_jpeg_p=0, da_scale_p=0, da_chshuf_p=0, da_unsharpmask_p=0, da_grayscale_p=0,
                 bicubic_only=False,
                 deblur=0, resize_blur_p=0.1,
                 noise_level=-1, style=None,
                 training=True):
        super().__init__(input_dir, num_samples, hard_example_history_size=16)
        datasets = {}
        for scale_factor, model_offset in zip(scale_factors, model_offsets):
            if scale_factor == 1:
                deblur = 0
            elif scale_factor == 2:
                deblur = 0.025
            elif scale_factor == 4:
                deblur = 0.05
            datasets[scale_factor] = Waifu2xDataset(
                input_dir=input_dir,
                model_offset=model_offset,
                scale_factor=scale_factor,
                tile_size=tile_size,
                num_samples=num_samples,
                da_jpeg_p=da_jpeg_p,
                da_scale_p=da_scale_p,
                da_chshuf_p=da_chshuf_p,
                da_unsharpmask_p=da_unsharpmask_p,
                da_grayscale_p=da_grayscale_p,
                bicubic_only=bicubic_only,
                deblur=deblur,
                resize_blur_p=resize_blur_p,
                noise_level=noise_level,
                style=style,
                training=training
            )
            # Note:
            #  image list from input_dir is sorted by names
            #  so the index points to the same image for self[index] and all in each dataset[index].

        self.datasets = datasets
        self.scale_factors = scale_factors
        self.training = training
        if self._sampler is not None:
            self._batch_sampler = MultiBatchSampler(
                samplers=[self._sampler for _ in self.scale_factors],
                batch_size=batch_size,
                num_samples=num_samples,
                shuffle=self.training,
            )
        else:
            self._batch_sampler = MultiBatchSampler(
                samplers=[list(range(len(self))) for _ in self.scale_factors],
                batch_size=batch_size,
                shuffle=self.training,
            )

    def batch_sampler(self):
        return self._batch_sampler

    def __getitem__(self, batch_sampler_index):
        scale_factor_index, index = self._batch_sampler.split_index(batch_sampler_index)
        scale_factor = self.scale_factors[scale_factor_index]
        dataset = self.datasets[scale_factor]
        x, y, idx = dataset[index]

        return x, y, scale_factor, idx


def _test_unified():
    dataset = Waifu2xUnifiedDataset(
        input_dir="./data/waifu2x/eval",
        scale_factors=[1, 2, 4],
        model_offsets=[8, 16, 32],
        tile_size=64,
        batch_size=4,
        training=False
    )
    batch_sampler = dataset.batch_sampler()
    for batch in batch_sampler:
        xs = []
        ys = []
        scale_factors = []
        for i in batch:
            x, y, scale_factor, idx = dataset[i]
            xs.append(x)
            ys.append(y)
            scale_factors.append(torch.tensor(scale_factor))
        x_batch = torch.stack(xs)
        y_batch = torch.stack(ys)
        scale_factors = torch.stack(scale_factors)
        print(x_batch.shape, y_batch.shape, scale_factors)


if __name__ == "__main__":
    _test_unified()
