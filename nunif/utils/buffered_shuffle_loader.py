import torch

# TODO: refactor
#

class BufferedShuffleLoader():
    def __init__(self, data_loader, batch_size, buffer_size):
        assert(data_loader.drop_last == True)
        assert(data_loader.batch_size >= batch_size and data_loader.batch_size % batch_size == 0)
        self.data_loader = data_loader
        self.iter = None
        self.batch_size = batch_size
        self.dl_batch_size = data_loader.batch_size
        self.buffer = None
        self.buffer_size = buffer_size
        self.elm = min(buffer_size * self.dl_batch_size,
                       len(data_loader) * self.dl_batch_size)
        self.dim = 0
        self.index = 0
        self.count = 0
        self.iter_per_batch = self.dl_batch_size // batch_size
        self.perm = torch.empty(self.iter_per_batch * batch_size).long()
        self.iter_finish = False

    def __enter__(self):
        self.data_loader.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.data_loader.__exit__(exc_type, exc_val, exc_tb)

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.data_loader.dataset) // self.batch_size

    def _init(self):
        assert(self.buffer is None)
        self.iter = iter(data_loader)
        batch = next(self.iter)
        assert(isinstance(batch, (tuple, list)))
        bs = batch[0].shape[0]
        self.dim = len(batch)
        self.buffer = [None] * self.dim
        for i in range(self.dim):
            assert(isinstance(batch[i], torch.Tensor))
            assert(batch[i].shape[0] == self.dl_batch_size)
            size = list(batch[i].shape)
            size[0] = self.buffer_size * batch[i].shape[0]
            self.buffer[i] = batch[i].new_empty(size)
        index = 0
        for i in range(self.dim):
            self.buffer[i][index:index + bs] = batch[i]
        index += bs
        while index < self.elm:
            batch = next(self.iter)
            for i in range(self.dim):
                self.buffer[i][index:index + bs] = batch[i]
            index += bs

    def _update_sample(self):
        perm = torch.randperm(self.elm)[:self.batch_size]
        sample = []
        for i in range(self.dim):
            sample.append(self.buffer[i][perm].contiguous())
        self.perm[self.index * self.batch_size:(self.index+1)*self.batch_size].copy_(perm)
        self.index += 1
        if self.index >= self.iter_per_batch:
            batch = next(self.iter)
            for i in range(self.dim):
                self.buffer[i][self.perm] = batch[i]
            self.index = 0
        return sample

    def _sample(self):
        if self.index >= self.elm // self.batch_size:
            raise StopIteration()
        sample = []
        for i in range(self.dim):
            sample.append(self.buffer[i][self.index * self.batch_size:(self.index + 1)*self.batch_size].contiguous())
        self.index += 1
        return sample

    def _clear(self):
        self.buffer = None
        self.iter = None
        self.index = 0
        self.count = 0
        self.iter_finish = False

    def _next(self):
        if not self.iter_finish:
            try:
                if self.buffer is None:
                    self._init()
                return self._update_sample()
            except StopIteration:
                self.iter_finish = True
                self.index = 0
                return self._sample()
        else:
            try:
                return self._sample()
            except StopIteration:
                self._clear()
                raise StopIteration()

    def __next__(self):
        if self.count >= len(self):
            self._clear()
            raise StopIteration()
        self.count += 1
        return self._next()

if __name__ == "__main__":
    INTERNAL_BATCH_SIZE = 20
    BATCH_SIZE = 2
    BUFFER_SIZE = 4
    x = torch.ones(100, 1, 1, 1)
    y = torch.zeros(100, 1, 1, 1)
    for i in range(x.shape[0]):
        x[i].fill_(i)
        y[i].fill_(-i)
    dataset = torch.utils.data.TensorDataset(x, y)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=INTERNAL_BATCH_SIZE, shuffle=False, drop_last=True)
    wrapper = BufferedShuffleLoader(data_loader, batch_size=BATCH_SIZE, buffer_size=BUFFER_SIZE)
    for epoch in range(10):
        print("************** epoch", epoch)
        i = 0
        for xx, yy in wrapper:
            print(i)
            print(xx.view(-1), yy.view(-1))
            i += 1
        print("len", len(wrapper))
