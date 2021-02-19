from collections.abc import Sequence
import math
import torch
import numpy as np


class BatchTensors(Sequence):
    def __init__(self, *tensors, bs=4, mask=None):
        self.tensors = tensors
        self.l = len(tensors[0])
        self.bs = bs
        if mask is None:
            self.mask = np.ones(self.l, dtype=bool)
        else:
            self.mask = np.array(mask, dtype=bool)

    def __getitem__(self, idx):
        index = np.cumsum(self.mask)
        start = idx * self.bs
        end = min(start + self.bs, self.l)
        if end - start <= 0:
            raise IndexError
        batch_mask = np.where((start <= index) & (index < end), self.mask, False)
        return [torch.from_numpy(np.array(t[batch_mask])).to(torch.float32) for t in self.tensors]

    def __len__(self):
        return math.ceil(sum(self.mask) / self.bs)