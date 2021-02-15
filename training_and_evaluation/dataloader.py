from collections.abc import Sequence
import math
import torch
import numpy as np



class BatchTensors(Sequence):
    def __init__(self, bs, *tensors):
        self.tensors = tensors
        self.l = len(tensors[0])
        self.bs = bs

    def __getitem__(self, idx):
        start = idx * self.bs
        end = min(start + self.bs, self.l)
        if end - start <= 0:
            raise IndexError
        return [torch.from_numpy(np.array(t[start:end])).to(torch.float32) for t in self.tensors]

    def __len__(self):
        return math.ceil(self.l / self.bs)