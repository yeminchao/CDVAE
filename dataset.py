import torch
import numpy as np
from torch.utils.data import Dataset


class CDVAEDataSet(Dataset):
    def __init__(self, data, label):
        self.data = np.array(data, dtype=np.float32)
        self.label = np.array(label, dtype=np.float32)
        self.data = torch.from_numpy(self.data)
        self.label = torch.from_numpy(self.label)
        self.len = self.label.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.len
