import torch.utils.data as data_utils
import numpy as np

class CustomDataLoader(data_utils.Dataset):
    def __init__(self, x: np.ndarray, cons: np.ndarray, y: np.ndarray, c: np.ndarray) -> None:
        self.x = x
        self.cons = cons
        self.y = y
        self.c = c

        super().__init__()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.cons[index], self.y[index], self.c[index]