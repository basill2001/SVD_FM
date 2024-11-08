import torch.utils.data as data_utils

class CustomDataLoader(data_utils.Dataset):
    def __init__(self, x, cons, y, c) -> None:
        self.x = x
        self.cons = cons
        self.y = y
        self.c = c

        super().__init__()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.cons[index], self.y[index], self.c[index]