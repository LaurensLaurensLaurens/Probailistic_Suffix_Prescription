import torch
from torch.utils import data

class DL_Dataset(data.Dataset):
    def __init__(self, x_case, x_event, prefix_len, t, y, weights):
        self.x_case = x_case.to(torch.float32).detach().requires_grad_(False)

        self.x_event = (
            x_event.to(torch.float32).detach().requires_grad_(False)
            if x_event is not None
            else torch.zeros_like(x_case).detach().requires_grad_(False)
        )

        self.prefix_len = (
            prefix_len.detach().requires_grad_(False)
            if prefix_len is not None
            else torch.zeros(x_case.size(0), dtype=torch.int64)
        )

        self.t = t.to(torch.float32).detach().requires_grad_(False)

        self.y = y.to(torch.float32).detach().requires_grad_(False)
        self.y = self.y.unsqueeze(1) if self.y.ndim == 1 else self.y
        if self.y.ndim == 2 and self.y.shape[1] > self.y.shape[0]:
            self.y = self.y.t()  # or self.y = self.y.T (both work)

        self.weights = (
            weights.to(torch.float32).detach().requires_grad_(False)
            if weights is not None
            else torch.ones_like(self.y).detach().requires_grad_(False)
        )

    def __len__(self):
        return self.x_event.size(0)

    def __getitem__(self, index):
        return (
            self.x_case[index],
            self.x_event[index],
            self.prefix_len[index],
            self.t[index],
            self.y[index],
            self.weights[index]
        )

class ML_Dataset():
    def __init__(self, x_case, x_event, prefix_len, t, y, weights):
        self.x_case = x_case.detach().cpu().numpy() if x_case is not None else None
        self.x_event = x_event.detach().cpu().numpy() if x_event is not None else None
        self.prefix_len = prefix_len.detach().cpu().numpy() if prefix_len is not None else None
        self.t = t.detach().cpu().numpy() if t is not None else None
        self.y = y.detach().cpu().numpy() if y is not None else None
        self.weights = weights.detach().cpu().numpy() if weights is not None else None

class RL_Dataset():
    def __init__(self, x_case, x_event, prefix_len, t, y, weights):
        self.x_case = x_case if x_case is not None else None
        self.x_event = x_event if x_event is not None else None
        self.prefix_len = prefix_len if prefix_len is not None else None
        self.t = t if t is not None else None
        self.y = y if y is not None else None
        self.weights = weights if weights is not None else None