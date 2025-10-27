
import torch
import torch.nn as nn

class LinearProbe(nn.Module):
    """sigma(wˆT h + b) scalar probe in [0,1]."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)
        self.calib_A = None
        self.calib_B = None

    def forward(self, h: torch.Tensor):
        logit = self.linear(h).squeeze(-1)
        c = torch.sigmoid(logit)
        return c, logit

    def predict_calibrated(self, h: torch.Tensor):
        c, logit = self.forward(h)
        if (self.calib_A is not None) and (self.calib_B is not None):
            z = self.calib_A * logit + self.calib_B
            return torch.sigmoid(z)
        return c


