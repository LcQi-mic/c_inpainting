import torch
from torch.nn import functional as F

class Loss(torch.nn.Module):
    def __init__(self, alpha_1=1., alpha_2=1.):
        super().__init__()
        self.mse_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.SmoothL1Loss()
        
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2

    def __call__(self, x, y):
        l1 = self.l1_loss(x, y)
        mse = self.mse_loss(x, y)
        total_loss = self.alpha_1 * l1 + self.alpha_2 * mse

        return total_loss, (l1, mse)