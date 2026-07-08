import torch
import torch.nn as nn

class SwiGlU(nn.Module):
    def __init__(self, d_hidden : int, d_ff : int): #en général, d_ff = 8/3d_model
        super().__init__()
        self.w_in = nn.Linear(d_hidden,d_ff)