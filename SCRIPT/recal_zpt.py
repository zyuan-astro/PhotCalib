import torch
import torch.nn.functional as F


torch.set_default_dtype(torch.float64)

from typing import Sequence


class get_zpt(torch.nn.Module):
    def __init__(self, n_fields: int):
        super().__init__()
        
        # define a vector for field offset correction
        
        self.zpt = torch.nn.Parameter(torch.zeros(n_fields))
             

    def forward(self, id_f: torch.Tensor):
        
        
        z_f = torch.unsqueeze(self.zpt[id_f], dim=1)

        
        return z_f

