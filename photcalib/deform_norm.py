import torch
import torch.nn.functional as F


torch.set_default_dtype(torch.float32)

## Define a moduel deform
from typing import Sequence
class deform(torch.nn.Module):
    def __init__(self, n_input: int, 
                 n_hidden: Sequence[int], 
                 n_output: int,
                 n_fields: int):
        super().__init__()
        
        # define a vector for field offset correction
        
        self.zpt = torch.nn.Parameter(torch.zeros(n_fields))
     
        # define a sequence of layers for FoV correction
                                      
        self.sequence  = torch.nn.Sequential()
                                      
        self.sequence.add_module("input", torch.nn.Linear(n_input, n_hidden[0]))
          
        self.sequence.add_module("hidden1",torch.nn.Linear(n_hidden[0], n_hidden[1]))
        self.sequence.add_module("activation",torch.nn.SELU())
        self.sequence.add_module("norm", torch.nn.BatchNorm1d(n_hidden[1]))

        
        self.sequence.add_module("hidden2",torch.nn.Linear(n_hidden[1], n_hidden[2]))
        self.sequence.add_module("activation",torch.nn.SELU())
        self.sequence.add_module("norm", torch.nn.BatchNorm1d(n_hidden[2]))
        
        self.sequence.add_module("hidden3",torch.nn.Linear(n_hidden[2], n_hidden[3]))
        self.sequence.add_module("activation",torch.nn.SELU())
        self.sequence.add_module("norm", torch.nn.BatchNorm1d(n_hidden[3]))
            
        self.sequence.add_module("output",  torch.nn.Linear(n_hidden[-1], n_output))
        

    def forward(self, x_f: torch.Tensor, y_f: torch.Tensor, id_f: torch.Tensor) -> torch.Tensor:
        
        z_f0 = torch.stack([x_f, y_f]).T
        
        for layer in self.sequence:
            
            z_f0 = layer(z_f0)
                
        z_f = z_f0 + torch.unsqueeze(self.zpt[id_f], dim=1)
        
        return z_f