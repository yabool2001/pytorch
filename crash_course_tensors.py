import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print ( f"{device=}" )
print ( f"{torch.__version__=}" )

scalar = torch.empty ( size = ( 1 , ) ) ; print ( f"{scalar=}" )
vector = torch.empty ( size = ( 2 , ) ) ; print ( f"{vector=}" )
matrix = torch.empty ( size = ( 2 , 3 ) ) ; print ( f"{matrix=}" )
tensor = torch.empty ( size = ( 2 , 3 , 4 ) ) ; print ( f"{tensor=}" ) # 3 dimensions
# Alternatively, we can have it all simpler without the size, e.g.:
tensor2 = torch.empty ( 2, 3, 4 ) ; print ( f"{tensor2=}" )