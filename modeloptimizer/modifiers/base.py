import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class ModifierBase(ABC):
    def __init__(self, device: torch.device = torch.device("cuda")):
        super().__init__()
        self.device = device

    def forward(self, x):
        return x

    