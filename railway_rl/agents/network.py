import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, observation_dim: int, actions: int, hidden: list[int]) -> None:
        super().__init__()
        layers = []
        previous = observation_dim
        for width in hidden:
            layers.extend([nn.Linear(previous, width), nn.ReLU()])
            previous = width
        layers.append(nn.Linear(previous, actions))
        self.layers = nn.Sequential(*layers)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.layers(observation)
