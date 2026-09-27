import torch
from torch import nn

#Using the same encoding matrix as bert for the ques+options too 

class DecisionEmbedding(nn.Module):
    def __init__(self, embedding_layer: nn.Module) -> None:
        super().__init__()
        self.embedding_layer = embedding_layer

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding_layer(input_ids)
