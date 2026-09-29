import torch
from torch import nn

#Using the same encoding matrix as bert for the ques+options too 

class DecisionEmbedding(nn.Module):
    def __init__(self, embedding_layer: nn.Module, max_sequence_length: int) -> None:
        super().__init__()
        self.embedding_layer = embedding_layer
        self.position_embedding = nn.Embedding(
            max_sequence_length,
            embedding_layer.embedding_dim,
        )
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        sequence_length = input_ids.size(1)
        if sequence_length > self.position_embedding.num_embeddings:
            raise ValueError(
                f"Decision sequence length {sequence_length} exceeds configured maximum "
                f"{self.position_embedding.num_embeddings}"
            )
        position_ids = torch.arange(sequence_length, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        return self.embedding_layer(input_ids) + self.position_embedding(position_ids)
