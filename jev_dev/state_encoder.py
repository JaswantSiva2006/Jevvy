import torch
from torch import nn
from transformers import AutoModel

#Encoding the tokenized state (could use the same encoding matrix for ques+opt)

class StateEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden_size = 768
        self.encoder = AutoModel.from_pretrained("answerdotai/ModernBERT-base")
        if self.encoder.config.hidden_size != self.hidden_size:
            raise ValueError(
                f"Expected hidden size {self.hidden_size}, got {self.encoder.config.hidden_size}"
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.last_hidden_state
