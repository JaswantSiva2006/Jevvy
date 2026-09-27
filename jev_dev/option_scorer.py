from collections.abc import Sequence

import torch
from torch import nn

#Using a simple linear head to see which option closely aligjns and then assigning probab

class OptionScorer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scorer = nn.Linear(768, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        option_positions: torch.Tensor | Sequence[Sequence[int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(option_positions, torch.Tensor):
            if len(option_positions) != hidden_states.size(0):
                raise ValueError("option_positions must have one entry per batch item")
            if any(len(positions) == 0 for positions in option_positions):
                raise ValueError("each batch item must contain at least one option")
            max_options = max(len(positions) for positions in option_positions)
            positions_tensor = torch.full(
                (hidden_states.size(0), max_options),
                -1,
                dtype=torch.long,
                device=hidden_states.device,
            )
            for batch_index, positions in enumerate(option_positions):
                positions_tensor[batch_index, : len(positions)] = torch.as_tensor(
                    positions,
                    dtype=torch.long,
                    device=hidden_states.device,
                )
            option_positions = positions_tensor
        else:
            option_positions = option_positions.to(
                device=hidden_states.device,
                dtype=torch.long,
            )

        if option_positions.ndim != 2:
            raise ValueError("option_positions must be a two-dimensional tensor")
        if option_positions.size(0) != hidden_states.size(0):
            raise ValueError("option_positions batch size must match hidden_states")

        option_mask = option_positions >= 0
        if not option_mask.any(dim=1).all():
            raise ValueError("each batch item must contain at least one option")
        if (option_positions[option_mask] >= hidden_states.size(1)).any():
            raise ValueError("option position exceeds the sequence length")

        gather_indices = option_positions.clamp_min(0).unsqueeze(-1).expand(-1, -1, 768)
        option_hidden_states = hidden_states.gather(1, gather_indices)
        logits = self.scorer(option_hidden_states).squeeze(-1)
        logits = logits.masked_fill(~option_mask, float("-inf"))
        probabilities = torch.softmax(logits, dim=-1)
        return logits, probabilities
