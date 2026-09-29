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
        attention_mask: torch.Tensor | None = None,
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

        if attention_mask is None:
            sequence_ends = torch.full(
                (hidden_states.size(0),),
                hidden_states.size(1),
                dtype=torch.long,
                device=hidden_states.device,
            )
        else:
            if attention_mask.shape != hidden_states.shape[:2]:
                raise ValueError("attention_mask shape must match hidden states")
            sequence_ends = attention_mask.to(hidden_states.device).sum(dim=1) - 1

        option_hidden_states = hidden_states.new_zeros(
            hidden_states.size(0),
            option_positions.size(1),
            hidden_states.size(2),
        )
        for batch_index in range(hidden_states.size(0)):
            valid_positions = option_positions[batch_index][option_mask[batch_index]]
            for option_index, marker_position in enumerate(valid_positions):
                span_start = marker_position.item() + 1
                if option_index + 1 < valid_positions.numel():
                    span_end = valid_positions[option_index + 1].item()
                else:
                    span_end = sequence_ends[batch_index].item()
                if span_start >= span_end:
                    raise ValueError(
                        f"option {option_index} in batch item {batch_index} has no text tokens"
                    )
                option_hidden_states[batch_index, option_index] = hidden_states[
                    batch_index, span_start:span_end
                ].mean(dim=0)

        logits = self.scorer(option_hidden_states).squeeze(-1)
        logits = logits.masked_fill(~option_mask, float("-inf"))
        probabilities = torch.softmax(logits, dim=-1)
        return logits, probabilities
