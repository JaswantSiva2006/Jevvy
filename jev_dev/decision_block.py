import torch
from torch import nn

#Using a 12 head transformer block for the cross attention part 

class DecisionBlock(nn.Module):
    def __init__(self, hidden_size: int = 768, num_attention_heads: int = 12) -> None:
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            hidden_size,
            num_attention_heads,
            batch_first=True,
        )
        self.cross_attention = nn.MultiheadAttention(
            hidden_size,
            num_attention_heads,
            batch_first=True,
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.cross_attention_norm = nn.LayerNorm(hidden_size)
        self.feed_forward_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        decision_embeddings: torch.Tensor,
        H_state: torch.Tensor,
        decision_attention_mask: torch.Tensor | None = None,
        state_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        decision_padding_mask = (
            decision_attention_mask == 0
            if decision_attention_mask is not None
            else None
        )
        state_padding_mask = (
            state_attention_mask == 0 if state_attention_mask is not None else None
        )

        self_attention_output, _ = self.self_attention(
            decision_embeddings,
            decision_embeddings,
            decision_embeddings,
            key_padding_mask=decision_padding_mask,
            need_weights=False,
        )
        hidden_states = self.self_attention_norm(
            decision_embeddings + self_attention_output
        )

        cross_attention_output, _ = self.cross_attention(
            hidden_states,
            H_state,
            H_state,
            key_padding_mask=state_padding_mask,
            need_weights=False,
        )
        hidden_states = self.cross_attention_norm(
            hidden_states + cross_attention_output
        )

        feed_forward_output = self.feed_forward(hidden_states)
        return self.feed_forward_norm(hidden_states + feed_forward_output)
