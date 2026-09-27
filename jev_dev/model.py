import torch
from torch import nn

from .decision_block import DecisionBlock
from .decision_embedding import DecisionEmbedding
from .option_scorer import OptionScorer
from .state_encoder import StateEncoder

#Connecting the diff layers according the arch i feel is Jev lol

class JevDecisionModel(nn.Module):
    def __init__(self, tokenizer_vocab_size: int) -> None:
        super().__init__()
        self.state_encoder = StateEncoder()
        embedding_layer = self.state_encoder.encoder.get_input_embeddings()
        self.pretrained_vocab_size = embedding_layer.num_embeddings
        if tokenizer_vocab_size < self.pretrained_vocab_size:
            raise ValueError("tokenizer vocabulary cannot be smaller than the model vocabulary")
        if tokenizer_vocab_size > self.pretrained_vocab_size:
            self.state_encoder.encoder.resize_token_embeddings(tokenizer_vocab_size)
        embedding_layer = self.state_encoder.encoder.get_input_embeddings()

        for parameter in self.state_encoder.parameters():
            parameter.requires_grad = False
        embedding_layer.weight.requires_grad = True
        embedding_layer.weight.register_hook(self._mask_pretrained_embedding_gradients)

        self.decision_embedding = DecisionEmbedding(embedding_layer)
        self.decision_blocks = nn.ModuleList([DecisionBlock(), DecisionBlock()])
        self.option_scorer = OptionScorer()

    def _mask_pretrained_embedding_gradients(
        self,
        gradient: torch.Tensor,
    ) -> torch.Tensor:
        gradient[: self.pretrained_vocab_size].zero_()
        return gradient

    def train(self, mode: bool = True) -> "JevDecisionModel":
        super().train(mode)
        self.state_encoder.eval()
        return self

    def forward(
        self,
        state_input_ids: torch.Tensor,
        state_attention_mask: torch.Tensor,
        decision_input_ids: torch.Tensor,
        decision_attention_mask: torch.Tensor,
        option_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            H_state = self.state_encoder(state_input_ids, state_attention_mask)

        hidden_states = self.decision_embedding(decision_input_ids)
        for decision_block in self.decision_blocks:
            hidden_states = decision_block(
                hidden_states,
                H_state,
                decision_attention_mask,
                state_attention_mask,
            )

        return self.option_scorer(hidden_states, option_positions)
