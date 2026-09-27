from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


MODEL_NAME = "answerdotai/ModernBERT-base"
QUESTION_TOKEN = "[QUESTION]"
OPTION_TOKEN = "[OPTION]"


@dataclass(frozen=True)
class DecisionModelInputs:
    #Just a formatted way of input for the decision transformer 

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    option_positions: list[int]


class DecisionTokenizer:
    #Tokenizer itself (Using ModernBERT here)

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name
        )
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [QUESTION_TOKEN, OPTION_TOKEN]}
        ) #Adding the planned special tokens (questions,optionsss)

        option_token_id = self.tokenizer.convert_tokens_to_ids(OPTION_TOKEN)
        if option_token_id == self.tokenizer.unk_token_id:
            raise RuntimeError(f"Failed to add {OPTION_TOKEN} to the tokenizer")
        self.option_token_id = option_token_id

    def format(self, question: str, options: Sequence[str]) -> str: #Ordering stuff
        if not question.strip():
            raise ValueError("question must not be empty") #prolly might have to have some other fallback here later 
        if not options:
            raise ValueError("at least one option is required")
        if any(not option.strip() for option in options):
            raise ValueError("options must not be empty")

        option_text = " ".join(f"{OPTION_TOKEN} {option}" for option in options)
        return f"{QUESTION_TOKEN} {question} {option_text}"

    def encode(
        self,
        question: str,
        options: Sequence[str],
        *,
        max_length: int | None = None,
        truncation: bool = False,
    ) -> DecisionModelInputs: #COnverting it into tokens and getting option id pos
        text = self.format(question, options)
        if truncation and max_length is not None:
            question_token_id = self.tokenizer.convert_tokens_to_ids(QUESTION_TOKEN)
            cls_token_id = self.tokenizer.cls_token_id
            sep_token_id = self.tokenizer.sep_token_id
            if cls_token_id is None or sep_token_id is None:
                raise ValueError("ModernBERT tokenizer must define CLS and SEP tokens")
            question_ids = self.tokenizer(
                question,
                add_special_tokens=False,
            )["input_ids"]
            option_ids = [
                self.tokenizer(option, add_special_tokens=False)["input_ids"]
                for option in options
            ]
            special_token_count = self.tokenizer.num_special_tokens_to_add(pair=False)
            mandatory_token_count = 1 + len(options) + special_token_count
            if max_length < mandatory_token_count:
                raise ValueError(
                    "max_length is too small to preserve all decision markers"
                )

            segments = [question_ids, *option_ids]
            allocations = [0] * len(segments)
            remaining_budget = max_length - mandatory_token_count
            while remaining_budget > 0:
                allocated_token = False
                for index, segment in enumerate(segments):
                    if allocations[index] < len(segment):
                        allocations[index] += 1
                        remaining_budget -= 1
                        allocated_token = True
                        if remaining_budget == 0:
                            break
                if not allocated_token:
                    break

            token_ids = [question_token_id]
            token_ids.extend(question_ids[: allocations[0]])
            for index, current_option_ids in enumerate(option_ids, start=1):
                token_ids.append(self.option_token_id)
                token_ids.extend(current_option_ids[: allocations[index]])

            input_ids = torch.tensor(
                [cls_token_id, *token_ids, sep_token_id],
                dtype=torch.long,
            )
            attention_mask = torch.ones_like(input_ids)
            option_positions = (
                (input_ids == self.option_token_id)
                .nonzero(as_tuple=False)
                .flatten()
                .tolist()
            )
            return DecisionModelInputs(input_ids, attention_mask, option_positions)

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=truncation,
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        option_positions = (
            (input_ids == self.option_token_id).nonzero(as_tuple=False).flatten().tolist()
        )

        if not truncation and len(option_positions) != len(options):
            raise RuntimeError("Not all [OPTION] tokens were preserved during tokenization")

        return DecisionModelInputs(input_ids, attention_mask, option_positions)

    def load_resized_model(self, model_name: str = MODEL_NAME) -> PreTrainedModel:
        model = AutoModel.from_pretrained(model_name)
        model.resize_token_embeddings(len(self.tokenizer)) #Cause of the added special tokens 
        return model
