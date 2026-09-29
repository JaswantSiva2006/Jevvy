import argparse
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .model import JevDecisionModel
from .tokenization import DecisionTokenizer

#training the decision block and the option scorer linear head using cross ent loss (to start with)

def collate_batch(
    batch: list[dict[str, Any]],
    decision_tokenizer: DecisionTokenizer,
    state_max_length: int,
    decision_max_length: int,
) -> dict[str, torch.Tensor]:
    state_input_ids = []
    state_attention_masks = []
    decision_input_ids = []
    decision_attention_masks = []
    option_positions = []

    for sample in batch:
        state_encoding = decision_tokenizer.tokenizer(
            sample["state"],
            return_tensors="pt",
            truncation=True,
            max_length=state_max_length,
        )
        decision_encoding = decision_tokenizer.encode(
            sample["question"],
            sample["options"],
            truncation=True,
            max_length=decision_max_length,
        )
        if len(decision_encoding.option_positions) != len(sample["options"]):
            raise ValueError("decision_max_length truncated one or more options")

        state_input_ids.append(state_encoding["input_ids"].squeeze(0))
        state_attention_masks.append(state_encoding["attention_mask"].squeeze(0))
        decision_input_ids.append(decision_encoding.input_ids)
        decision_attention_masks.append(decision_encoding.attention_mask)
        option_positions.append(
            torch.tensor(decision_encoding.option_positions, dtype=torch.long)
        )

    pad_token_id = decision_tokenizer.tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer must define a pad token")

    return {
        "state_input_ids": pad_sequence(
            state_input_ids,
            batch_first=True,
            padding_value=pad_token_id,
        ),
        "state_attention_mask": pad_sequence(
            state_attention_masks,
            batch_first=True,
            padding_value=0,
        ),
        "decision_input_ids": pad_sequence(
            decision_input_ids,
            batch_first=True,
            padding_value=pad_token_id,
        ),
        "decision_attention_mask": pad_sequence(
            decision_attention_masks,
            batch_first=True,
            padding_value=0,
        ),
        "option_positions": pad_sequence(
            option_positions,
            batch_first=True,
            padding_value=-1,
        ),
        "labels": torch.tensor([sample["label"] for sample in batch], dtype=torch.long),
    }


def move_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def run_epoch(
    model: JevDecisionModel,
    data_loader: DataLoader,
    loss_function: nn.CrossEntropyLoss,
    device: torch.device,
    optimizer: AdamW | None = None,
    epoch: int = 0,
    phase: str = "validation",
    log_interval: int = 500,
) -> tuple[float, float]:
    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch_index, batch in enumerate(data_loader, start=1):
        batch = move_batch(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits, _ = model(
                state_input_ids=batch["state_input_ids"],
                state_attention_mask=batch["state_attention_mask"],
                decision_input_ids=batch["decision_input_ids"],
                decision_attention_mask=batch["decision_attention_mask"],
                option_positions=batch["option_positions"],
            )
            loss = loss_function(logits, batch["labels"])

        if training:
            loss.backward()
            optimizer.step()

        batch_size = batch["labels"].size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=-1) == batch["labels"]).sum().item()
        total_examples += batch_size

        if log_interval > 0 and batch_index % log_interval == 0:
            print(
                f"epoch={epoch} "
                f"phase={phase} "
                f"batch={batch_index}/{len(data_loader)} "
                f"loss={total_loss / total_examples:.6f} "
                f"accuracy={total_correct / total_examples:.6f}",
                flush=True,
            )

    return total_loss / total_examples, total_correct / total_examples


def train(args: argparse.Namespace) -> None:
    from .dataset import load_jev_choice_dataset

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but PyTorch cannot see a GPU. Enable a Kaggle GPU accelerator."
        )
    device = torch.device(args.device)
    if device.type == "cuda":
        print(f"device={device} gpu={torch.cuda.get_device_name(device)}", flush=True)
    else:
        print(f"device={device}", flush=True)
    decision_tokenizer = DecisionTokenizer()
    train_dataset = load_jev_choice_dataset(args.train_split)
    validation_dataset = load_jev_choice_dataset(args.validation_split)
    collate = partial(
        collate_batch,
        decision_tokenizer=decision_tokenizer,
        state_max_length=args.state_max_length,
        decision_max_length=args.decision_max_length,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )

    model = JevDecisionModel(
        len(decision_tokenizer.tokenizer),
        max_decision_length=args.decision_max_length,
    ).to(device)
    embedding_weight = model.state_encoder.encoder.get_input_embeddings().weight
    other_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and parameter is not embedding_weight
    ]
    optimizer = AdamW(
        [
            {"params": other_parameters},
            {"params": [embedding_weight], "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
    )
    loss_function = nn.CrossEntropyLoss()
    best_validation_accuracy = float("-inf")
    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    latest_checkpoint_path = checkpoint_path.with_name("latest_model.pt")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            loss_function,
            device,
            optimizer,
            epoch,
            "train",
            args.log_interval,
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "tokenizer_vocab_size": len(decision_tokenizer.tokenizer),
                "arguments": vars(args),
            },
            latest_checkpoint_path,
        )
        validation_loss, validation_accuracy = run_epoch(
            model,
            validation_loader,
            loss_function,
            device,
            epoch=epoch,
            phase="validation",
            log_interval=args.log_interval,
        )
        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.6f} "
            f"train_accuracy={train_accuracy:.6f} "
            f"validation_loss={validation_loss:.6f} "
            f"validation_accuracy={validation_accuracy:.6f}"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_accuracy,
                    "tokenizer_vocab_size": len(decision_tokenizer.tokenizer),
                    "arguments": vars(args),
                },
                checkpoint_path,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="test")
    parser.add_argument("--state-max-length", type=int, default=8192)
    parser.add_argument("--decision-max-length", type=int, default=8192)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-path", default="checkpoints/best_model.pt")
    parser.add_argument("--log-interval", type=int, default=500)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
