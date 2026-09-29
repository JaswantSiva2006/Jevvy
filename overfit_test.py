import argparse
import math
import random
from functools import partial

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from jev_dev.dataset import load_jev_choice_dataset
from jev_dev.model import JevDecisionModel
from jev_dev.tokenization import OPTION_TOKEN, QUESTION_TOKEN, DecisionTokenizer
from jev_dev.train import collate_batch, move_batch


EARLY_STOP_ACCURACY = 0.98
PASS_ACCURACY = 0.95
REPORT_INTERVAL = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--state-max-length", type=int, default=512)
    parser.add_argument("--decision-max-length", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-count", type=int, choices=[8, 100], default=100)
    args = parser.parse_args()
    if args.max_steps < 1 or args.max_steps > 2000:
        parser.error("--max-steps must be between 1 and 2000")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_samples(
    dataset: Dataset,
    tokenizer: DecisionTokenizer,
    decision_max_length: int,
    sample_count: int,
) -> None:
    if len(dataset) != sample_count:
        raise RuntimeError(f"Expected exactly {sample_count} samples, got {len(dataset)}")
    for sample_index, sample in enumerate(dataset):
        option_count = len(sample["options"])
        label = int(sample["label"])
        if not 0 <= label < option_count:
            raise ValueError(
                f"Sample {sample_index} has label {label} outside [0, {option_count})"
            )
        if option_count < 2:
            raise ValueError(
                f"Sample {sample_index} needs at least two options to measure incorrect logits"
            )
        encoding = tokenizer.encode(
            sample["question"],
            sample["options"],
            truncation=True,
            max_length=decision_max_length,
        )
        if len(encoding.option_positions) != option_count:
            raise ValueError(
                f"Sample {sample_index} has {len(encoding.option_positions)} option positions "
                f"for {option_count} options"
            )
        for position in encoding.option_positions:
            if encoding.input_ids[position].item() != tokenizer.option_token_id:
                raise ValueError(
                    f"Sample {sample_index} option_position {position} does not point to "
                    f"{OPTION_TOKEN}"
                )
    print(f"Validated exactly {len(dataset)} Choice training samples", flush=True)
    print(f"Verified all labels and {OPTION_TOKEN} positions", flush=True)


def verify_architecture(
    model: JevDecisionModel,
    data_loader: DataLoader,
    device: torch.device,
) -> None:
    batch = move_batch(next(iter(data_loader)), device)
    captured = {}

    def capture_representations(module, inputs) -> None:
        captured["option_representations"] = inputs[0].detach()

    handle = model.option_scorer.scorer.register_forward_pre_hook(
        capture_representations
    )
    model.eval()
    with torch.no_grad():
        logits, _ = model(
            state_input_ids=batch["state_input_ids"],
            state_attention_mask=batch["state_attention_mask"],
            decision_input_ids=batch["decision_input_ids"],
            decision_attention_mask=batch["decision_attention_mask"],
            option_positions=batch["option_positions"],
        )
    handle.remove()
    model.train()

    option_representations = captured["option_representations"]
    example_spans = []
    example_distances = []
    example_logits = []
    example_stds = []
    for batch_index in range(logits.size(0)):
        positions = batch["option_positions"][batch_index]
        positions = positions[positions >= 0]
        sequence_end = batch["decision_attention_mask"][batch_index].sum().item() - 1
        spans = []
        for option_index, position in enumerate(positions):
            start = position.item() + 1
            end = (
                positions[option_index + 1].item()
                if option_index + 1 < positions.numel()
                else sequence_end
            )
            spans.append(list(range(start, end)))
        if len({position for span in spans for position in span}) != sum(
            len(span) for span in spans
        ):
            raise AssertionError("option text spans overlap")
        if len({tuple(span) for span in spans}) != len(spans):
            raise AssertionError("different options have identical token positions")

        representations = option_representations[batch_index, : positions.numel()]
        distances = torch.pdist(representations.float(), p=2)
        if distances.numel() == 0 or not (distances > 0).all():
            raise AssertionError("pairwise option representation distance is zero")
        valid_logits = logits[batch_index, : positions.numel()]
        logit_std = valid_logits.float().std(unbiased=False)
        if torch.equal(valid_logits, valid_logits[0].expand_as(valid_logits)):
            raise AssertionError("option logits are identical")
        if not logit_std > 0:
            raise AssertionError("option logit standard deviation is not positive")
        if batch_index == 0:
            example_spans = spans
            example_distances = distances.cpu().tolist()
            example_logits = valid_logits.cpu().tolist()
            example_stds = logit_std.item()

    print("Architecture diagnostics passed", flush=True)
    print(f"example_option_token_spans={example_spans}", flush=True)
    print(
        f"example_pairwise_option_representation_distances={example_distances}",
        flush=True,
    )
    print(f"example_option_logits={example_logits}", flush=True)
    print(f"example_option_logit_std={example_stds:.6e}", flush=True)


def gradient_norm(parameters) -> float:
    squared_norm = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            squared_norm += parameter.grad.detach().float().norm(2).item() ** 2
    return math.sqrt(squared_norm)


def print_gradient_norms(
    model: JevDecisionModel,
    tokenizer: DecisionTokenizer,
) -> None:
    print("Gradient norms after first backward pass:", flush=True)
    for index, block in enumerate(model.decision_blocks):
        print(
            f"decision_blocks.{index}: {gradient_norm(block.parameters()):.6e}",
            flush=True,
        )
    print(
        f"option_scorer: {gradient_norm(model.option_scorer.parameters()):.6e}",
        flush=True,
    )
    embedding_weight = model.state_encoder.encoder.get_input_embeddings().weight
    if embedding_weight.requires_grad:
        question_token_id = tokenizer.tokenizer.convert_tokens_to_ids(QUESTION_TOKEN)
        option_token_id = tokenizer.tokenizer.convert_tokens_to_ids(OPTION_TOKEN)
        if embedding_weight.grad is None:
            print(f"{QUESTION_TOKEN} embedding: no gradient", flush=True)
            print(f"{OPTION_TOKEN} embedding: no gradient", flush=True)
        else:
            print(
                f"{QUESTION_TOKEN} embedding: "
                f"{embedding_weight.grad[question_token_id].detach().float().norm(2).item():.6e}",
                flush=True,
            )
            print(
                f"{OPTION_TOKEN} embedding: "
                f"{embedding_weight.grad[option_token_id].detach().float().norm(2).item():.6e}",
                flush=True,
            )


def evaluate(
    model: JevDecisionModel,
    data_loader: DataLoader,
    loss_function: nn.CrossEntropyLoss,
    device: torch.device,
) -> tuple[float, float, float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    correct_logits = []
    highest_incorrect_logits = []
    with torch.no_grad():
        for batch in data_loader:
            batch = move_batch(batch, device)
            logits, _ = model(
                state_input_ids=batch["state_input_ids"],
                state_attention_mask=batch["state_attention_mask"],
                decision_input_ids=batch["decision_input_ids"],
                decision_attention_mask=batch["decision_attention_mask"],
                option_positions=batch["option_positions"],
            )
            labels = batch["labels"]
            loss = loss_function(logits, labels)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
            total_examples += batch_size
            correct_logits.append(logits.gather(1, labels.unsqueeze(1)).squeeze(1))
            incorrect_mask = torch.ones_like(logits, dtype=torch.bool)
            incorrect_mask.scatter_(1, labels.unsqueeze(1), False)
            highest_incorrect_logits.append(
                logits.masked_fill(~incorrect_mask, float("-inf")).max(dim=1).values
            )
    model.train()
    return (
        total_loss / total_examples,
        total_correct / total_examples,
        torch.cat(correct_logits).mean().item(),
        torch.cat(highest_incorrect_logits).mean().item(),
    )


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot see a CUDA GPU")
    device = torch.device(args.device)
    set_seed(args.seed)
    if device.type == "cuda":
        print(f"Device: {device} ({torch.cuda.get_device_name(device)})", flush=True)
    else:
        print(f"Device: {device}", flush=True)

    tokenizer = DecisionTokenizer()
    dataset = load_jev_choice_dataset("train", max_samples=args.sample_count)
    if len(dataset) < args.sample_count:
        raise RuntimeError(
            f"Training split contains only {len(dataset)} Choice samples"
        )
    validate_samples(
        dataset,
        tokenizer,
        args.decision_max_length,
        args.sample_count,
    )

    collate = partial(
        collate_batch,
        decision_tokenizer=tokenizer,
        state_max_length=args.state_max_length,
        decision_max_length=args.decision_max_length,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    evaluation_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )

    model = JevDecisionModel(
        len(tokenizer.tokenizer),
        max_decision_length=args.decision_max_length,
    ).to(device)
    print("Trainable parameters:", flush=True)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            print(f"{name} shape={tuple(parameter.shape)}", flush=True)

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
    verify_architecture(model, evaluation_loader, device)
    train_iterator = iter(train_loader)
    final_accuracy = 0.0
    final_step = 0

    model.train()
    for step in range(1, args.max_steps + 1):
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(
            state_input_ids=batch["state_input_ids"],
            state_attention_mask=batch["state_attention_mask"],
            decision_input_ids=batch["decision_input_ids"],
            decision_attention_mask=batch["decision_attention_mask"],
            option_positions=batch["option_positions"],
        )
        loss = loss_function(logits, batch["labels"])
        loss.backward()
        if step == 1:
            print_gradient_norms(model, tokenizer)
        optimizer.step()
        final_step = step

        if step % REPORT_INTERVAL == 0 or step == args.max_steps:
            report_loss, final_accuracy, mean_correct, mean_incorrect = evaluate(
                model,
                evaluation_loader,
                loss_function,
                device,
            )
            print(
                f"step={step} loss={report_loss:.6f} "
                f"training_accuracy={final_accuracy:.4%} "
                f"mean_correct_option_logit={mean_correct:.6f} "
                f"mean_highest_incorrect_option_logit={mean_incorrect:.6f}",
                flush=True,
            )
            if final_accuracy >= EARLY_STOP_ACCURACY:
                print(
                    f"Early stop: training accuracy reached {final_accuracy:.2%}",
                    flush=True,
                )
                break

    if final_accuracy >= PASS_ACCURACY:
        print(
            f"RESULT: PASS - model learned the {args.sample_count}-sample set to "
            f"{final_accuracy:.2%} accuracy in {final_step} optimizer steps",
            flush=True,
        )
    else:
        print(
            f"RESULT: FAILURE - model reached only {final_accuracy:.2%} training "
            f"accuracy after {final_step} optimizer steps; required at least 95.00%",
            flush=True,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
