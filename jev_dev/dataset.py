import json
from collections.abc import Mapping
from typing import Any

from datasets import load_dataset
from torch.utils.data import Dataset


DATASET_NAME = "avbiswas/bev-decision-150K"

#Cleaning up dataset

class JevChoiceDataset(Dataset):
    def __init__(self, split: str = "train") -> None:
        rows = load_dataset(DATASET_NAME, split=split)
        self.samples: list[dict[str, Any]] = []

        for state_index, row in enumerate(rows):
            questions = row["questions_json"]
            if isinstance(questions, str):
                questions = json.loads(questions)
            if not isinstance(questions, Mapping):
                raise ValueError(f"Invalid questions_json at row {state_index}")

            state_id = f"{split}:{state_index}"
            for question_id, question_data in questions.items():
                if not isinstance(question_data, Mapping):
                    raise ValueError(
                        f"Invalid question {question_id} at row {state_index}"
                    )
                if question_data.get("type") != "choice":
                    continue

                criteria = question_data.get("criteria")
                if not isinstance(criteria, Mapping) or not criteria:
                    raise ValueError(
                        f"Choice question {question_id} has invalid options"
                    )

                option_keys = [str(key) for key in criteria.keys()]
                options = list(criteria.values())
                provided_label = str(question_data.get("label"))
                if provided_label not in option_keys:
                    raise ValueError(
                        f"Label {provided_label!r} is not an option for question "
                        f"{question_id} at row {state_index}"
                    )

                self.samples.append(
                    {
                        "state_id": state_id,
                        "question_id": str(question_id),
                        "state": row["state"],
                        "question": question_data["instructions"],
                        "options": options,
                        "option_keys": option_keys,
                        "label": option_keys.index(provided_label),
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


def load_jev_choice_dataset(split: str = "train") -> JevChoiceDataset:
    return JevChoiceDataset(split=split)
