import argparse
import json
import os
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer


RUN_DIR = Path(__file__).resolve().parent
PERSONALIZATION_DIR = RUN_DIR.parents[1]
ROOT_DIR = PERSONALIZATION_DIR
DATASET_PATH = PERSONALIZATION_DIR / "datasets" / "final" / "activation_training_dataset.json"
STEERING_VECTOR_DIR = PERSONALIZATION_DIR / "activation_steering" / "steering_vectors"

if str(PERSONALIZATION_DIR) not in sys.path:
    sys.path.insert(0, str(PERSONALIZATION_DIR))

from activation_steering import SteeringDataset, SteeringVector
from activation_steering.utils import ContrastivePair


load_dotenv(ROOT_DIR / ".env")
DEFAULT_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_SETTINGS = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {
        "output_folder": "llama",
        "use_chat_template": True,
    },
    "Qwen/Qwen3-8B": {
        "output_folder": "qwen",
        "use_chat_template": True,
    },
    "google/gemma-2-9b": {
        "output_folder": "gemma",
        "use_chat_template": False,
    },
    "mistralai/Mistral-7B-Instruct-v0.3": {
        "output_folder": "mistral",
        "use_chat_template": True,
    },
}


class _HiddenStateOnlyWrapper(torch.nn.Module):
    """
    Minimal wrapper used only by this script so vector training can read
    hidden states without paying for the full LM-head logits projection.
    """

    def __init__(self, causal_lm_model: torch.nn.Module) -> None:
        super().__init__()
        self.model = causal_lm_model.model
        self.config = causal_lm_model.config

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def forward(self, *args, **kwargs):
        kwargs.pop("labels", None)
        kwargs.pop("logits_to_keep", None)
        kwargs["use_cache"] = False
        kwargs["return_dict"] = True
        return self.model(*args, **kwargs)


def load_dataset() -> dict[str, dict]:
    with DATASET_PATH.open() as file:
        return json.load(file)


def get_value_names(data: dict[str, dict]) -> list[str]:
    first_item = next(iter(data.values()))
    return list(first_item["tension_values_a"].keys())


def has_exact_one(value_scores: dict[str, float | dict], value_name: str) -> bool:
    raw_score = value_scores.get(value_name, 0)
    score = raw_score.get("score", 0) if isinstance(raw_score, dict) else raw_score
    return float(score) == 1.0


def build_value_examples(
    data: dict[str, dict],
    value_name: str,
) -> tuple[list[str], list[str], list[str], int]:
    questions: list[str] = []
    positive_suffixes: list[str] = []
    negative_suffixes: list[str] = []
    skipped_ambiguous = 0

    for scenario in data.values():
        in_a = has_exact_one(scenario["tension_values_a"], value_name)
        in_b = has_exact_one(scenario["tension_values_b"], value_name)

        if in_a and in_b:
            skipped_ambiguous += 1
            continue
        if not in_a and not in_b:
            continue

        questions.append(scenario["post"])
        if in_a:
            positive_suffixes.append(scenario["A"])
            negative_suffixes.append(scenario["B"])
        else:
            positive_suffixes.append(scenario["B"])
            negative_suffixes.append(scenario["A"])

    return questions, positive_suffixes, negative_suffixes, skipped_ambiguous


def print_value_example_counts(data: dict[str, dict], value_names: list[str]) -> None:
    print("Exact-1 example counts by value:")
    for value_name in value_names:
        questions, _, _, skipped_ambiguous = build_value_examples(data, value_name)
        ambiguous_note = (
            f", skipped ambiguous={skipped_ambiguous}"
            if skipped_ambiguous
            else ""
        )
        print(f"  {value_name}: {len(questions)} usable examples{ambiguous_note}")


def output_dir_for_model(model_name: str) -> Path:
    if model_name not in MODEL_SETTINGS:
        supported = ", ".join(sorted(MODEL_SETTINGS))
        raise ValueError(f"Unsupported model '{model_name}'. Supported models: {supported}")
    return STEERING_VECTOR_DIR / MODEL_SETTINGS[model_name]["output_folder"]


def make_behavior_vector(
    *,
    model,
    tokenizer,
    data: dict[str, dict],
    value_name: str,
    output_dir: Path,
    output_suffix: str,
    use_chat_template: bool,
) -> None:
    questions, positive_suffixes, negative_suffixes, skipped_ambiguous = build_value_examples(data, value_name)

    if not questions:
        raise ValueError(f"No exact-1 examples found for value '{value_name}'.")

    print(
        f"{value_name}: using {len(questions)} examples"
        + (f" (skipped {skipped_ambiguous} ambiguous scenarios)" if skipped_ambiguous else "")
    )

    behavior_dataset = SteeringDataset(
        tokenizer=tokenizer,
        examples=[(question, question) for question in questions],
        disable_suffixes=True,
        use_chat_template=use_chat_template,
    )
    behavior_dataset.suffixes = list(zip(positive_suffixes, negative_suffixes))
    behavior_dataset.formatted_dataset = [
        ContrastivePair(
            positive=pair.positive + positive_suffix,
            negative=pair.negative + negative_suffix,
        )
        for pair, positive_suffix, negative_suffix in zip(
            behavior_dataset.formatted_dataset_pre_populated,
            positive_suffixes,
            negative_suffixes,
        )
    ]
    print(f"{value_name}: paired dataset size {len(behavior_dataset.formatted_dataset)}")
    print("Actual positive example:", behavior_dataset.formatted_dataset[0].positive)
    print("Actual negative example:", behavior_dataset.formatted_dataset[0].negative)

    behavior_vector = SteeringVector.train(
        model=model,
        tokenizer=tokenizer,
        steering_dataset=behavior_dataset,
        method="pca_pairwise",
        accumulate_last_x_tokens="suffix-only",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{value_name}_behavior_vector{output_suffix}"
    behavior_vector.save(str(output_path))
    del behavior_vector, behavior_dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train behavior vectors from activation_training_dataset.json."
    )
    parser.add_argument(
        "values",
        nargs="*",
        help="Specific value names to train. If omitted, train all values in the dataset.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        choices=sorted(MODEL_SETTINGS),
        help=f"Hugging Face model name to load. Default: {DEFAULT_MODEL_NAME}",
    )
    parser.add_argument(
        "--output-suffix",
        default="_v2",
        help="Suffix appended to saved vector filenames before .svec.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_dataset()
    available_values = get_value_names(data)
    requested_values = args.values or available_values

    unknown_values = sorted(set(requested_values) - set(available_values))
    if unknown_values:
        raise ValueError(
            f"Unknown values: {unknown_values}. Available values: {available_values}"
        )

    print_value_example_counts(data, requested_values)
    output_dir = output_dir_for_model(args.model)
    use_chat_template = bool(MODEL_SETTINGS[args.model]["use_chat_template"])
    print(f"Saving behavior vectors to: {output_dir}")
    print(f"Using chat template for training prompts: {use_chat_template}")

    token = os.environ.get("HF_TOKEN")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        dtype=torch.float16,
        token=token,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=token)
    print(
        "Using hidden-state-only wrapper for vector training "
        "(skips full logits projection to reduce VRAM)."
    )
    training_model = _HiddenStateOnlyWrapper(model)

    for value_name in requested_values:
        make_behavior_vector(
            model=training_model,
            tokenizer=tokenizer,
            data=data,
            value_name=value_name,
            output_dir=output_dir,
            output_suffix=args.output_suffix,
            use_chat_template=use_chat_template,
        )


if __name__ == "__main__":
    main()
