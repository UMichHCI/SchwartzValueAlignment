import gc
import inspect
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from dotenv import load_dotenv
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import DPOTrainer

try:
    from trl import DPOConfig
except ImportError:
    DPOConfig = None


MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct" # "Qwen/Qwen3-8B" "meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL_PREFIX = "qwen3" if "qwen3" in MODEL_NAME.lower() else "llama31"
MODEL_FAMILY_DIR = "qwen" if "qwen3" in MODEL_NAME.lower() else "llama"
RUN_DIR = Path(__file__).resolve().parent
ROOT_DIR = RUN_DIR.parents[1]
DATA_ROOT = Path(__file__).resolve().parents[2] / "datasets" / "final"
TRAINING_TYPE = "DPO_GOLD"
DATA_PATH = DATA_ROOT / "lora_dpo_gold.json"
OUTPUT_ROOT = RUN_DIR.parent / "lora_parameters_gold_gpt4o" / MODEL_FAMILY_DIR
DPO_BETA = 0.1
load_dotenv(ROOT_DIR / ".env")
HF_TOKEN = os.environ.get("HF_TOKEN")


def build_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )


def build_model(bnb_config, tokenizer):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        token=HF_TOKEN,
    )
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    return model


def build_peft_config():
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )


def build_dpo_training_args(output_dir):
    args_class = DPOConfig or TrainingArguments
    args_kwargs = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 2e-4,
        "num_train_epochs": 3,
        "logging_steps": 10,
        "save_steps": 100,
        "save_total_limit": 2,
        "bf16": torch.cuda.is_available(),
        "fp16": not torch.cuda.is_available(),
        "optim": "paged_adamw_32bit",
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "weight_decay": 0.0,
        "report_to": "none",
    }
    if DPOConfig is not None:
        args_kwargs["beta"] = DPO_BETA
    return args_class(**args_kwargs)


def load_dpo_datasets_by_value(data_path):
    grouped_data = json.loads(data_path.read_text(encoding="utf-8"))
    return {
        value_name: Dataset.from_dict(value_dataset)
        for value_name, value_dataset in sorted(grouped_data.items())
    }


def build_trainer(model, output_dir, value_dataset, tokenizer):
    trainer_kwargs = tokenizer_trainer_kwargs(DPOTrainer, tokenizer)
    if DPOConfig is None and "beta" in inspect.signature(DPOTrainer.__init__).parameters:
        trainer_kwargs["beta"] = DPO_BETA

    return DPOTrainer(
        model=model,
        args=build_dpo_training_args(output_dir),
        train_dataset=value_dataset,
        peft_config=build_peft_config(),
        **trainer_kwargs,
    )


def tokenizer_trainer_kwargs(trainer_class, tokenizer):
    trainer_params = inspect.signature(trainer_class.__init__).parameters
    if "processing_class" in trainer_params:
        return {"processing_class": tokenizer}
    if "tokenizer" in trainer_params:
        return {"tokenizer": tokenizer}
    return {}


def adapter_output_dir(training_type, value_name):
    training_type_slug = training_type.lower()
    return (
        OUTPUT_ROOT
        / training_type_slug
        / f"{MODEL_PREFIX}_{value_name}_{training_type_slug}_lora"
    )


def print_value_counts(counts, available_values):
    print("Examples per value:")
    for value_name in available_values:
        print(f"  {value_name}: {counts[value_name]}")


def main():
    training_type = TRAINING_TYPE
    data_path = DATA_PATH
    datasets_by_value = load_dpo_datasets_by_value(data_path)
    available_values = sorted(datasets_by_value)
    counts = {value_name: len(dataset) for value_name, dataset in datasets_by_value.items()}

    print(f"Training type: {training_type}")
    print(f"Loaded dataset from: {data_path}")
    print(f"Total examples loaded: {sum(counts.values())}")
    print_value_counts(counts, available_values)

    tokenizer = build_tokenizer()
    bnb_config = build_bnb_config()

    for value_name in available_values:
        value_dataset = datasets_by_value[value_name]
        output_dir = adapter_output_dir(training_type, value_name)

        print("\n" + "=" * 80)
        print(f"Training {training_type} adapter for value: {value_name}")
        print(f"Examples used: {len(value_dataset)}")
        print(f"Output dir: {output_dir}")
        print("=" * 80)

        model = build_model(bnb_config, tokenizer)
        trainer = build_trainer(model, output_dir, value_dataset, tokenizer)

        trainer.train()
        trainer.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        print(f"Saved LoRA adapter to: {output_dir}")

        del trainer
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
