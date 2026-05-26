#!/usr/bin/env python3
"""Generate one scenario response with a base model, activation steering, or LoRA.

Edit the settings at the top of this file, then run:

    python generate_response.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

load_dotenv(ROOT_DIR / ".env")

STEERING_VECTOR_DIR = ROOT_DIR / "activation_steering" / "steering_vectors"
LORA_GOLD_ADAPTER_DIR = ROOT_DIR / "lora" / "lora_parameters_gold_gpt4o"
SCENARIOS_PATH = ROOT_DIR / "datasets" / "scenarios_eval.json"

SYSTEM_PROMPT = """
You are a supportive, practical, and logically rigorous advisor.

When the user asks for advice between competing paths, analyze the trade-offs
carefully and recommend one direction clearly.
Base your reasoning on the information provided. Do not invent facts.
You must take a clear side and give a decisive recommendation.
Do not say that both options are equally good, equally valid, or that the user
should simply balance both without stating which way to lean.
If the trade-offs are close, still choose the direction you think the user
should lean toward most.
The final answer should make it obvious which option or side you recommend.
Keep the answer concise, action-oriented, and limited to 2-4 sentences.
Do not mention value labels, score numbers, or Schwartz terminology explicitly.
""".strip()

MODEL_SETTINGS = {
    "llama": {
        "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "vector_dir": STEERING_VECTOR_DIR / "llama",
        "vector_name_suffix": "_behavior_vector_v2",
        "behavior_vector_strength": 1.0,
        "behavior_layer_ids": [[15, 16, 17, 18, 19, 20], [16, 17, 18, 19, 20, 21]],
        "adapter_prefix": "llama31",
        "gold_adapter_dir": LORA_GOLD_ADAPTER_DIR / "llama",
        "chat_template_kwargs": {},
    },
    "qwen": {
        "model_id": "Qwen/Qwen3-8B",
        "vector_dir": STEERING_VECTOR_DIR / "qwen",
        "vector_name_suffix": "_behavior_vector_v2",
        "behavior_vector_strength": 15.0,
        "behavior_layer_ids": [[19, 20, 21, 22], [23, 24, 25, 26]],
        "adapter_prefix": "qwen3",
        "gold_adapter_dir": LORA_GOLD_ADAPTER_DIR / "qwen",
        "chat_template_kwargs": {"enable_thinking": False},
    },
}

# ----------------------
# Top-of-file run settings.

BACKEND = "lora"  # "base", "activation_steering", or "lora"
MODEL_FAMILY = "llama"  # "qwen" or "llama"
SCENARIO_KEY = "scenario_103"

STEERING_VALUES = [
    # "self_direction_thought",
    # "self_direction_action",
    # "stimulation",
    # "hedonism",
    # "achievement",
    # "power_dominance",
    # "power_resources",
    # "face",
    # "security_personal",
    # "security_societal",
    # "tradition",
    # "conformity_rules",
    # "conformity_interpersonal",
    # "humility",
    "benevolence_caring",
    "benevolence_dependability",
    # "universalism_concern",
    # "universalism_nature",
    # "universalism_tolerance",
]

# LoRA-only settings. This repo supports only DPO_GOLD adapters.
LORA_ADAPTER_WEIGHTS: list[float] | None = None

# Activation-steering-only overrides. Use None for model defaults.
BEHAVIOR_VECTOR_STRENGTH: float | None = None
BEHAVIOR_LAYER_IDS: list[int] | list[list[int]] | None = None

QWEN_THINKING = False

GEN_SETTINGS = {
    "do_sample": False,
    "max_new_tokens": 300,
    "repetition_penalty": 1.1,
    # Used only when do_sample is True.
    "temperature": 0.7,
    "top_p": 0.9,
}

# ----------------------


def hf_token_kwargs() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN")
    return {"token": token} if token else {}


def load_model_and_tokenizer(settings: dict[str, Any]):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        **hf_token_kwargs(),
    }
    print(f"Loading model: {settings['model_id']}")
    model = AutoModelForCausalLM.from_pretrained(settings["model_id"], **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(settings["model_id"], **hf_token_kwargs())
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generation_settings(tokenizer) -> dict[str, Any]:
    settings = {
        "do_sample": bool(GEN_SETTINGS["do_sample"]),
        "max_new_tokens": int(GEN_SETTINGS["max_new_tokens"]),
        "repetition_penalty": float(GEN_SETTINGS["repetition_penalty"]),
        "pad_token_id": tokenizer.eos_token_id,
    }
    if settings["do_sample"]:
        settings["temperature"] = float(GEN_SETTINGS["temperature"])
        settings["top_p"] = float(GEN_SETTINGS["top_p"])
    return settings


def load_scenario() -> dict[str, Any]:
    scenarios = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))
    if SCENARIO_KEY not in scenarios:
        available = ", ".join(sorted(scenarios)[:10])
        raise ValueError(
            f"Unknown SCENARIO_KEY {SCENARIO_KEY!r}. "
            f"First available scenario keys: {available}"
        )
    return scenarios[SCENARIO_KEY]


def scenario_prompt(scenario: dict[str, Any]) -> str:
    return str(scenario["post"]).strip()


def active_value_scores(value_scores: dict[str, Any]) -> list[tuple[str, float]]:
    active_scores = []
    for value_name, raw_score in value_scores.items():
        score = raw_score.get("score", 0) if isinstance(raw_score, dict) else raw_score
        score = float(score or 0)
        if score > 0:
            active_scores.append((value_name, score))
    return active_scores


def print_scenario_details(scenario: dict[str, Any]) -> None:
    print("\n=== Scenario ===")
    print(scenario_prompt(scenario))

    print("\n=== Side A ===")
    print(str(scenario.get("A", "")).strip())
    print("Associated values:", format_value_scores(active_value_scores(scenario.get("tension_values_a", {}))))

    print("\n=== Side B ===")
    print(str(scenario.get("B", "")).strip())
    print("Associated values:", format_value_scores(active_value_scores(scenario.get("tension_values_b", {}))))


def format_value_scores(value_scores: list[tuple[str, float]]) -> str:
    if not value_scores:
        return "none"
    return ", ".join(f"{value_name}={score:g}" for value_name, score in value_scores)


def split_out_thinking(text: str) -> tuple[str, str]:
    raw = (text or "").strip()
    thinking_chunks = [
        match.group(1).strip()
        for match in re.finditer(r"<think>(.*?)</think>", raw, flags=re.IGNORECASE | re.DOTALL)
        if match.group(1).strip()
    ]
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.IGNORECASE | re.DOTALL)
    return cleaned.strip(), "\n\n".join(thinking_chunks).strip()


def generate_with_transformers(
    *,
    model,
    tokenizer,
    settings: dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    chat_template_kwargs: dict[str, Any],
) -> tuple[str, str]:
    import torch

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        **chat_template_kwargs,
    )
    inputs = inputs.to(model.device)
    input_len = inputs.shape[-1]
    with torch.no_grad():
        outputs = model.generate(inputs, **settings)
    decoded = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return split_out_thinking(decoded)


def resolve_layer_ids(layer_ids_config: list[Any] | None, default_layers: list[Any], n_vectors: int) -> list[list[int]]:
    layer_config = layer_ids_config if layer_ids_config is not None else default_layers
    if not layer_config:
        raise ValueError("Layer configuration cannot be empty.")
    if all(isinstance(item, int) for item in layer_config):
        return [list(layer_config)] * n_vectors
    if all(isinstance(item, list) for item in layer_config):
        if len(layer_config) == 1:
            return [list(layer_config[0])] * n_vectors
        if len(layer_config) == n_vectors:
            return [list(item) for item in layer_config]
    raise ValueError("Layers must be list[int], one nested list, or one nested list per value.")


def run_activation_steering(settings: dict[str, Any], user_prompt: str) -> None:
    if not STEERING_VALUES:
        raise ValueError("STEERING_VALUES is required for activation_steering.")

    from activation_steering import MalleableModel, SteeringVector
    from activation_steering.leash_layer import LeashLayer

    model, tokenizer = load_model_and_tokenizer(settings)
    malleable_model = MalleableModel(model=model, tokenizer=tokenizer)
    malleable_model.reset_leash_to_default()

    vectors = []
    for value_name in STEERING_VALUES:
        vector_path = settings["vector_dir"] / f"{value_name}{settings['vector_name_suffix']}"
        if not Path(f"{vector_path}.svec").exists():
            raise FileNotFoundError(f"Missing steering vector: {vector_path}.svec")
        vectors.append(SteeringVector.load(str(vector_path)))

    layer_ids = resolve_layer_ids(BEHAVIOR_LAYER_IDS, settings["behavior_layer_ids"], len(vectors))
    strength = (
        BEHAVIOR_VECTOR_STRENGTH
        if BEHAVIOR_VECTOR_STRENGTH is not None
        else settings["behavior_vector_strength"]
    )
    malleable_model.multibehavior(
        behavior_vectors=vectors,
        behavior_layer_ids=layer_ids,
        behavior_vector_strengths=[float(strength)] * len(vectors),
        use_ooi_preventive_normalization=False,
        apply_behavior_on_first_call=False,
    )

    text, thinking = generate_with_transformers(
        model=malleable_model.model,
        tokenizer=tokenizer,
        settings=generation_settings(tokenizer),
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        chat_template_kwargs=settings["chat_template_kwargs"],
    )
    LeashLayer.reset_class()
    print_response("Activation Steering", text, thinking)


def adapter_path(settings: dict[str, Any], training_type: str, value_name: str) -> Path:
    return (
        settings["gold_adapter_dir"]
        / training_type
        / f"{settings['adapter_prefix']}_{value_name}_{training_type}_lora"
    )


def normalized_weights(n_adapters: int, weights: list[float] | None) -> list[float]:
    if weights is None:
        return [1.0 / n_adapters] * n_adapters
    if len(weights) != n_adapters:
        raise ValueError("LORA_ADAPTER_WEIGHTS must match STEERING_VALUES length.")
    total = sum(weights)
    if total <= 0:
        raise ValueError("LORA_ADAPTER_WEIGHTS must sum to a positive value.")
    return [weight / total for weight in weights]


def load_safetensors(path: Path) -> dict[str, torch.Tensor]:
    import torch
    from safetensors.torch import load_file

    return load_file(str(path))


def save_safetensors(state_dict: dict[str, Any], path: Path) -> None:
    from safetensors.torch import save_file

    save_file(state_dict, str(path))


def write_merged_adapter(adapter_paths: list[Path], weights: list[float], out_dir: Path) -> None:
    import torch

    configs = []
    state_dicts = []
    for path in adapter_paths:
        config_path = path / "adapter_config.json"
        weights_path = path / "adapter_model.safetensors"
        if not config_path.exists() or not weights_path.exists():
            raise FileNotFoundError(f"Missing LoRA adapter files under: {path}")
        configs.append(json.loads(config_path.read_text(encoding="utf-8")))
        state_dicts.append(load_safetensors(weights_path))

    if any(config != configs[0] for config in configs[1:]):
        raise ValueError("Selected LoRA adapter configs do not match.")
    if any(set(state_dict) != set(state_dicts[0]) for state_dict in state_dicts[1:]):
        raise ValueError("Selected LoRA adapter weight keys do not match.")

    merged = {}
    for key in sorted(state_dicts[0]):
        reference = state_dicts[0][key]
        weighted = sum(state_dict[key].to(torch.float32) * weight for state_dict, weight in zip(state_dicts, weights))
        merged[key] = weighted.to(reference.dtype)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "adapter_config.json").write_text(json.dumps(configs[0], indent=2), encoding="utf-8")
    save_safetensors(merged, out_dir / "adapter_model.safetensors")


def run_lora(settings: dict[str, Any], user_prompt: str) -> None:
    if not STEERING_VALUES:
        raise ValueError("STEERING_VALUES is required for lora.")

    from peft import PeftModel

    training_type = "dpo_gold"
    adapter_paths = [adapter_path(settings, training_type, value_name) for value_name in STEERING_VALUES]
    weights = normalized_weights(len(adapter_paths), LORA_ADAPTER_WEIGHTS)
    model, tokenizer = load_model_and_tokenizer(settings)

    with tempfile.TemporaryDirectory(prefix="merged_lora_adapter_") as temp_name:
        merged_path = Path(temp_name)
        write_merged_adapter(adapter_paths, weights, merged_path)
        model = PeftModel.from_pretrained(model, str(merged_path))
        model.eval()
        text, thinking = generate_with_transformers(
            model=model,
            tokenizer=tokenizer,
            settings=generation_settings(tokenizer),
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            chat_template_kwargs=settings["chat_template_kwargs"],
        )
    print_response("LoRA", text, thinking)


def print_response(label: str, text: str, thinking: str) -> None:
    print(f"\n=== {label} Response ===")
    print(text)
    if thinking:
        print(f"\n=== {label} Thinking ===")
        print(thinking)


def active_model_settings() -> dict[str, Any]:
    if MODEL_FAMILY not in MODEL_SETTINGS:
        raise ValueError(f"MODEL_FAMILY must be one of: {', '.join(sorted(MODEL_SETTINGS))}")
    settings = dict(MODEL_SETTINGS[MODEL_FAMILY])
    settings["chat_template_kwargs"] = dict(settings["chat_template_kwargs"])
    if MODEL_FAMILY == "qwen":
        settings["chat_template_kwargs"]["enable_thinking"] = bool(QWEN_THINKING)
    return settings


def main() -> None:
    scenario = load_scenario()
    user_prompt = scenario_prompt(scenario)
    settings = active_model_settings()

    print(f"Backend: {BACKEND}")
    print(f"Model family: {MODEL_FAMILY}")
    print(f"Scenario: {SCENARIO_KEY}")
    if STEERING_VALUES:
        print(f"Values: {', '.join(STEERING_VALUES)}")
    print_scenario_details(scenario)

    if BACKEND == "base":
        model, tokenizer = load_model_and_tokenizer(settings)
        text, thinking = generate_with_transformers(
            model=model,
            tokenizer=tokenizer,
            settings=generation_settings(tokenizer),
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            chat_template_kwargs=settings["chat_template_kwargs"],
        )
        print_response("Base", text, thinking)
    elif BACKEND == "activation_steering":
        run_activation_steering(settings, user_prompt)
    elif BACKEND == "lora":
        run_lora(settings, user_prompt)
    else:
        raise ValueError("BACKEND must be 'base', 'activation_steering', or 'lora'.")


if __name__ == "__main__":
    main()
