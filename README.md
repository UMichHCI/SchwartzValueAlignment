# Schwartz Value Alignment

Repo for generating advice responses with:

- base Llama/Qwen models
- activation steering over Schwartz value vectors
- DPO_GOLD LoRA adapters

The script uses scenarios from `datasets/scenarios_eval.json`. Edit settings at the top of each script, then run the script.

The `activation_steering/` code is forked from
[IBM/activation-steering](https://github.com/IBM/activation-steering), which is
licensed under Apache 2.0. The upstream license is preserved at
`activation_steering/LICENSE`.

## Setup

Create an environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with a Hugging Face token that has access to the gated models:

```bash
cp .env.example .env
```

Then edit `.env`:

```text
HF_TOKEN=your_huggingface_token_here
```

The runtime and training scripts load this file automatically.

## Files

- `generate_response.py`: generate one response from a selected scenario.
- `activation_steering/Run/get_behavior_vector_v2.py`: train activation steering vectors.
- `lora/run/lora_trainer.py`: train DPO_GOLD LoRA adapters.
- `datasets/final/activation_training_dataset.json`: activation steering training data.
- `datasets/final/lora_dpo_gold.json`: DPO_GOLD LoRA training data.
- `datasets/scenarios_eval.json`: generation scenarios.

LoRA adapter outputs live under `lora/lora_parameters_gold_gpt4o/` and are gitignored.

## Generate

Open `generate_response.py` and edit the top-of-file settings:

```python
BACKEND = "base"  # "base", "activation_steering", or "lora"
MODEL_FAMILY = "qwen"  # "qwen" or "llama"
SCENARIO_KEY = "scenario_103"
```

For activation steering or LoRA, uncomment one or more values in `STEERING_VALUES`.

Run:

```bash
python generate_response.py
```

For LoRA generation, this repo only supports DPO_GOLD adapters. Train or add adapters under:

```text
lora/lora_parameters_gold_gpt4o/llama/dpo_gold/
lora/lora_parameters_gold_gpt4o/qwen/dpo_gold/
```

## Train Activation Steering Vectors

Train all value vectors for Qwen:

```bash
python activation_steering/Run/get_behavior_vector_v2.py --model Qwen/Qwen3-8B
```

Train all value vectors for Llama:

```bash
python activation_steering/Run/get_behavior_vector_v2.py --model meta-llama/Meta-Llama-3.1-8B-Instruct
```

Train only selected values:

```bash
python activation_steering/Run/get_behavior_vector_v2.py \
  benevolence_caring benevolence_dependability \
  --model Qwen/Qwen3-8B
```

Outputs are saved to:

```text
activation_steering/steering_vectors/qwen/
activation_steering/steering_vectors/llama/
```

## Train DPO_GOLD LoRA Adapters

Open `lora/run/lora_trainer.py` and set:

```python
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
```

or:

```python
MODEL_NAME = "Qwen/Qwen3-8B"
```

Run:

```bash
python lora/run/lora_trainer.py
```

The trainer creates one DPO_GOLD adapter per value from `datasets/final/lora_dpo_gold.json`.

Outputs are saved to:

```text
lora/lora_parameters_gold_gpt4o/llama/dpo_gold/
lora/lora_parameters_gold_gpt4o/qwen/dpo_gold/
```
