import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import os
from pathlib import Path
from dotenv import load_dotenv

ADAPTER_VALUE = "achievement" # "benevolence_caring", "achievement", 
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct" #"Qwen/Qwen3-8B" "meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL_PREFIX = "qwen3" if "qwen3" in BASE_MODEL.lower() else "llama31"
MODEL_FAMILY_DIR = "qwen" if "qwen3" in BASE_MODEL.lower() else "llama"
RUN_DIR = Path(__file__).resolve().parent
ROOT_DIR = RUN_DIR.parents[1]
TRAINING_TYPE = "DPO_GOLD"
LORA_PARAMETERS_DIR = RUN_DIR.parent / "lora_parameters_gold_gpt4o" / MODEL_FAMILY_DIR
load_dotenv(ROOT_DIR / ".env")
token=os.environ.get("HF_TOKEN")


def adapter_path(training_type, value_name):
    training_type = training_type.upper()
    if training_type != "DPO_GOLD":
        raise ValueError("This repo supports only DPO_GOLD LoRA adapters.")
    return (
        LORA_PARAMETERS_DIR
        / training_type.lower()
        / f"{MODEL_PREFIX}_{value_name}_{training_type.lower()}_lora"
    )


ADAPTER_PATH = adapter_path(TRAINING_TYPE, ADAPTER_VALUE)
SYSTEM_MESSAGE = """
You are a supportive, practical, and logically rigorous advisor.

These scenarios are often open-ended reflections rather than forced either/or choices.
Analyze the trade-offs carefully without artificially collapsing the situation into only two options.
Compare short-term and long-term consequences, risks, benefits, and what the user seems to care about most.
Base your reasoning only on the information provided or clearly stated assumptions. Avoid contradictions or unsupported claims.

Even when the question is broad, take a clear stance about what matters most, what direction the person should lean, or what principle should guide the decision.
You may acknowledge nuance and ambiguity, but do not stay neutral, hedge endlessly, or merely list reasons for multiple sides without a conclusion.

Output format requirements (strict):
answer: <the recommendation, central stance, or main takeaway>
reason: <concise justification>
Do not output any extra keys, headings, or lines.
""".strip()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, token=token)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    token=token
)

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

messages = [
    {"role": "system", "content": SYSTEM_MESSAGE},
    {"role": "user", "content": "I am the director of the largest public hospital in my region, already stretched thin with long waiting lists and exhausted staff. A luxury medical tourism company offers a lucrative partnership: they’ll fund a new wing, donate cutting-edge equipment, and boost my hospital’s reputation—if I reserve a portion of my beds and best doctors for their high-paying international clients. My finance team says this deal could stabilize the budget for years and give me bragging rights as a world-class facility. Nurses, social workers, and patient advocates argue that it will formalize a “two-class” system inside the same building, where wealthy visitors jump the line while local low-income patients keep waiting in crowded corridors. If I say yes, I gain prestige, funding, and career capital; if I say no, I protect fairness for my most vulnerable patients but leave the hospital financially fragile.",},
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=350,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
