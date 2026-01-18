import os
import re
import torch
import warnings
from pathlib import Path
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

def trim_to_full_sentences(text):
    """Trim text to the last complete sentence ending with ., !, or ?."""
    matches = list(re.finditer(r"[.!?]", text))
    if not matches:
        return text
    return text[: matches[-1].end()]


# Load HF token from project-root .env (parent of Scripts/)
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
warnings.filterwarnings("ignore")  # ignore all Python warnings

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not found. Put it into your project-root .env file as: HF_TOKEN=hf_...")

cache_dir = os.path.expanduser("~/.cache/huggingface/transformers")
model_name = "meta-llama/Llama-3.2-1B"  # or "meta-llama/Meta-Llama-3-8B"

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=HF_TOKEN,
    cache_dir=cache_dir,
).to(device)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=HF_TOKEN,
    cache_dir=cache_dir
)

# Ensure we have a pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

prompt = "Once upon a time there was a little boy"

# Tokenize
encoded = tokenizer(prompt, return_tensors="pt")
input_ids = encoded.input_ids.to(device)
attention_mask = encoded.attention_mask.to(device)

# Generate: sampling + repetition controls; generate a bit extra then trim to full sentences
output = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=80,          # generate more than needed, then trim
    do_sample=True,
    temperature=0.9,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.15,
    no_repeat_ngram_size=3,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

# Decode
output_text = tokenizer.decode(output[0], skip_special_tokens=True)

# Keep only continuation, trim to complete sentences, and print
continuation = output_text[len(prompt):].lstrip()
continuation = trim_to_full_sentences(continuation)

print(prompt + " " + continuation)
