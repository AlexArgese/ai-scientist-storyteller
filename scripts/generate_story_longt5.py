import os, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, PeftConfig

MODEL_DIR = "./longt5_local_base_lora_4bit"  # cartella con adapter_model.safetensors

# (opzionale) cache locale scrivibile
os.environ.setdefault("TRANSFORMERS_CACHE", "/docker/argese/volumes/training4/hf_cache")
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)

# Carica tokenizer del base model (più sicuro)
BASE_NAME = "google/long-t5-local-base"
tokenizer = AutoTokenizer.from_pretrained(BASE_NAME, use_fast=True)

# Carica base + LoRA
base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_NAME, torch_dtype="auto")
model = PeftModel.from_pretrained(base_model, MODEL_DIR)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Leggi paper
with open("paper.txt", "r") as f:
    paper_text = f.read()

# Prompt T5-like, breve e mirato
instruction = (
    "You are a helpful tutor. Write the INTRODUCTION of a short story, in clear English, "
    "addressed to a Student, based on the following paper content. Keep it 3–5 paragraphs."
)

prompt = f"{instruction}\n\nPaper:\n{paper_text}"

# Tokenizza con un cap sugli input (LongT5 locale ≈ 1024 token input)
enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
enc = {k: v.to(device) for k, v in enc.items()}

# Generazione deterministica per test (no sampling)
with torch.no_grad():
    out = model.generate(
        **enc,
        max_new_tokens=300,
        num_beams=4,
        do_sample=False,
        length_penalty=1.0,
        no_repeat_ngram_size=3,
        early_stopping=True,
    )

text = tokenizer.decode(out[0], skip_special_tokens=True)
print("\n=== GENERATED STORY (DETERMINISTIC TEST) ===\n")
print(text)
