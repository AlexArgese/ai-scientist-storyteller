# train_flan_t5_lora.py
import os, argparse, numpy as np, torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    BitsAndBytesConfig, DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from evaluate import load as load_metric
from transformers.trainer_utils import get_last_checkpoint, set_seed

# ---------- Config di base ----------
DATA_DIR   = "/docker/argese/volumes/final_data"
TRAIN_JSON = os.path.join(DATA_DIR, "train_ready_trunc.json")
TEST_JSON  = os.path.join(DATA_DIR, "test_ready_trunc.json")

# Modello con contesto 1024 reale
MODEL_NAME = "google/long-t5-local-base"
OUT_DIR    = "./longt5_local_base_lora_4bit_multi"

MAX_INPUT_LEN  = 1024
MAX_TARGET_LEN = 1024
SEED = 42
# ------------------------------------

def is_main_process():
    return (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() == 0)

def build_datasets(tokenizer):
    ds = load_dataset("json", data_files={"train": TRAIN_JSON, "test": TEST_JSON})
    split = ds["train"].train_test_split(test_size=0.125, seed=SEED)
    train_data, val_data = split["train"], split["test"]
    test_data = ds["test"]

    def tok(batch):
        model_inputs = tokenizer(
            batch["input"],
            max_length=MAX_INPUT_LEN,
            truncation=True,
            padding=False,
        )
        labels = tokenizer(
            text_target=batch["output"],
            max_length=MAX_TARGET_LEN,
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_tok = train_data.map(tok, batched=True, remove_columns=train_data.column_names)
    val_tok   = val_data.map(tok,   batched=True, remove_columns=val_data.column_names)
    test_tok  = test_data.map(tok,  batched=True, remove_columns=test_data.column_names)
    return train_tok, val_tok, test_tok

def build_model(local_rank: int):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    base = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": local_rank},
        torch_dtype="auto",
    )

    peft_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q","k","v","o","wi","wo"],
        modules_to_save=["lm_head"],
    )

    model = get_peft_model(base, peft_cfg)
    model.print_trainable_parameters()

    if hasattr(model, "config"):
        model.config.use_cache = False

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    return model

# --- callback per mostrare progresso eval/generazione ---
import time
from transformers import TrainerCallback

class EvalProgressCallback(TrainerCallback):
    def __init__(self, every=5):
        self.every = every
        self._t0 = None
        self._seen = 0
        self._total = None

    def on_evaluate(self, args, state, control, **kwargs):
        self._t0 = time.time()
        self._seen = 0
        dl = kwargs.get("eval_dataloader")
        self._total = len(dl) if dl is not None else None
        if self._total:
            print(f"[eval] started: {self._total} batches (predict_with_generate={args.predict_with_generate})")

    def on_prediction_step(self, args, state, control, **kwargs):
        self._seen += 1
        if self._total and (self._seen % self.every == 0 or self._seen == self._total):
            dt = max(time.time() - self._t0, 1e-9)
            rate = self._seen / dt
            eta = (self._total - self._seen) / max(rate, 1e-9)
            print(f"[eval] {self._seen}/{self._total} batches | {rate:.2f} it/s | ETA {eta:.1f}s")

    def on_evaluate_end(self, args, state, control, **kwargs):
        if self._total:
            print(f"[eval] done in {time.time() - self._t0:.1f}s")

# --- callback per attivare/disattivare cache ---
class ToggleCacheOnEval(TrainerCallback):
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is not None and hasattr(model, "config"):
            model.config.use_cache = True
    def on_evaluate_end(self, args, state, control, model=None, **kwargs):
        if model is not None and hasattr(model, "config"):
            model.config.use_cache = False

def build_trainer(model, tokenizer, train_ds, val_ds, cli_args):
    rouge = load_metric("rouge")
    sacrebleu = load_metric("sacrebleu")

    def safe_decode(batch, max_len):
        batch = [np.asarray(x)[:max_len] for x in batch]
        return tokenizer.batch_decode(batch, skip_special_tokens=True)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        pred_txt = safe_decode(preds,  MAX_TARGET_LEN)
        lab_txt  = safe_decode(labels, MAX_TARGET_LEN)

        r = rouge.compute(predictions=pred_txt, references=lab_txt)
        sb = sacrebleu.compute(predictions=pred_txt, references=[[t] for t in lab_txt])

        def _get(x):
            try:
                return float(x)
            except Exception:
                return float(getattr(x, "mid", x).fmeasure)
        return {
            "rouge1": round(_get(r["rouge1"]) * 100, 2),
            "rouge2": round(_get(r["rouge2"]) * 100, 2),
            "rougeL": round(_get(r["rougeL"]) * 100, 2),
            "sacrebleu": round(sb["score"], 2),
        }

    logging_steps = 100

    args = Seq2SeqTrainingArguments(
        output_dir=OUT_DIR,
        do_train=True,
        do_eval=True,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        logging_first_step=True,
        load_best_model_at_end=True,
        logging_steps=logging_steps,
        logging_strategy="steps",
        disable_tqdm=False,
        num_train_epochs=cli_args.epochs,
        per_device_train_batch_size=cli_args.train_bs,
        per_device_eval_batch_size=cli_args.eval_bs,
        gradient_accumulation_steps=cli_args.grad_accum,
        learning_rate=cli_args.lr,
        warmup_steps=500,
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LEN,
        save_total_limit=2,
        report_to="none",
        fp16=False,
        bf16=True,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=True,
        gradient_checkpointing=True,

        # --- FIX OOM ---
        eval_accumulation_steps=2,
        generation_num_beams=1,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        model=model,
    )

    return Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if is_main_process() else None,
        callbacks=[EvalProgressCallback(every=5), ToggleCacheOnEval()],
    )

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--train_bs", type=int, default=1)
    p.add_argument("--eval_bs", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    return p.parse_args()

def main():
    cli_args = parse_args()
    set_seed(SEED)

    torch.backends.cuda.matmul.allow_tf32 = True
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if is_main_process():
        os.makedirs(OUT_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    train_ds, val_ds, _ = build_datasets(tokenizer)
    model = build_model(local_rank)

    trainer = build_trainer(model, tokenizer, train_ds, val_ds, cli_args)

    #ckpt = get_last_checkpoint(OUT_DIR) if os.path.isdir(OUT_DIR) else None
    #if is_main_process() and ckpt:
    #    print(f"Riprendo da checkpoint: {ckpt}")
    ckpt = None

    trainer.train(resume_from_checkpoint=ckpt)

    if is_main_process():
        trainer.save_model(OUT_DIR)
        tokenizer.save_pretrained(OUT_DIR)
        print("✔ Training completato e modello salvato in", OUT_DIR)

if __name__ == "__main__":
    main()
