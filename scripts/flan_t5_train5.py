#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, re, json, csv, argparse, random
from typing import List, Tuple, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments,
    EarlyStoppingCallback, BitsAndBytesConfig
)
from transformers.trainer_utils import get_last_checkpoint

from rouge_score import rouge_scorer
import sacrebleu

# === PEFT (LoRA / QLoRA) ===
HAS_PEFT = True
try:
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
except Exception:
    HAS_PEFT = False


# ============== IO utils ==============
def read_json_array(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} non è una lista JSON.")
    return data

def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def read_csv(path: str) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out.append(row)
    return out

def load_any(path: str) -> List[dict]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        return read_jsonl(path)
    if ext == ".json":
        return read_json_array(path)
    if ext == ".csv":
        return read_csv(path)
    raise ValueError(f"Formato non supportato: {ext}")

def map_record(rec: dict, source_field: str, target_field: str) -> Tuple[str,str]:
    if source_field in rec and target_field in rec:
        return str(rec[source_field]), str(rec[target_field])
    for s in ["input","source","prompt","instruction","question","text","context"]:
        for t in ["output","target","answer","story","completion","response"]:
            if s in rec and t in rec:
                return str(rec[s]), str(rec[t])
    raise KeyError(f"Manca source/target nel record: {list(rec.keys())}")


# ============== Prompt builder (allineato ai tuoi dati) ==============
PERSONAS_RE = re.compile(r"Generate a story for these personas:\s*(.+)", re.I)
SECTION_RE  = re.compile(r"Section:\s*([^\n\r]+)", re.I)

def parse_personas_and_section(raw_input: str) -> Tuple[str, str]:
    personas = "General"
    section = "Section"
    m = PERSONAS_RE.search(raw_input)
    if m: personas = m.group(1).strip()
    m = SECTION_RE.search(raw_input)
    if m: section = m.group(1).strip()
    return personas, section

def build_storyteller_prompt(raw_input: str, style: str="storyteller") -> str:
    personas, section = parse_personas_and_section(raw_input)
    content = raw_input
    m = re.search(r"^(Abstract|Introduction|Methods?|Results?|Conclusion)\s*:\s*", raw_input, re.I | re.M)
    if m:
        start = m.start()
        header = raw_input[start:].splitlines()[0]
        body   = raw_input[start+len(header):].strip()
        content = f"{header}\n{body}".strip()

    if style == "storyteller":
        return (
            "You are an **AI Scientist Storyteller**. Your role is to turn scientific content into engaging, self-contained stories.\n"
            f"Personas: {personas}\n"
            f"Section: {section}\n\n"
            "Rules:\n"
            "- Write ONLY the narrative (no lists, JSON, tables, citations, URLs, or DOIs).\n"
            "- 6–9 sentences (~300 words). Third person voice.\n"
            "- Be faithful to the source; keep the text self-contained and coherent.\n"
            "- Adapt tone and vocabulary to the personas.\n\n"
            f"Content:\n{content}\n\n"
            "Story:"
        )
    elif style == "raw":
        return raw_input
    else:
        return (
            f"Rewrite the following scientific content as a story for: {personas}. "
            f"Section: {section}. 6–9 sentences, third person, no citations/URLs.\n\n"
            f"{content}\n\nStory:"
        )


# ============== Dataset ==============
class StoryDataset(Dataset):
    def __init__(self, rows: List[dict], tok, max_src=768, max_tgt=384,
                 source_field="input", target_field="output",
                 prompt_style="storyteller"):
        self.rows = rows; self.tok = tok
        self.max_src = max_src; self.max_tgt = max_tgt
        self.source_field = source_field; self.target_field = target_field
        self.prompt_style = prompt_style

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        rec = self.rows[idx]
        src_raw, tgt = map_record(rec, self.source_field, self.target_field)
        src_prompt = build_storyteller_prompt(src_raw, style=self.prompt_style)
        model_inputs = self.tok(src_prompt, max_length=self.max_src, truncation=True)
        labels = self.tok(text_target=tgt, max_length=self.max_tgt, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


# ============== Safe decode helpers (fix OverflowError) ==============
def _safe_decode_batch(tokenizer, ids) -> List[str]:
    arr = np.asarray(ids)
    # Se per errore arrivano logits (ndim=3), riduciamo con argmax
    if arr.ndim == 3:
        arr = arr.argmax(axis=-1)
    arr = arr.astype(np.int64, copy=False)

    eos = tokenizer.eos_token_id or tokenizer.pad_token_id or 0
    vocab_size = getattr(tokenizer, "vocab_size", None)

    # Clippa/maschera qualsiasi ID invalido
    arr = np.where(arr < 0, eos, arr)
    if vocab_size is not None:
        arr = np.where(arr >= vocab_size, eos, arr)

    try:
        return tokenizer.batch_decode(arr.tolist(), skip_special_tokens=True)
    except Exception:
        # Fallback ultra-robusto per singolo esempio
        out = []
        for row in arr.tolist():
            valid = []
            for x in row:
                if x is None: continue
                if x < 0: 
                    valid.append(eos); continue
                if vocab_size is not None and x >= vocab_size:
                    valid.append(eos); continue
                valid.append(int(x))
            try:
                out.append(tokenizer.decode(valid, skip_special_tokens=True))
            except Exception:
                out.append("")
        return out

def _safe_decode_labels(tokenizer, labels) -> List[str]:
    texts=[]
    eos = tokenizer.eos_token_id or tokenizer.pad_token_id or 0
    vocab_size = getattr(tokenizer, "vocab_size", None)
    for l in labels:
        l = [x for x in l if x != -100]
        fixed = []
        for x in l:
            if x is None or x < 0:
                fixed.append(eos)
            elif vocab_size is not None and x >= vocab_size:
                fixed.append(eos)
            else:
                fixed.append(int(x))
        texts.append(tokenizer.decode(fixed, skip_special_tokens=True))
    return texts


# ============== Metrics (ROUGE/BLEU + score composito per early stopping) ==============
def _pp_for_rouge(texts: List[str]) -> List[str]:
    out=[]
    for t in texts:
        t = re.sub(r"\s+"," ", (t or "")).strip()
        t = re.sub(r"([.!?])\s+", r"\1\n", t)  # una frase per riga
        out.append(t)
    return out

def compute_metrics_builder(tokenizer):
    rouge = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        pred_txt = _safe_decode_batch(tokenizer, preds)
        gold_txt = _safe_decode_labels(tokenizer, labels)

        P=_pp_for_rouge(pred_txt); R=_pp_for_rouge(gold_txt)
        r1=r2=rL=0.0
        for p,r in zip(P,R):
            s = rouge.score(r,p)
            r1+=s["rouge1"].fmeasure; r2+=s["rouge2"].fmeasure; rL+=s["rougeL"].fmeasure
        n=max(1,len(P))
        r1/=n; r2/=n; rL/=n

        bleu = sacrebleu.corpus_bleu(pred_txt, [gold_txt]).score  # 0..100
        earlystop = 0.7 * rL + 0.3 * (bleu / 100.0)

        return {
            "rouge1": r1, "rouge2": r2, "rougeL": rL,
            "bleu": bleu,
            "earlystop_score": earlystop
        }
    return compute_metrics


# ============== LoRA helper ==============
def maybe_wrap_lora(model, use_lora=True, r=16, alpha=32, dropout=0.05):
    if not use_lora: return model
    if not HAS_PEFT:
        print("[WARN] PEFT non installato: procedo senza LoRA.")
        return model
    target_modules = ["q","k","v","o","wi_0","wi_1","wo"]  # T5 blocks
    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
        target_modules=target_modules, task_type=TaskType.SEQ_2_SEQ_LM
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model


# ============== Split helper ==============
def split_train_val(
    rows: List[dict],
    val_split: float = 0.1,
    seed: int = 42,
    group_key: str = "id_story"
) -> Tuple[List[dict], List[dict]]:
    val_split = max(0.01, min(0.5, float(val_split)))
    rng = random.Random(seed)

    if group_key and group_key != "none" and any(group_key in r for r in rows):
        buckets: Dict[str, List[dict]] = {}
        for r in rows:
            k = str(r.get(group_key, "__nogroup__"))
            buckets.setdefault(k, []).append(r)
        keys = list(buckets.keys()); rng.shuffle(keys)
        n_val_groups = max(1, int(round(len(keys) * val_split)))
        val_keys = set(keys[:n_val_groups])
        val_rows = [r for k in val_keys for r in buckets[k]]
        train_rows = [r for k in keys[n_val_groups:] for r in buckets[k]]
    else:
        idx = list(range(len(rows))); rng.shuffle(idx)
        n_val = max(1, int(round(len(rows) * val_split)))
        val_idx = set(idx[:n_val])
        val_rows = [rows[i] for i in range(len(rows)) if i in val_idx]
        train_rows = [rows[i] for i in range(len(rows)) if i not in val_idx]

    if len(val_rows) == 0 or len(train_rows) == 0:
        raise ValueError("Split non valido: uno dei due insiemi è vuoto. Riduci val_split.")
    return train_rows, val_rows


# ============== Train ==============
def set_seed(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def main():
    ap = argparse.ArgumentParser(description="Fine-tune FLAN-T5 (QLoRA opzionale) con prompt storyteller + early stopping. Se --val_file manca, crea split dal train.")
    ap.add_argument("--model", type=str, default="google/flan-t5-large")
    ap.add_argument("--train_file", type=str, required=True)
    ap.add_argument("--val_file", type=str, default=None)
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--split_group_key", type=str, default="id_story", choices=["id_story","id_paper","none"])
    ap.add_argument("--save_splits_dir", type=str, default=None)

    ap.add_argument("--source_field", type=str, default="input")
    ap.add_argument("--target_field", type=str, default="output")
    ap.add_argument("--output_dir", type=str, default="./flant5_sft")
    ap.add_argument("--max_source_len", type=int, default=768)
    ap.add_argument("--max_target_len", type=int, default=384)
    ap.add_argument("--prompt_style", type=str, default="storyteller", choices=["storyteller","raw","minimal"])

    # QLoRA / LoRA / precisione
    ap.add_argument("--load_in_4bit", action="store_true", help="Attiva QLoRA (usa BitsAndBytesConfig)")
    ap.add_argument("--use_lora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--gradient_checkpointing", action="store_true")

    # training
    ap.add_argument("--train_bs", type=int, default=2)
    ap.add_argument("--eval_bs", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--epochs", type=float, default=30.0)
    ap.add_argument("--scheduler", type=str, default="cosine")
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--eval_strategy", type=str, default="steps", choices=["steps","epoch"])
    ap.add_argument("--eval_steps", type=int, default=500)
    ap.add_argument("--save_strategy", type=str, default="steps", choices=["steps","epoch"])
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--save_total_limit", type=int, default=3)
    ap.add_argument("--gen_max_len", type=int, default=360)
    ap.add_argument("--gen_beams", type=int, default=4)
    ap.add_argument("--logging_steps", type=int, default=50)
    ap.add_argument("--es_patience", type=int, default=3)
    ap.add_argument("--es_threshold", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)

    # resume
    ap.add_argument("--resume", action="store_true", help="Riprende da last checkpoint in --output_dir se presente")
    args = ap.parse_args()

    set_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    if args.load_in_4bit:
        if not HAS_PEFT:
            raise RuntimeError("PEFT non disponibile: per QLoRA è richiesto peft>=0.10.0")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model,
            device_map="auto",
            quantization_config=quantization_config
        )
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None
        )
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False

    model = maybe_wrap_lora(
        model, use_lora=args.use_lora,
        r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout
    )

    # ---- dataset & split
    all_rows = load_any(args.train_file)
    if args.val_file and os.path.exists(args.val_file):
        train_rows = all_rows
        val_rows   = load_any(args.val_file)
        print(f"[INFO] Usata validation da file esterno: {args.val_file} (N={len(val_rows)})")
    else:
        train_rows, val_rows = split_train_val(
            all_rows, val_split=args.val_split, seed=args.seed, group_key=args.split_group_key
        )
        print(f"[INFO] Validation creata da train: val_split={args.val_split} | group_key={args.split_group_key}")
        print(f"[INFO] -> train N={len(train_rows)} | val N={len(val_rows)}")
        if args.save_splits_dir:
            os.makedirs(args.save_splits_dir, exist_ok=True)
            with open(os.path.join(args.save_splits_dir, "train_split.json"), "w", encoding="utf-8") as f:
                json.dump(train_rows, f, ensure_ascii=False, indent=2)
            with open(os.path.join(args.save_splits_dir, "val_split.json"), "w", encoding="utf-8") as f:
                json.dump(val_rows, f, ensure_ascii=False, indent=2)
            print(f"[INFO] Split salvato in {args.save_splits_dir}")

    train_ds = StoryDataset(train_rows, tok, args.max_source_len, args.max_target_len,
                            args.source_field, args.target_field, prompt_style=args.prompt_style)
    val_ds   = StoryDataset(val_rows, tok, args.max_source_len, args.max_target_len,
                            args.source_field, args.target_field, prompt_style=args.prompt_style)

    data_collator = DataCollatorForSeq2Seq(tok, model=model)
    metrics_fn = compute_metrics_builder(tok)

    targs = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
        lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup_ratio,

        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps if args.eval_strategy=="steps" else None,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps if args.save_strategy=="steps" else None,
        save_total_limit=args.save_total_limit,

        predict_with_generate=True,
        generation_max_length=args.gen_max_len,
        generation_num_beams=args.gen_beams,

        load_best_model_at_end=True,
        metric_for_best_model="earlystop_score",
        greater_is_better=True,

        fp16=False,
        bf16=torch.cuda.is_available(),
        report_to=["none"],
        logging_steps=args.logging_steps,
        seed=args.seed
    )

    callbacks = [EarlyStoppingCallback(
        early_stopping_patience=args.es_patience,
        early_stopping_threshold=args.es_threshold
    )]

    trainer = Seq2SeqTrainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=val_ds,
        tokenizer=tok, data_collator=data_collator,
        compute_metrics=metrics_fn,
        callbacks=callbacks
    )

    # resume da last checkpoint se richiesto o se esiste già
    resume_ckpt = None
    if args.resume:
        last_ckpt = get_last_checkpoint(args.output_dir)
        if last_ckpt:
            resume_ckpt = last_ckpt
            print(f"[INFO] Riprendo da checkpoint: {resume_ckpt}")

    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)

    metrics = trainer.evaluate(val_ds, max_length=args.gen_max_len, num_beams=args.gen_beams)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "eval_val_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("[DONE] Best model salvato in:", args.output_dir)

if __name__ == "__main__":
    main()
