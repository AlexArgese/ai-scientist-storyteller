import os
import json
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
import evaluate
import numpy as np


# -----------------------------
# Prompt template (training)
# -----------------------------
TRAIN_PROMPT = (
    "You are an expert science storyteller.\n\n"
    "TASK:\n"
    "Read the research paper text and write a clear, engaging narrative tailored to the Persona.\n"
    "- Keep technical fidelity; do not invent facts.\n"
    "- Explain in your own words (no quotes, no lists, no URLs).\n"
    "Persona: {persona}\n\n"
    "Paper:\n"
    "<<<BEGIN PAPER>>>\n"
    "{paper}\n"
    "<<<END PAPER>>>\n\n"
    "Write the story now:\n"
)

# Nota: durante l'inference potrai usare il prompt a 5 sezioni.
# Il modello, avendo visto questo stile didattico in training, si adatterà bene.


# -----------------------------
# Dataset JSONL
# -----------------------------
class StoryDataset(Dataset):
    """
    Accetta .jsonl con almeno: paper, story, persona.
    Se i nomi sono diversi (e.g. input/output), viene fatto un mapping automatico.
    """
    def __init__(self, path: str, tokenizer: AutoTokenizer, max_len: int = 4096):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Possibili chiavi alternative
        input_keys = ["paper", "input", "source", "document", "paper_text"]
        output_keys = ["story", "output", "target", "story_text", "response"]
        persona_keys = ["persona", "audience", "role"]

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)

                # paper
                paper = None
                for k in input_keys:
                    if k in obj and obj[k]:
                        paper = obj[k]
                        break

                # story/target
                story = None
                for k in output_keys:
                    if k in obj and obj[k]:
                        story = obj[k]
                        break

                # persona
                persona = None
                for k in persona_keys:
                    if k in obj and obj[k]:
                        persona = obj[k]
                        break
                if persona is None:
                    persona = "Student"

                if not paper or not story:
                    # salta righe non valide
                    continue

                prompt = TRAIN_PROMPT.format(persona=persona, paper=paper)
                # Per causal LM: input = prompt + story, i label sono la parte della story
                self.samples.append({"prompt": prompt, "story": story})

        if not self.samples:
            raise ValueError(f"Nessun esempio valido trovato in {path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        prompt_ids = self.tokenizer(
            item["prompt"], add_special_tokens=False, truncation=True, max_length=self.max_len
        )["input_ids"]
        story_ids = self.tokenizer(
            item["story"], add_special_tokens=False, truncation=True, max_length=self.max_len
        )["input_ids"]

        # concatena per causal LM
        input_ids = prompt_ids + story_ids
        if len(input_ids) > self.max_len:
            input_ids = input_ids[: self.max_len]

        # labels: ignora la parte di prompt (-100), apprende solo la storia
        labels = [-100] * min(len(prompt_ids), len(input_ids))
        labels += input_ids[len(labels):]

        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


# -----------------------------
# Preprocess dei logits -> token ids per le metriche
# -----------------------------
def preprocess_logits_for_metrics(logits, labels):
    # logits: (batch, seq_len, vocab) -> id con argmax
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


# -----------------------------
# Metrics (ROUGE, BLEU, BERTScore opz.)
# -----------------------------
def build_metrics(tokenizer: AutoTokenizer, use_bert_score: bool = False):
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("sacrebleu")
    bert_scorer = evaluate.load("bertscore") if use_bert_score else None

    def decode_batch(preds, labels):
        # preds e labels arrivano come np.ndarray/list; normalizziamo
        preds = np.asarray(preds)
        labels = np.asarray(labels)

        # rimpiazza -100 con pad_token_id per decode pulito
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        # assicurati che siano 2D: (batch, seq_len)
        if preds.ndim == 3:
            # nel caso fossero ancora logits
            preds = preds.argmax(axis=-1)
        elif preds.ndim == 1:
            preds = preds[None, :]

        if labels.ndim == 1:
            labels = labels[None, :]

        decoded_preds, decoded_labels = [], []
        for p, l in zip(preds, labels):
            p = np.where(p == -100, pad_id, p).astype(int).tolist()
            l = np.where(l == -100, pad_id, l).astype(int).tolist()
            decoded_preds.append(tokenizer.decode(p, skip_special_tokens=True))
            decoded_labels.append(tokenizer.decode(l, skip_special_tokens=True))
        return decoded_preds, decoded_labels

    def extract_generated_only(texts: List[str]) -> List[str]:
        # rimuove il prompt: tutto fino a "Write the story now:"
        out = []
        marker = "Write the story now:"
        for t in texts:
            pos = t.find(marker)
            if pos >= 0:
                out.append(t[pos + len(marker):].strip())
            else:
                out.append(t.strip())
        return out

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        # Con preprocess_logits_for_metrics, preds sono già token ids (batch, seq_len)
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds, decoded_labels = decode_batch(preds, labels)

        # Prendi solo la parte generata
        gen_preds = extract_generated_only(decoded_preds)
        gen_refs = extract_generated_only(decoded_labels)

        # ROUGE
        rouge_res = rouge.compute(predictions=gen_preds, references=gen_refs)
        rougeL = rouge_res.get("rougeL", 0.0)

        # BLEU (sacrebleu vuole lista di list)
        bleu_res = bleu.compute(predictions=gen_preds, references=[[r] for r in gen_refs])
        bleu_score = bleu_res.get("score", 0.0)

        metrics = {
            "rougeL": rougeL,
            "bleu": bleu_score,
        }

        # BERTScore (opzionale)
        if bert_scorer is not None:
            bs = bert_scorer.compute(
                predictions=gen_preds, references=gen_refs, lang="en"
            )
            metrics["bertscore_f1"] = float(np.mean(bs["f1"]))

        # metrica composita per early stopping / best model
        # pesi: 0.7 rougeL + 0.3 bleu (normalizzati su [0,1])
        comp = 0.7 * (rougeL / 100.0) + 0.3 * (bleu_score / 100.0)
        metrics["composite"] = comp

        return metrics

    return compute_metrics


# -----------------------------
# Main train
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./mistral-qlora-ft-merged")
    parser.add_argument("--train_file", default="train.jsonl")
    parser.add_argument("--val_file", default="val.jsonl")
    parser.add_argument("--output_dir", default="./sft_out")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=4096)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--use_bert_score", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(">> Carico modello…")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if (args.bf16 and torch.cuda.is_available()) else None,
        device_map="auto",
    )

    # --------- Dropout (best-effort: alcuni modelli non lo espongono) ----------
    for k in ["attention_dropout", "attn_dropout", "hidden_dropout", "hidden_dropout_prob"]:
        if hasattr(model.config, k):
            setattr(model.config, k, 0.1)

    # Per stabilità del training lungo
    model.config.use_cache = False  # richiesto per gradient checkpointing in alcuni modelli

    # Dataset
    train_ds = StoryDataset(args.train_file, tokenizer, max_len=args.max_len)
    val_ds = StoryDataset(args.val_file, tokenizer, max_len=args.max_len)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    compute_metrics = build_metrics(tokenizer, use_bert_score=args.use_bert_score)

    # Training args
    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        logging_steps=50,
        lr_scheduler_type="cosine",
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="composite",   # usa la metrica composita
        greater_is_better=True,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        seed=args.seed,
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        data_collator=data_collator,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,  # <--- AGGIUNTO
    )

    print(">> Inizio training…")
    trainer.train()
    print(">> Salvo best model…")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(">> Valutazione finale…")
    metrics = trainer.evaluate()
    print(metrics)


if __name__ == "__main__":
    main()
