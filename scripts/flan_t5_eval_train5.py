#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, re, json, csv, argparse, random, math
from typing import List, Dict, Tuple, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer
import sacrebleu

# PEFT (LoRA) opzionale
HAS_PEFT = True
try:
    from peft import AutoPeftModelForSeq2SeqLM, PeftConfig
except Exception:
    HAS_PEFT = False

# ---------- IO ----------
def read_json_array(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} non è una lista JSON.")
    return data

def read_jsonl(path: str) -> List[dict]:
    out=[]
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line: out.append(json.loads(line))
    return out

def read_csv(path: str) -> List[dict]:
    out=[]
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f): out.append(row)
    return out

def load_any(path: str) -> List[dict]:
    ext = os.path.splitext(path)[1].lower()
    if ext==".json": return read_json_array(path)
    if ext==".jsonl": return read_jsonl(path)
    if ext==".csv": return read_csv(path)
    raise ValueError(f"Formato non supportato: {ext}")

def map_record(rec: dict, source_field: str, target_field: str) -> Tuple[str,str]:
    if source_field in rec and target_field in rec:
        return str(rec[source_field]), str(rec[target_field])
    for s in ["input","source","prompt","instruction","question","text","context"]:
        for t in ["output","target","answer","story","completion","response"]:
            if s in rec and t in rec: return str(rec[s]), str(rec[t])
    raise KeyError(f"Manca source/target nel record: {list(rec.keys())}")

# ---------- Prompt ----------
PERSONAS_RE = re.compile(r"Generate a story for these personas:\s*(.+)", re.I)
SECTION_RE  = re.compile(r"Section:\s*([^\n\r]+)", re.I)

def parse_personas_and_section(raw_input: str) -> Tuple[str,str]:
    personas="General"; section="Section"
    m=PERSONAS_RE.search(raw_input); 
    if m: personas=m.group(1).strip()
    m=SECTION_RE.search(raw_input);
    if m: section=m.group(1).strip()
    return personas, section

def build_storyteller_prompt(raw_input: str) -> str:
    personas, section = parse_personas_and_section(raw_input)
    content = raw_input
    m = re.search(r"^(Abstract|Introduction|Methods?|Results?|Conclusion)\s*:\s*", raw_input, re.I|re.M)
    if m:
        start=m.start()
        header=raw_input[start:].splitlines()[0]
        body=raw_input[start+len(header):].strip()
        content=f"{header}\n{body}".strip()
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

# ---------- Dataset ----------
class EvalDataset(Dataset):
    def __init__(self, rows: List[dict], tokenizer, max_src=768,
                 source_field="input", target_field="output"):
        self.rows=rows; self.tok=tokenizer; self.max_src=max_src
        self.source_field=source_field; self.target_field=target_field

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        rec = self.rows[idx]
        src_raw, tgt = map_record(rec, self.source_field, self.target_field)
        prompt = build_storyteller_prompt(src_raw)
        enc = self.tok(prompt, max_length=self.max_src, truncation=True)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "target_text": tgt
        }

def collate_fn(batch, pad_id):
    input_ids=[torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
    attn=[torch.tensor(x["attention_mask"], dtype=torch.long) for x in batch]
    targets=[x["target_text"] for x in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
    attn = torch.nn.utils.rnn.pad_sequence(attn, batch_first=True, padding_value=0)
    return {"input_ids": input_ids, "attention_mask": attn, "targets": targets}

# ---------- Safe decode ----------
def safe_decode(tokenizer, ids: torch.Tensor) -> List[str]:
    arr = ids.detach().cpu().numpy()
    if arr.ndim==3: arr = arr.argmax(axis=-1)
    arr = arr.astype(np.int64, copy=False)
    eos = tokenizer.eos_token_id or tokenizer.pad_token_id or 0
    vsz = getattr(tokenizer, "vocab_size", None)
    arr = np.where(arr<0, eos, arr)
    if vsz is not None: arr = np.where(arr>=vsz, eos, arr)
    try:
        return tokenizer.batch_decode(arr.tolist(), skip_special_tokens=True)
    except Exception:
        out=[]
        for row in arr.tolist():
            row=[eos if (x is None or x<0 or (vsz is not None and x>=vsz)) else int(x) for x in row]
            try: out.append(tokenizer.decode(row, skip_special_tokens=True))
            except Exception: out.append("")
        return out

# ---------- Metrics ----------
def pp_for_rouge(texts: List[str]) -> List[str]:
    out=[]
    for t in texts:
        t=re.sub(r"\s+"," ", (t or "")).strip()
        t=re.sub(r"([.!?])\s+", r"\1\n", t)
        out.append(t)
    return out

def compute_agg_metrics(preds: List[str], refs: List[str]) -> Dict[str,float]:
    rouge = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)
    P=pp_for_rouge(preds); R=pp_for_rouge(refs)
    r1=r2=rL=0.0
    for p,r in zip(P,R):
        s=rouge.score(r,p)
        r1+=s["rouge1"].fmeasure; r2+=s["rouge2"].fmeasure; rL+=s["rougeL"].fmeasure
    n=max(1,len(P))
    r1/=n; r2/=n; rL/=n
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    earlystop = 0.7 * rL + 0.3 * (bleu/100.0)
    return {"rouge1": r1, "rouge2": r2, "rougeL": rL, "bleu": bleu, "earlystop_score": earlystop}

# ---------- Model loader (full o PEFT) ----------
def load_tok_model(model_path: str, device: torch.device):
    if HAS_PEFT:
        try:
            _ = PeftConfig.from_pretrained(model_path)
            tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = AutoPeftModelForSeq2SeqLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if device.type=="cuda" else None,
                device_map="auto" if device.type=="cuda" else None
            )
            return tok, model
        except Exception:
            pass
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device.type=="cuda" else None
    )
    model.to(device)
    return tok, model

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Evaluate FLAN-T5 (fine-tuned o PEFT) su file di test.")
    ap.add_argument("--model", required=True, help="cartella modello (PEFT o full) o nome HF")
    ap.add_argument("--test_file", required=True)
    ap.add_argument("--source_field", default="input")
    ap.add_argument("--target_field", default="output")
    ap.add_argument("--max_source_len", type=int, default=768)
    ap.add_argument("--max_new_tokens", type=int, default=360)
    ap.add_argument("--min_new_tokens", type=int, default=80)
    ap.add_argument("--gen_beams", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--preds_out", type=str, default="eval_test_preds.jsonl")
    ap.add_argument("--results_out", type=str, default="eval_test_metrics.json")
    ap.add_argument("--bertscore_model", type=str, default=None, help="es. roberta-large (opzionale, lento)")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok, model = load_tok_model(args.model, device)
    model.eval()

    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    rows = load_any(args.test_file)
    ds = EvalDataset(rows, tok, max_src=args.max_source_len,
                     source_field=args.source_field, target_field=args.target_field)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    collate_fn=lambda b: collate_fn(b, tok.pad_token_id))

    all_preds=[]; all_refs=[]
    with torch.no_grad():
        for batch in dl:
            inp=batch["input_ids"].to(device)
            att=batch["attention_mask"].to(device)
            gen = model.generate(
                input_ids=inp, attention_mask=att,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                num_beams=args.gen_beams,
                repetition_penalty=1.05,
                do_sample=False,
                pad_token_id=tok.eos_token_id
            )
            preds = safe_decode(tok, gen)
            all_preds.extend(preds)
            all_refs.extend(batch["targets"])

    # metriche aggregate
    metrics = compute_agg_metrics(all_preds, all_refs)

    # opzionale: BERTScore
    if args.bertscore_model:
        try:
            from bert_score import score as bertscore
            P,R,F1 = bertscore(all_preds, all_refs, model_type=args.bertscore_model, lang="en", rescale_with_baseline=True)
            metrics["bertscore_p"]=float(P.mean()); metrics["bertscore_r"]=float(R.mean()); metrics["bertscore_f1"]=float(F1.mean())
        except TypeError:
            from bert_score import score as bertscore
            P,R,F1 = bertscore(all_preds, all_refs, model_type=args.bertscore_model, lang="en", rescale_with_baseline=True)
            metrics["bertscore_p"]=float(P.mean()); metrics["bertscore_r"]=float(R.mean()); metrics["bertscore_f1"]=float(F1.mean())
        except Exception as e:
            metrics["bertscore_error"]=str(e)

    # salva
    with open(args.results_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with open(args.preds_out, "w", encoding="utf-8") as f:
        for pred,ref in zip(all_preds, all_refs):
            f.write(json.dumps({"prediction": pred, "reference": ref}, ensure_ascii=False) + "\n")

    print("[EVAL DONE] Metrics saved to:", args.results_out)
    print(metrics)

if __name__ == "__main__":
    main()
