#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, re, json, argparse, datetime
from typing import List, Tuple, Optional, Dict
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer
import sacrebleu

# PEFT fallback se --model è un adapter LoRA
try:
    from peft import PeftModel
    HAS_PEFT = True
except Exception:
    HAS_PEFT = False

# -----------------------
# Parsing & cleaning MD
# -----------------------
HEADING_RE = re.compile(r'^\s{0,3}(#{2,6})\s+(.+?)\s*$', re.MULTILINE)
INVALID_HEAD_RE = re.compile(r'^(table|figure|references|appendix|acknowledg)', re.I)

ALIASES = {
    "introduction": ["introduction","intro","background","overview","1 introduction"],
    "results": ["results","experiments","evaluation","validation","findings","benchmark",
                "performance","quantitative results","qualitative results","4 validation"],
    "conclusion": ["conclusion","conclusions","summary","discussion","final remarks","5 conclusion"],
}

def strip_metadata(md: str) -> str:
    t = md
    t = re.sub(r'(?s)^---\n.*?\n---\n', '\n', t)                      # YAML
    t = re.sub(r'<!--.*?-->', ' ', t, flags=re.DOTALL)               # commenti HTML
    t = re.sub(r'```.*?```', ' ', t, flags=re.DOTALL)                # code fences
    t = re.sub(r'!\[[^\]]*\]\([^)]+\)', ' ', t)                      # immagini md
    # tabelle
    out, in_tbl = [], False
    for line in t.splitlines():
        if re.match(r'^\s*\|', line): in_tbl = True; continue
        if in_tbl and re.match(r'^\s*[-|: ]+\s*$', line): continue
        if in_tbl and not re.match(r'^\s*\|', line): in_tbl = False
        if not in_tbl: out.append(line)
    t = "\n".join(out)
    # Figure/Table lines
    t = re.sub(r'^\s*(?:Fig\.?|Figure|Table)\s+\d+.*$', ' ', t, flags=re.I|re.M)
    # taglia dopo References
    t = re.split(r'^\s*##\s*References\s*$', t, flags=re.I|re.M)[0]
    # citazioni [1], [2,3]
    t = re.sub(r'\[\d+(?:\s*,\s*\d+)*\]', ' ', t)
    # parentesi “vuote”
    t = re.sub(r'\(\s*[^A-Za-z0-9)]*\)', '', t)
    # de-sillabazioni
    t = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', t)
    t = re.sub(r'(?<=\w)-\s+(?=[a-z])', '', t)
    # newline singoli -> spazio
    t = t.replace('\r','')
    t = re.sub(r'(?<!\n)\n(?!\n)', ' ', t)
    # normalizza spazi
    t = re.sub(r'[ \t]+',' ', t)
    t = re.sub(r'\n{3,}', '\n\n', t)
    return t.strip()

def ordered_sections_from_md(md: str) -> List[Tuple[str,str]]:
    m = list(HEADING_RE.finditer(md))
    if not m:
        return [("Body", md.strip())]
    out = []
    for i, x in enumerate(m):
        title = x.group(2).strip()
        start = x.end()
        end = m[i+1].start() if i+1 < len(m) else len(md)
        out.append((title, md[start:end].strip()))
    return out

def _norm(s: str) -> str:
    return re.sub(r'\s+',' ', re.sub(r'[^a-z0-9\s]',' ', s.lower())).strip()

def _first_idx_alias(ordered: List[Tuple[str,str]], alias_keys: List[str]) -> int:
    aset = set(alias_keys)
    for i,(t,_) in enumerate(ordered):
        if any(a in _norm(t) for a in aset):
            return i
    return -1

def find_by_alias(ordered: List[Tuple[str,str]], target: str) -> Optional[Tuple[str,str]]:
    target = target.lower()
    if target == "introduction":
        idx = _first_idx_alias(ordered, ALIASES["introduction"])
    elif target == "results":
        idx = _first_idx_alias(ordered, ALIASES["results"])
    elif target == "conclusion":
        idx = _first_idx_alias(ordered, ALIASES["conclusion"])
    else:
        idx = -1
    return ordered[idx] if idx >= 0 else None

def infer_methods_between_intro_and_results(ordered: List[Tuple[str,str]]) -> Optional[str]:
    i_intro   = _first_idx_alias(ordered, ALIASES["introduction"])
    i_results = _first_idx_alias(ordered, ALIASES["results"])
    if i_intro < 0 or i_results < 0 or i_results <= i_intro:
        return None
    keep = {"method","methods","methodology","approach","implementation","experimental setup",
            "framework","dataset","architecture","pipeline","model","materials"}
    drop = {"related work","related works","literature review","state of the art"}
    chunks = []
    for t,c in ordered[i_intro+1:i_results]:
        if INVALID_HEAD_RE.match(t): 
            continue
        tl = _norm(t)
        if any(k in tl for k in drop):
            continue
        if keep and not any(k in tl for k in keep):
            num_ratio = sum(ch.isdigit() for ch in c) / max(1, len(c))
            if num_ratio > 0.35:
                continue
        chunks.append(c.strip())
    body = "\n\n".join(x for x in chunks if x)
    return body if body else None

# -----------------------
# PROMPT IDENTICO al training (build_storyteller_prompt)
# -----------------------
def build_storyteller_prompt_training(persona: str, section_name: str, section_text: str) -> str:
    """
    Replica la tua build_storyteller_prompt(style='storyteller'):
      - intestazione 'You are an **AI Scientist Storyteller**...'
      - 'Personas: ...'
      - 'Section: ...'
      - 'Rules:' (bullet identici)
      - 'Content:' seguito da '<SectionName>: <text>'
      - 'Story:'
    """
    content_line = f"{section_name}: {section_text}".strip()
    return (
        "You are an **AI Scientist Storyteller**. Your role is to turn scientific content into engaging, self-contained stories.\n"
        f"Personas: {persona}\n"
        f"Section: {section_name}\n\n"
        "Rules:\n"
        "- Write ONLY the narrative (no lists, JSON, tables, citations, URLs, or DOIs).\n"
        "- 6–9 sentences (~300 words). Third person voice.\n"
        "- Be faithful to the source; keep the text self-contained and coherent.\n"
        "- Adapt tone and vocabulary to the personas.\n\n"
        f"Content:\n{content_line}\n\n"
        "Story:"
    )

# -----------------------
# Utility & Metrics
# -----------------------
def now_ts() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def sanitize(s: str) -> str:
    return re.sub(r'[^a-z0-9]+','_', s.lower()).strip('_')

def compute_rouge_bleu(ref: str, hyp: str) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)
    s = scorer.score(ref, hyp)
    bleu = sacrebleu.corpus_bleu([hyp], [[ref]]).score
    return {"rouge1": s["rouge1"].fmeasure,
            "rouge2": s["rouge2"].fmeasure,
            "rougeL": s["rougeL"].fmeasure,
            "bleu": bleu}

# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser(description="Generate VesselVerse stories using EXACT training prompt (one persona).")
    ap.add_argument("--model", type=str, required=True, help="Cartella del modello SFT (o HF id)")
    ap.add_argument("--base_model", type=str, default="google/flan-t5-large", help="Solo se --model è PEFT adapter")
    ap.add_argument("--md", type=str, required=True, help="Percorso a VesselVerse.md")
    ap.add_argument("--out_dir", type=str, default="./gen_vesselverse_trainprompt")
    ap.add_argument("--tag", type=str, default="sft_trainprompt_exact")
    ap.add_argument("--persona", type=str, default="Student")  # UNA sola persona
    ap.add_argument("--sections", nargs="+",
                    default=["Introduction","Methods","Results","Conclusion"])
    ap.add_argument("--max_source_len", type=int, default=768)
    ap.add_argument("--max_new_tokens", type=int, default=360)
    ap.add_argument("--min_new_tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--num_beams", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    # Carica full o adapter
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model, torch_dtype=torch.float16 if device.type == "cuda" else None
        )
    except Exception:
        if not HAS_PEFT:
            raise
        base = AutoModelForSeq2SeqLM.from_pretrained(
            args.base_model, torch_dtype=torch.float16 if device.type == "cuda" else None
        )
        model = PeftModel.from_pretrained(base, args.model)

    model.to(device); model.eval()

    # Leggi e prepara sezioni
    raw = open(args.md, "r", encoding="utf-8").read()
    cleaned = strip_metadata(raw)
    ordered = ordered_sections_from_md(cleaned)

    used: Dict[str, Tuple[str,str]] = {}

    intro = find_by_alias(ordered, "introduction")
    if intro is None:
        intro = next(((t,c) for (t,c) in ordered if not INVALID_HEAD_RE.match(t)), ordered[0])
    used["Introduction"] = (intro[0], intro[1])

    method_body = infer_methods_between_intro_and_results(ordered)
    if method_body:
        used["Methods"] = ("Methods", method_body)  # nome sezione mostrato nel prompt
    else:
        used["Methods"] = ("Methods", intro[1])

    results = find_by_alias(ordered, "results")
    if results is None:
        # fallback su "Validation" ecc.
        results = next(((t,c) for (t,c) in ordered if "validation" in _norm(t)), None) or intro
    used["Results"] = ("Results", results[1])

    concl = find_by_alias(ordered, "conclusion") or ordered[-1]
    used["Conclusion"] = ("Conclusion", concl[1])

    os.makedirs(args.out_dir, exist_ok=True)
    ts = now_ts()
    metrics_path = os.path.join(args.out_dir, f"metrics_{args.tag}.jsonl")
    with open(metrics_path, "a", encoding="utf-8") as metr_f:

        # vieta URL/DOI espliciti in generazione
        bad_words = ["http", "https", "www", "arxiv", "doi", "url"]
        bad_words_ids = []
        for w in bad_words:
            ids = tok(w, add_special_tokens=False).input_ids
            if ids: bad_words_ids.append(ids)

        for req in args.sections:
            if req not in used:
                print(f"[WARN] '{req}' non disponibile. Skip.")
                continue

            sec_name_for_prompt, body = used[req]  # sec_name_for_prompt = Abstract/Introduction/Methods/Results/Conclusion
            prompt = build_storyteller_prompt_training(args.persona, sec_name_for_prompt, body)

            enc = tok(prompt, return_tensors="pt", truncation=True, max_length=args.max_source_len).to(device)

            gen_kwargs = dict(
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=max(0, args.min_new_tokens),
                do_sample=(args.temperature > 0.0),
                temperature=max(args.temperature, 1e-6),
                top_p=args.top_p,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
                no_repeat_ngram_size=4,
                repetition_penalty=1.12,
                early_stopping=True,
            )
            if not gen_kwargs["do_sample"]:
                gen_kwargs["num_beams"] = max(1, args.num_beams)
            if bad_words_ids:
                gen_kwargs["bad_words_ids"] = bad_words_ids

            with torch.no_grad():
                out_ids = model.generate(**enc, **gen_kwargs)
            out_txt = tok.decode(out_ids[0], skip_special_tokens=True).strip()

            base = f"{ts}_vesselverse_{sanitize(args.persona)}_{sanitize(req)}_{sanitize(sec_name_for_prompt)}_{sanitize(args.tag)}"
            out_txt_path    = os.path.join(args.out_dir, base + ".txt")
            out_prompt_path = os.path.join(args.out_dir, base + ".prompt.txt")
            out_ref_path    = os.path.join(args.out_dir, base + ".ref.txt")

            open(out_txt_path, "w", encoding="utf-8").write(out_txt)
            open(out_prompt_path, "w", encoding="utf-8").write(prompt)
            open(out_ref_path, "w", encoding="utf-8").write(body)

            m = compute_rouge_bleu(body, out_txt)
            row = {
                "timestamp": ts,
                "paper": "VesselVerse",
                "persona": args.persona,
                "section_requested": req,
                "section_used_for_prompt": sec_name_for_prompt,
                "model": args.model,
                "max_source_len": args.max_source_len,
                "max_new_tokens": args.max_new_tokens,
                "min_new_tokens": args.min_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "num_beams": gen_kwargs.get("num_beams", 1),
                "rouge1_f": m["rouge1"], "rouge2_f": m["rouge2"], "rougeL_f": m["rougeL"], "bleu": m["bleu"],
                "output_txt_path": out_txt_path
            }
            metr_f.write(json.dumps(row, ensure_ascii=False) + "\n"); metr_f.flush()

            print("="*80)
            print(f"[{req}] persona={args.persona} → USED SECTION NAME in prompt: {sec_name_for_prompt}")
            print(f"Saved: {out_txt_path}")
            print(f"ROUGE-L={m['rougeL']:.3f} | BLEU={m['bleu']:.2f}")

    print("\n[DONE] Files @", args.out_dir)
    print("[METRICS] →", metrics_path)

if __name__ == "__main__":
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass
    main()
