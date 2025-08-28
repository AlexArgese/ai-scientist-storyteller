# run_story_flan_t5_raw.py
# Esempi:
#  1) Batch (Intro, Methods, Results, Conclusion):
#     python run_story_flan_t5_raw.py --md /path/VesselVerse.md \
#       --sections Introduction Methods Results Conclusion \
#       --model google/flan-t5-large \
#       --max_input_tokens 480 --max_new_tokens 360 --min_new_tokens 140 \
#       --temperature 0.0 --top_p 1.0 \
#       --bertscore_model roberta-large \
#       --out_dir ./outputs/flant5_raw --save_prompt --save_ref \
#       --run_tag flan_raw_baseline
#
#  2) Singola sezione:
#     python run_story_flan_t5_raw.py --md /path/VesselVerse.md --section Methods ...

import re, os, argparse, json, csv
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
import random
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer
import sacrebleu
from bert_score import score as bertscore

# -----------------------
# Utils: cleaning & split
# -----------------------
HEADING_RE = re.compile(r'^\s{0,3}(#{2,6})\s+(.+?)\s*$', re.MULTILINE)

def load_md(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def strip_metadata(md: str) -> str:
    """
    Pulisce il Markdown estratto rimuovendo:
    - front-matter YAML, commenti HTML, blocchi codice, immagini
    - tabelle markdown e linee "Figure/Table N"
    - sezione References e tutto ciò che segue
    - citazioni [1], [2,3]
    - parentesi orfane tipo "(, )" o "(e.g., )"
    - sillabazioni spezzate e newline singoli dentro paragrafi
    """
    text = md
    text = re.sub(r'(?s)^---\n.*?\n---\n', '\n', text)                 # front-matter
    text = re.sub(r'<!--.*?-->', ' ', text, flags=re.DOTALL)           # commenti HTML
    text = re.sub(r'```.*?```', ' ', text, flags=re.DOTALL)            # code blocks
    text = re.sub(r'!\[[^\]]*\]\([^)]+\)', ' ', text)                  # immagini

    # tabelle markdown
    out_lines, in_table = [], False
    for line in text.splitlines():
        if re.match(r'^\s*\|', line): in_table = True; continue
        if in_table and re.match(r'^\s*[-|: ]+\s*$', line): continue
        if in_table and not re.match(r'^\s*\|', line): in_table = False
        if not in_table: out_lines.append(line)
    text = "\n".join(out_lines)

    # righe Figure/Table
    text = re.sub(r'^\s*(?:Fig\.?|Figure|Table)\s+\d+.*$', ' ', text,
                  flags=re.IGNORECASE | re.MULTILINE)

    # References (taglia tutto il seguito)
    text = re.split(r'^\s*##\s*References\s*$', text,
                    flags=re.IGNORECASE | re.MULTILINE)[0]

    # citazioni [1], [2,3]
    text = re.sub(r'\[\d+(?:\s*,\s*\d+)*\]', ' ', text)

    # parentesi orfane: solo punteggiatura/spazi oppure (e.g./i.e. + punteggiatura)
    text = re.sub(r'\(\s*[^A-Za-z0-9)]*\)', '', text)  # solo punteggiatura/spazi
    text = re.sub(r'\(\s*(?:e\.g\.|i\.e\.)?\s*[,;:]?\s*\)', '', text, flags=re.IGNORECASE)

    # de-sillabazione newline + intra-line
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
    text = re.sub(r'(?<=\w)-\s+(?=[a-z])', '', text)

    # unisci newline singoli nei paragrafi
    text = text.replace('\r','')
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

    # normalizza spazi
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def split_sections(md: str) -> Dict[str, str]:
    """Restituisce dict {section_title: content} usando headings Markdown (##, ###, ...)."""
    sections = {}
    matches = list(HEADING_RE.finditer(md))
    if not matches:
        sections["Body"] = md
        return sections
    for i, m in enumerate(matches):
        level = len(m.group(1))
        title = m.group(2).strip()
        start = m.end()
        end = len(md)
        for m2 in matches[i+1:]:
            if len(m2.group(1)) <= level:
                end = m2.start()
                break
        content = md[start:end].strip()
        sections[title] = content
    return sections

# -----------------------
# Alias + euristica Methods (Intro → Results/Validation)
# -----------------------
ALIASES = {
    "introduction": ["introduction","intro","background","overview"],
    "methods": ["method","methods","methodology","materials and methods","approach",
                "implementation details","experimental setup","experiments","framework"],
    "results": ["results","experiments","evaluation","validation","findings",
                "quantitative results","qualitative results","benchmark","performance"],
    "conclusion": ["conclusion","conclusions","summary","final remarks"],
}
_INVALID_HEAD_RE = re.compile(r'^(table|figure|references|appendix|acknowledg)', re.I)

def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def ordered_sections_from_md(md: str) -> List[Tuple[str,str]]:
    """Lista ordinata [(title, content)] via heading Markdown."""
    matches = list(HEADING_RE.finditer(md))
    if not matches:
        return [("Body", md.strip())]
    ordered = []
    for i, m in enumerate(matches):
        title = m.group(2).strip()
        start = m.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(md)
        content = md[start:end].strip()
        ordered.append((title, content))
    return ordered

def _first_index_with_alias(ordered: List[Tuple[str,str]], alias_keys: List[str]) -> int:
    aliases = set(alias_keys)
    for idx, (t, _) in enumerate(ordered):
        tl = _norm(t)
        if any(a in tl for a in aliases):
            return idx
    return -1

def _looks_like_table_block(s: str, thresh=0.35) -> bool:
    lines = [l for l in s.splitlines() if l.strip()]
    if not lines: return False
    num_ratio = sum(ch.isdigit() for ch in s) / max(1, len(s))
    return num_ratio > thresh

def infer_methods_between_intro_and_results(ordered: List[Tuple[str,str]]) -> Tuple[str, str]:
    """
    Se non c'è un titolo 'Methods', inferisce 'Method (inferred)' concatenando
    i blocchi tra 'Introduction' e 'Results/Validation/...' filtrando Related Works.
    """
    intro_alias   = ALIASES["introduction"]
    results_alias = ALIASES["results"]
    i_intro   = _first_index_with_alias(ordered, intro_alias)
    i_results = _first_index_with_alias(ordered, results_alias)
    if i_intro < 0 or i_results < 0 or i_results <= i_intro:
        return None, None

    keep_keys = {"method","methods","methodology","approach","implementation",
                 "experimental setup","framework","dataset","architecture","pipeline","model"}
    drop_keys = {"related work","related works","literature review","state of the art"}

    chunks = []
    for t, c in ordered[i_intro+1:i_results]:
        tl = _norm(t)
        if _INVALID_HEAD_RE.match(t):            # salta Table/Figure/Refs/Appendix
            continue
        if any(k in tl for k in drop_keys):      # salta Related/Literature
            continue
        if keep_keys and not any(k in tl for k in keep_keys):
            if _looks_like_table_block(c):
                continue
        chunks.append(c.strip())

    body = "\n\n".join(x for x in chunks if x)
    return ("Method (inferred)", body if body else None)

def _is_valid_heading(h: str) -> bool:
    return not _INVALID_HEAD_RE.match(h)

def find_section_name(candidates: List[str], wanted: str) -> str:
    """Matching robusto con alias e filtro heading non-sezione."""
    wanted_low = wanted.lower().strip()
    cand = [t for t in candidates if _is_valid_heading(t)]
    # exact
    for t in cand:
        if t.lower() == wanted_low:
            return t
    # substring
    for t in cand:
        if wanted_low in t.lower():
            return t
    # alias
    if wanted_low in ALIASES:
        keys = ALIASES[wanted_low]
        for t in cand:
            tl = t.lower()
            if any(k in tl for k in keys):
                return t
    # fallback “sensato”
    prefs = {
        "introduction": ["introduction","background"],
        "methods": ["method","methods","experimental","approach","implementation"],
        "results": ["results","experiments","evaluation","validation"],
        "conclusion": ["conclusion","summary"],
    }.get(wanted_low, [])
    for p in prefs:
        for t in cand:
            if p in t.lower():
                return t
    return cand[0] if cand else (candidates[0] if candidates else "Body")

# -----------------------
# Prompting & Generation
# -----------------------
def make_prompt(section_text: str) -> str:
    return (
        "You are an AI Scientist Storyteller. Turn the following scientific section into an engaging, self-contained story for a university student.\n"
        "Rules:\n"
        "- 6–9 sentences (~300 words)\n"
        "- Be faithful to the source but explain clearly\n"
        "- No references, citations, URLs, DOIs, tables, or figures\n"
        "- Write in third person; avoid 'I', 'we', or personal anecdotes.\n\n"
        f"Section:\n{section_text}\n\n"
        "Story:"
    )

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def generate_story(
    model_name: str,
    prompt: str,
    max_input_tokens: int = 480,
    max_new_tokens: int = 360,
    min_new_tokens: int = 140,
    temperature: float = 0.0,
    top_p: float = 1.0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else None
    ).to(device)

    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens)
    enc = {k: v.to(device) for k, v in enc.items()}
    input_len = int(enc["input_ids"].shape[1])

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        num_beams=1,                 # greedy
        repetition_penalty=1.15,     # più forte per ridurre ripetizioni
        no_repeat_ngram_size=4,
    )
    if isinstance(min_new_tokens, int) and min_new_tokens > 0:
        gen_kwargs["min_new_tokens"] = min_new_tokens

    if temperature and temperature > 0.0:
        gen_kwargs.update(dict(do_sample=True, temperature=temperature, top_p=top_p))
    else:
        gen_kwargs.update(dict(do_sample=False))

    with torch.no_grad():
        out = model.generate(**enc, **gen_kwargs)

    text = tok.decode(out[0], skip_special_tokens=True).strip()
    gen_len = int(out.shape[1] - input_len)
    return text, input_len, gen_len

# -----------------------
# Metrics
# -----------------------
def compute_rouge(pred: str, ref: str) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)
    s = scorer.score(ref, pred)
    return {k: v.fmeasure for k, v in s.items()}

def compute_bleu(pred: str, ref: str) -> float:
    return sacrebleu.corpus_bleu([pred], [[ref]]).score

def compute_bertscore(pred: str, ref: str, model: str = "roberta-large") -> Dict[str, float]:
    try:
        P,R,F1 = bertscore([pred], [ref], model_type=model, lang="en", rescale_with_baseline=True)
        return {"precision": float(P[0]), "recall": float(R[0]), "f1": float(F1[0])}
    except Exception as e:
        print(f"[WARN] Skipping BERTScore due to error: {e}")
        return {"precision": float('nan'), "recall": float('nan'), "f1": float('nan')}

# -----------------------
# Saving helpers
# -----------------------
def ensure_dir(p: str): os.makedirs(p, exist_ok=True)
def safe_slug(s: str) -> str:
    s = s.lower()
    s = re.sub(r'[^a-z0-9]+', '_', s).strip('_')
    return s or "untitled"

def append_jsonl(path: str, obj: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def append_csv(path: str, row: dict, fieldnames: List[str]):
    file_exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists: w.writeheader()
        w.writerow(row)

# -----------------------
# Core per sezione
# -----------------------
def process_section(paper_name: str,
                    cleaned: str,
                    sections_map: Dict[str,str],
                    ordered_sections: List[Tuple[str,str]],
                    desired_label: str,
                    args,
                    ts_group: str):
    # 1) scegli la sezione
    chosen_title = find_section_name(list(sections_map.keys()), desired_label)

    # 2) euristica Methods se non esiste titolo "method"
    if desired_label.lower().startswith("method"):
        invalid = (
            chosen_title.lower() in {"body", "introduction"} or
            _INVALID_HEAD_RE.match(chosen_title) or
            ("method" not in chosen_title.lower())
        )
        if invalid:
            inferred_title, inferred_body = infer_methods_between_intro_and_results(ordered_sections)
            if inferred_body:
                chosen_title = inferred_title
                sections_map[chosen_title] = inferred_body

    section_text = sections_map[chosen_title].strip()

    # 3) prompt + generazione
    prompt = make_prompt(section_text)
    pred, in_len, gen_len = generate_story(
        model_name=args.model, prompt=prompt,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        temperature=args.temperature, top_p=args.top_p
    )

    # 4) metriche
    rouge = compute_rouge(pred, section_text)
    bleu  = compute_bleu(pred, section_text)
    bert  = compute_bertscore(pred, section_text, model=args.bertscore_model)

    # 5) salvataggi
    ensure_dir(args.out_dir)
    base = f"{ts_group}_{safe_slug(paper_name)}_{safe_slug(chosen_title)}_{safe_slug(args.model)}"
    out_txt  = os.path.join(args.out_dir, base + ".txt")
    with open(out_txt, "w", encoding="utf-8") as f: f.write(pred)

    if args.save_prompt:
        with open(os.path.join(args.out_dir, base + ".prompt.txt"), "w", encoding="utf-8") as f:
            f.write(prompt)
    if args.save_ref:
        with open(os.path.join(args.out_dir, base + ".ref.txt"), "w", encoding="utf-8") as f:
            f.write(section_text)

    record = {
        "timestamp": ts_group,
        "run_tag": args.run_tag,
        "paper": paper_name,
        "section_requested": desired_label,
        "section_used": chosen_title,
        "model": args.model,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "bertscore_model": args.bertscore_model,
        "input_token_len": in_len,
        "gen_token_len": gen_len,
        "rouge1_f": float(rouge["rouge1"]),
        "rouge2_f": float(rouge["rouge2"]),
        "rougeL_f": float(rouge["rougeL"]),
        "bleu": float(bleu),
        "bertscore_p": float(bert["precision"]) if not np.isnan(bert["precision"]) else None,
        "bertscore_r": float(bert["recall"]) if not np.isnan(bert["recall"]) else None,
        "bertscore_f1": float(bert["f1"]) if not np.isnan(bert["f1"]) else None,
        "output_txt_path": out_txt,
    }
    append_jsonl(os.path.join(args.out_dir, "results.jsonl"), record)

    csv_fields = ["timestamp","run_tag","paper","section_requested","section_used","model",
                  "max_input_tokens","max_new_tokens","min_new_tokens","temperature","top_p","seed",
                  "bertscore_model","input_token_len","gen_token_len",
                  "rouge1_f","rouge2_f","rougeL_f","bleu","bertscore_p","bertscore_r","bertscore_f1",
                  "output_txt_path"]
    append_csv(os.path.join(args.out_dir, "results.csv"), record, csv_fields)

    # 6) stampa breve
    print("="*80)
    print(f"[{desired_label}]  →  USED SECTION: {chosen_title}")
    print("="*80)
    print("GENERATED STORY:\n" + pred + "\n")
    print("METRICS:")
    print(f"  ROUGE-1 F1: {rouge['rouge1']:.4f} | ROUGE-2 F1: {rouge['rouge2']:.4f} | ROUGE-L F1: {rouge['rougeL']:.4f}")
    print(f"  BLEU: {bleu:.2f}")
    if not np.isnan(record["bertscore_f1"] or np.nan):
        print(f"  BERTScore  P: {record['bertscore_p']:.4f}  R: {record['bertscore_r']:.4f}  F1: {record['bertscore_f1']:.4f}")
    else:
        print("  BERTScore: skipped")
    print(f"[Saved] {out_txt}\n")

# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", type=str, required=True)
    # singola sezione OPPURE batch
    ap.add_argument("--section", type=str, help="Sezione singola (es. Introduction)")
    ap.add_argument("--sections", nargs="+", help="Elenco sezioni per il batch (es. Introduction Methods Results Conclusion)")
    ap.add_argument("--model", type=str, default="google/flan-t5-large")
    ap.add_argument("--max_input_tokens", type=int, default=480)
    ap.add_argument("--max_new_tokens", type=int, default=360)
    ap.add_argument("--min_new_tokens", type=int, default=140)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bertscore_model", type=str, default="roberta-large")
    # Salvataggi
    ap.add_argument("--out_dir", type=str, default="./outputs/flant5_raw")
    ap.add_argument("--save_prompt", action="store_true")
    ap.add_argument("--save_ref", action="store_true")
    ap.add_argument("--run_tag", type=str, default="")
    args = ap.parse_args()

    if not args.section and not args.sections:
        # default batch comodo
        args.sections = ["Introduction","Methods","Results","Conclusion"]

    set_seed(args.seed)

    raw = load_md(args.md)
    cleaned = strip_metadata(raw)
    sections_map = split_sections(cleaned)          # {heading: content}
    ordered_sections = ordered_sections_from_md(cleaned)  # [(heading, content)]

    paper_name = os.path.splitext(os.path.basename(args.md))[0]
    ts_group = datetime.now().strftime("%Y%m%d_%H%M%S")

    tasks = args.sections if args.sections else [args.section]

    print("[DEBUG] Headings found:", " | ".join(list(sections_map.keys())[:60]))
    print(f"[INFO] Generating for sections: {tasks}\n")

    for desired_label in tasks:
        process_section(
            paper_name=paper_name,
            cleaned=cleaned,
            sections_map=sections_map,
            ordered_sections=ordered_sections,
            desired_label=desired_label,
            args=args,
            ts_group=ts_group
        )

if __name__ == "__main__":
    # Suggerimento env (cartelle scrivibili):
    # export HF_HOME=/docker/argese/volumes/raw_experiment/.cache/huggingface
    # export MPLCONFIGDIR=/docker/argese/volumes/raw_experiment/.cache/matplotlib
    # export TOKENIZERS_PARALLELISM=false
    main()

