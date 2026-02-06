#!/usr/bin/env python3
# infer_splitter.py 

import argparse, os, re, json, sys, math, random
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel

from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

TRACE_LOG_FILE = os.environ.get("TRACE_LOG_FILE")
TRACE_REQ_ID = os.environ.get("TRACE_REQ_ID", "-")

def _now_iso():
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"

def trace(event: str, message: str = "", **data):
    rec = {"ts": _now_iso(), "req_id": TRACE_REQ_ID, "event": event, "message": message, **data}
    line = json.dumps(rec, ensure_ascii=False)
    # write to file (if provided)
    if TRACE_LOG_FILE:
        try:
            with open(TRACE_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass
    try:
        print(f"[trace] {event} {message} :: {json.dumps(data, ensure_ascii=False)}", file=sys.stderr, flush=True)
    except Exception:
        pass

# -----------------------------
# Cache helpers (avoid disk quota in $HOME)
# -----------------------------
def get_cache_dir() -> Optional[str]:
    return os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE") or None

# -----------------------------
# Seed 
# -----------------------------
def set_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -----------------------------
# Persona rubrics 
# -----------------------------
PERSONA_RUBRICS: Dict[str, Dict[str, Any]] = {
    "General Public": {
        "expertise": "Low",
        "goal": "Understand what AI is and why it matters.",
        "style": (
            "Use simple, curiosity-driven language. Avoid jargon and equations. "
            "Give relatable examples and explain why each section is relevant to everyday life."
        ),
        "must_have_any": ["Background", "Real-world impact", "Why it matters"],
        "avoid": ["Ablation", "Implementation details", "Mathematical proofs"],
    },

    "Investor": {
        "expertise": "Low–Medium",
        "goal": "Spot AI trends for business or funding decisions.",
        "style": (
            "Focus on market potential, differentiation, scalability, and risks. "
            "Avoid technical depth unless tied to business value."
        ),
        "must_have_any": ["Market/Applications", "Value Proposition", "Risks"],
        "avoid": ["Ablation", "Proofs", "Training losses"],
    },

    "Student": {
        "expertise": "Medium",
        "goal": "Learn AI fundamentals and expand technical knowledge.",
        "style": (
            "Use educational tone with short definitions and examples. "
            "Highlight key concepts, motivation, and what the reader should learn."
        ),
        "must_have_any": ["Key Concepts", "Motivation", "Takeaways"],
        "avoid": [],
    },

    "Journalist": {
        "expertise": "Medium",
        "goal": "Report clearly and accurately on AI developments.",
        "style": (
            "Write as if explaining to an informed but non-technical audience. "
            "Emphasize significance, societal implications, and credible results."
        ),
        "must_have_any": ["Background", "Results", "Implications"],
        "avoid": ["Mathematical derivations", "Code-level details"],
    },

    "Policy Maker": {
        "expertise": "Medium–High",
        "goal": "Assess the social, ethical, and legal implications of AI.",
        "style": (
            "Prioritize transparency, accountability, and governance. "
            "Avoid deep technical discussion; emphasize societal impact and ethics."
        ),
        "must_have_any": ["Ethical Implications", "Regulation", "Societal Impact"],
        "avoid": ["Optimization details", "Model internals"],
    },

    "Teacher": {
        "expertise": "High",
        "goal": "Teach AI concepts and methods effectively to others.",
        "style": (
            "Organize clearly for instructional purposes. Emphasize learning objectives, "
            "key examples, and limitations to encourage critical thinking."
        ),
        "must_have_any": ["Learning Objectives", "Examples", "Limitations"],
        "avoid": [],
    },

    "Researchers & Engineers": {
        "expertise": "High",
        "goal": "Produce, implement, and evaluate advanced AI research and systems.",
        "style": (
            "Be concise, technical, and implementation-aware. Highlight novelty, methodology, datasets, "
            "metrics, results, engineering tradeoffs, and limitations."
        ),
        "must_have_any": [
            "Methodology",
            "Experiments",
            "Results",
            "Limitations",
            "Implementation Details",
        ],
        "avoid": ["Marketing tone", "Oversimplified analogies"],
    },
}

# -----------------------------
# Markdown cleaning 
# -----------------------------
COMMENT_RE = re.compile(r'^\s*<!--\s*(\{.*?\})\s*-->\s*$', re.IGNORECASE)
H1_RE = re.compile(r'^\s*#\s+(.*)')
HX_RE = re.compile(r'^\s*#{1,6}\s+(.*)')
STOP_HEAD_RE = re.compile(
    r'^\s*#{1,6}\s+(references?|bibliography|appendix|acknowledgements?)\b',
    re.IGNORECASE
)

def _is_table_line(line: str) -> bool:
    if line.strip().startswith("|"):
        return True
    s = line.strip().replace(" ", "")
    if set(s) <= {"|", "-", ":"} and ("|" in s or "-" in s):
        return True
    return False

def _looks_like_figure_caption(line: str) -> bool:
    s = line.strip()
    return bool(re.match(r'^(fig(\.|ure)?|tab(\.|le)?)\s*[\d:.\- ]', s, re.IGNORECASE))

def fix_line_wraps(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"(\w)[-\u2010\u2011]\s*\n\s*(\w)", r"\1\2", text)
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = re.sub(r"(\b[A-Za-z]{2,})-\s+([a-z]{2,}\b)", r"\1\2", text)
    text = re.sub(r"([a-z,;:])\n([a-z])", r"\1 \2", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text

DROP_HEADING_RE = re.compile(r'^\s*#{1,6}\s*(table|fig(?:\.|ure)?)\b', re.IGNORECASE)

def normalize_spacing_and_punct(text: str) -> str:
    text = re.sub(r'\s+([,.;:!?%])', r'\1', text)
    text = re.sub(r' +([\)\]\}])', r'\1', text)
    text = re.sub(r'([\(\[\{]) +', r'\1', text)
    text = re.sub(r'[ ]{2,}', ' ', text)
    text = re.sub(r'\s+’s\b', "’s", text)
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")
    text = re.sub(r'\bbest\s*ranked\b', 'best-ranked', text, flags=re.IGNORECASE)
    return text

def fix_common_academic_artifacts(text: str) -> str:
    pairs = {
        r'\blargescale\b': 'large-scale',
        r'\binterrater\b': 'inter-rater',
        r'\bwithing\b': 'within',
        r'\bnnU-\s*Net\b': 'nnU-Net',
    }
    for pat, rep in pairs.items():
        text = re.sub(pat, rep, text, flags=re.IGNORECASE)
    return text

def drop_figure_table_headings(text: str) -> str:
    lines = []
    for ln in text.splitlines():
        if DROP_HEADING_RE.match(ln):
            continue
        lines.append(ln)
    return "\n".join(lines)

def clean_pdf_markdown(md_text: str) -> Tuple[Optional[str], str]:
    title = None
    out_lines: List[str] = []
    skip_types = {"table", "image", "figure", "header", "footer"}
    current_block_type: Optional[str] = None
    stop_now = False

    lines = md_text.splitlines()
    for line in lines:
        if stop_now:
            break

        m = COMMENT_RE.match(line)
        if m:
            try:
                meta = json.loads(m.group(1))
                t = str(meta.get("type", "")).lower()
                current_block_type = t if t else None
            except Exception:
                current_block_type = None
            continue

        if STOP_HEAD_RE.match(line):
            stop_now = True
            break

        if current_block_type in skip_types:
            continue

        if _is_table_line(line): continue
        if re.search(r'!\[[^\]]*\]\([^)]+\)', line): continue
        if _looks_like_figure_caption(line): continue

        m1 = H1_RE.match(line)
        if m1 and title is None:
            title = m1.group(1).strip()
            continue

        out_lines.append(line)
        if HX_RE.match(line):
            current_block_type = None

    text = "\n".join(out_lines).strip()
    text = fix_line_wraps(text)
    text = drop_figure_table_headings(text)
    text = normalize_spacing_and_punct(text)
    text = fix_common_academic_artifacts(text)
    return title, text

# -----------------------------
# Prompt building 
# -----------------------------
INSTR_TEMPLATE = (
    "You are an AI paper splitter. Split the paper into exactly {n_sections} logical sections "
    "tailored to the specified Persona."
)

def _persona_block(persona: str, style_strength: int, n_sections: int) -> str:
    spec = PERSONA_RUBRICS.get(persona, PERSONA_RUBRICS["General Public"])
    must_any = spec.get("must_have_any", [])
    avoid = spec.get("avoid", [])
    style = spec.get("style", "")

    lines = [
        f"Persona: {persona}",
        f"Persona style (strength {style_strength}/5): {style}",
        f"Target sections: {n_sections}",
        "All titles/descriptions MUST be grounded in the paper text (no fabrication).",
    ]
    if must_any:
        lines.append(f"At least one section title MUST include one of: {', '.join(must_any)}.")
    if avoid:
        lines.append(f"Avoid section titles about: {', '.join(avoid)}.")
    return "\n".join(lines)

def build_prompt(persona: str, title: str, paper_text: str,
                 n_sections: int = 5, style_strength: int = 3,
                 include_few_shot: bool = False) -> str:

    instr = INSTR_TEMPLATE.format(n_sections=n_sections if isinstance(n_sections, int) else "~5")
    persona_block = _persona_block(persona, style_strength, n_sections)

    few_shot = ""
    if include_few_shot:
        few_shot = (
            "\n<EXEMPLARS>\n"
            "[Persona: Investor]\n"
            "[{\"title\":\"Background on Generative Models\",\"description\":\"Brief context relevant for non-experts.\"},"
            "{\"title\":\"Architecture\",\"description\":\"High-level components tied to differentiation.\"},"
            "{\"title\":\"Applications & Moat\",\"description\":\"Potential markets, risks, differentiators.\"}]\n"
            "[Persona: Researchers & Engineers]\n"
            "[{\"title\":\"Model Architecture\",\"description\":\"Key modules and design choices.\"},"
            "{\"title\":\"Training & Evaluation\",\"description\":\"Objectives, datasets, metrics.\"},"
            "{\"title\":\"Results & Ablations\",\"description\":\"Comparisons, ablations, failure modes.\"}]\n"
            "</EXEMPLARS>\n"
        )

    schema = (
        "<SCHEMA>\n"
        "Return ONLY a JSON list of objects, no markdown, no preface.\n"
        "Schema: [ {{\"title\":\"...\",\"description\":\"...\"}}, ... ]\n"
        "Constraints:\n"
        f"- Exactly {n_sections} items.\n"
        "- Titles: <= 10 words, persona-appropriate.\n"
        "- Descriptions: 1–2 sentences, grounded in the paper.\n"
        "- English only.\n"
        "</SCHEMA>\n"
    )


    prompt = (
        "<INSTRUCTION>\n" + instr + "\n</INSTRUCTION>\n"
        "<PERSONA_GUIDE>\n" + persona_block + "\n</PERSONA_GUIDE>\n"
        + few_shot +
        "<TITLE>\n" + (title or "Untitled") + "\n</TITLE>\n"
        "<PAPER>\n" + paper_text + "\n</PAPER>\n"
        + schema +
        "<RESPONSE>\n"
        "Return the list inside <JSON> tags strictly.\n"
        "<JSON>\n"
    )
    return prompt

# -----------------------------
# JSON extraction
# -----------------------------
def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.I)
    return s.strip()

def extract_json_list(text: str) -> List[Dict[str, Any]]:
    """
    Estrae la PRIMA lista JSON plausibile.
    Priorità: contenuto tra <JSON>...</JSON>, altrimenti primo blocco [ ... ] bilanciato.
    """
    s = _strip_code_fences(text)
    # 1) tag <JSON>
    m = re.search(r"<JSON>\s*(\[.*?\])\s*</JSON>", s, flags=re.S)
    if m:
        cand = m.group(1)
        try:
            obj = json.loads(cand)
            if isinstance(obj, list):
                return obj
        except Exception:
            pass
    # 2) primo [ ... ] bilanciato
    start = s.find("[")
    if start != -1:
        depth, i, in_str, esc = 0, start, False, False
        while i < len(s):
            ch = s[i]
            if in_str:
                if esc: esc = False
                elif ch == "\\": esc = True
                elif ch == '"': in_str = False
            else:
                if ch == '"': in_str = True
                elif ch == "[": depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        chunk = s[start:i+1]
                        try:
                            obj = json.loads(chunk)
                            if isinstance(obj, list):
                                return obj
                        except Exception:
                            break
            i += 1
    # 3) fallback: prova parse diretto
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass
    return []

# -----------------------------
# Model loading
# -----------------------------
def load_model_and_tokenizer(base_model: str, adapter_path: str):
    cache_dir = get_cache_dir()
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True, cache_dir=cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        cache_dir=cache_dir,
    )
    try:
        model = PeftModel.from_pretrained(base, adapter_path, cache_dir=cache_dir)
    except TypeError:
        model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    # usa cache per stabilità/latenza
    try:
        model.config.use_cache = True
    except Exception:
        pass
    return tok, model

# -----------------------------
# Truncation budgeting
# -----------------------------
def max_ctx_len(model) -> int:
    return int(getattr(model.config, "max_position_embeddings", 8192))

def truncate_for_budget(tok, model, prompt_prefix: str, paper_text: str, max_new_tokens: int, safety_margin: int = 256) -> Tuple[str, bool]:
    max_pos = max_ctx_len(model)
    # tokenizza prefix + paper per stimare
    ids_prefix = tok(prompt_prefix, add_special_tokens=False).input_ids
    ids_paper = tok(paper_text, add_special_tokens=False).input_ids

    budget_input = max_pos - max_new_tokens - safety_margin
    need = len(ids_prefix) + len(ids_paper)
    if need <= budget_input:
        return paper_text, False

    # taglia paper_text per rientrare
    allow = max(0, budget_input - len(ids_prefix))
    if allow <= 0:
        keep_ids = ids_paper[-max(0, budget_input // 2):]
    else:
        keep_ids = ids_paper[:allow]
    paper_trunc = tok.decode(keep_ids, skip_special_tokens=True)
    return paper_trunc, True

# -----------------------------
# Generation
# -----------------------------
def normalize_title(t: str) -> str:
    s = (t or "").strip()
    s = re.sub(r'^\s*#+\s*', '', s)               
    s = re.sub(r'^\s*\d+[\).\-:\s]+\s*', '', s)    
    s = s.strip('"').strip("'").strip("`").strip()
    s = re.sub(r'\s{2,}', ' ', s)
    return s

def generate_sections(
    tok, model,
    persona: str, title: str, paper_text: str,
    max_new_tokens: int = 768,
    temperature: float = 0.0,
    top_p: float = 0.9,
    n_sections: int = 5,
    style_strength: int = 3,
    include_few_shot: bool = False,
    truncation_margin: int = 256,
    stderr_log: bool = True):

    skeleton = (
        "<INSTRUCTION>\n"
        f"You are an AI paper splitter. Split the paper into exactly {n_sections} logical sections tailored to the Persona.\n"
        "</INSTRUCTION>\n"
        "<PERSONA_GUIDE>\n" + _persona_block(persona, style_strength, n_sections) + "\n</PERSONA_GUIDE>\n"
        "<TITLE>\n" + (title or "Untitled") + "\n</TITLE>\n"
        "<PAPER>\n"
    )

    paper_fit, was_trunc = truncate_for_budget(
        tok, model, prompt_prefix=skeleton, paper_text=paper_text,
        max_new_tokens=max_new_tokens, safety_margin=truncation_margin
    )

    if was_trunc and stderr_log:
        cut_pct = 100.0 * (1.0 - (len(paper_fit) / max(1, len(paper_text))))
        print(f"[splitter] WARNING: paper truncated by ~{cut_pct:.1f}% to fit context budget", file=sys.stderr)

    prompt = build_prompt(
        persona, title, paper_fit,
        n_sections=n_sections, style_strength=style_strength,
        include_few_shot=include_few_shot
    )

    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=max_ctx_len(model)).to(model.device)

    do_sample = temperature > 0.0
    gen_cfg = GenerationConfig(
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p) if do_sample else 1.0,
        do_sample=do_sample,
        repetition_penalty=1.05 if do_sample else 1.0,
        typical_p=0.95 if do_sample else None,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    with torch.no_grad():
        out = model.generate(**inputs, generation_config=gen_cfg)
    text_out_full = tok.decode(out[0], skip_special_tokens=True)

    sections = extract_json_list(text_out_full)
    try:
        for idx, sec in enumerate(sections):
            t = sec.get("title") or f"Section {idx+1}"
            trace("outline.section.start", f"Generating outline section '{t}'", index=idx, title=t)
            print(json.dumps({
                "type": "splitter_progress",
                "index": idx,
                "title": t
            }), flush=True)
    except Exception:
        pass

    cleaned = []
    for it in sections:
        if isinstance(it, dict):
            t = normalize_title(str(it.get("title", "")))
            d = str(it.get("description", "")).strip()
            if t or d:
                cleaned.append({"title": t, "description": d})
    return cleaned

# -----------------------------
# Input handling
# -----------------------------
TEXT_KEYS = ["paper_text", "text", "content", "full_text", "body"]
TITLE_KEYS = ["paper_title", "title"]

def _load_inputs_any(path: str) -> List[Dict[str, Any]]:
    """
    Supports:
      - .jsonl with records containing: persona + (paper_text|text|content) OR markdown in 'md'
      - .md/.txt: requires --persona
    """
    if path.lower().endswith(".jsonl"):
        ds = load_dataset("json", data_files=path, split="train")
        records = []
        for ex in ds:
            persona = ex.get("persona") or "General Public"
            md = ex.get("md")
            raw_text = None
            if md:
                raw_text = md
            else:
                for k in TEXT_KEYS:
                    if k in ex and ex[k]:
                        raw_text = ex[k]
                        break
            if raw_text is None:
                continue
            title = None
            for tk in TITLE_KEYS:
                if tk in ex and ex[tk]:
                    title = ex[tk]
                    break
            rec_id = ex.get("id") or ex.get("id_paper") or ex.get("paper_id") or None
            records.append({"id": rec_id, "persona": persona, "md": raw_text, "title": title})
        return records
    else:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return [{"id": None, "persona": None, "md": content, "title": None}]

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, required=True,
                    help="e.g. Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter_path", type=str, required=True,
                    help="e.g. out-splitter-qwen7b/checkpoint-100")
    ap.add_argument("--input_path", type=str, required=True,
                    help="Markdown/TXT file OR JSONL with persona+text or md")
    ap.add_argument("--output_jsonl", type=str, required=True,
                    help="Output JSONL with sections and metadata")

    ap.add_argument("--persona", type=str, default=None,
                    help="Required if input_path is a single .md/.txt file")
    ap.add_argument("--id", type=str, default=None,
                    help="Optional id for single file input")

    ap.add_argument("--max_new_tokens", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="Creativity of section generation (0.0 = deterministic)")
    ap.add_argument("--top_p", type=float, default=0.9,
                    help="Nucleus sampling for splitter when temperature > 0.")

    # Persona-aware knobs
    ap.add_argument("--sections", type=int, default=5,
                    help="Target number of sections to return.")
    ap.add_argument("--style_strength", type=int, default=3,
                    help="How strongly to bias titles/descriptions toward persona (1–5).")
    ap.add_argument("--few_shot", action="store_true",
                    help="Include short persona exemplars in the prompt.")

    # NEW: controlli di budgeting
    ap.add_argument("--safety_margin", type=int, default=256,
                    help="Token margin to leave free in the context window (default 256).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (default 42).")

    args = ap.parse_args()
    set_all_seeds(args.seed)

    trace("splitter.model.loading", "Loading splitter model...",
      base_model=args.base_model, adapter=args.adapter_path,
      max_new_tokens=args.max_new_tokens,
      temperature=args.temperature, top_p=args.top_p, sections=args.sections)

    tok, model = load_model_and_tokenizer(args.base_model, args.adapter_path)

    trace("splitter.model_loaded", "Splitter model loaded.",
        base_model=args.base_model, adapter=args.adapter_path)
    

    # Load inputs
    inputs = _load_inputs_any(args.input_path)

    # For single file inputs, require persona/id from CLI
    if len(inputs) == 1 and inputs[0]["persona"] is None:
        if not args.persona:
            print("Error: --persona is required for .md/.txt input", file=sys.stderr)
            sys.exit(2)
        inputs[0]["persona"] = args.persona
        inputs[0]["id"] = args.id

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    n_ok = 0
    with open(args.output_jsonl, "w", encoding="utf-8") as fout:
        for ex in inputs:
            raw_md = ex["md"]
            persona = ex["persona"] or "General Public"
            pre_title = ex.get("title") or ""
            rid = ex.get("id")

            trace("splitter.item.start", "Starting split for item.",
              persona=persona, provided_title=pre_title)            

            title_md, cleaned_text = clean_pdf_markdown(raw_md)
            title = pre_title or title_md or ""

            trace("splitter.markdown.cleaned", "Markdown cleaned.",
              detected_title=title, paper_chars=len(cleaned_text))

            style_strength = args.style_strength
            if style_strength is None or str(style_strength).strip() == "":
                style_strength = max(1, min(5, int(round(2 + args.temperature * 3))))
                
            trace("splitter.generate.begin", f"Generating {args.sections} section candidates…")
            print(json.dumps({ "type": "splitter_start" }), flush=True)

            sections = generate_sections(
                tok, model, persona, title, cleaned_text,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                n_sections=args.sections,
                style_strength=style_strength,
                include_few_shot=bool(args.few_shot),
                truncation_margin=int(args.safety_margin),
                stderr_log=True
            )

            sec_titles = [ (it.get("title") or "").strip() for it in sections ]
            trace("splitter.generate.done", f"Detected {len(sections)} section candidates.",
                  titles=sec_titles)
            print(json.dumps({ "type": "splitter_done" }), flush=True)
            if sec_titles:
                # primo titolo leggibile
                trace("splitter.section.detected", f"Detected first section '{sec_titles[0]}'…", first=sec_titles[0])


            rec_out = {
                "id": rid,
                "persona": persona,
                "paper_title": title,
                "sections": sections,
                "num_sections": len(sections),
                "temperature": args.temperature,
                "cleaned_text": cleaned_text,
                # extra metadata per audit/debug
                "n_sections_target": args.sections,
                "style_strength": style_strength,
                "few_shot": bool(args.few_shot),
                "safety_margin": int(args.safety_margin),
                "seed": int(args.seed),
            }
            fout.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
            n_ok += 1

    print(f"[✓] Saved -> {args.output_jsonl} ({n_ok} records)")
    trace("splitter.output.saved", "Splitter output saved.", path=args.output_jsonl, records=n_ok)
    

if __name__ == "__main__":
    main()
