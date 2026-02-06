#!/usr/bin/env python3
# infer_from_splits.py 

import os, re, json, argparse, math, random, sys
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel
import time

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from datetime import datetime

TRACE_LOG_FILE = os.environ.get("TRACE_LOG_FILE")
TRACE_REQ_ID = os.environ.get("TRACE_REQ_ID", "-")

SAVE_ARTIFACTS = os.getenv("SAVE_ARTIFACTS", "0") == "1"
def save_artifact(dirpath: str, name: str, content: str):
    if not SAVE_ARTIFACTS:
        return None
    try:
        os.makedirs(dirpath, exist_ok=True)
        path = os.path.join(dirpath, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content if isinstance(content, str) else str(content))
        trace("artifact.saved", f"Saved {name}", path=path, bytes=os.path.getsize(path))
        return path
    except Exception as e:
        trace("artifact.error", f"Fail saving {name}", error=str(e))
        return None

def _now_iso():
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"

def trace(event: str, message: str = "", **data):
    rec = {"ts": _now_iso(), "req_id": TRACE_REQ_ID, "event": event, "message": message, **data}
    line = json.dumps(rec, ensure_ascii=False)
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

# ===============================
# CONFIG
# ===============================
BASE_MODEL  = os.environ.get("STORY_BASE_MODEL", "Qwen/Qwen2.5-32B-Instruct")
ADAPTER_DIR = "qwen32b_storyteller_lora/final_best"
BF16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8

# ===============================
# SEED & UTILS
# ===============================

def set_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def max_ctx_len(model) -> int:
    return int(getattr(model.config, "max_position_embeddings", 8192))

def decode_clean_line(s: str) -> str:
    s = s.strip()
    s = s.splitlines()[0].strip()
    s = s.strip('"').strip("'").strip("`")
    s = re.sub(r'^(Human|User|Assistant|System)\s*:\s*', '', s, flags=re.I)
    s = re.sub(r'\s{2,}', ' ', s)
    s = re.sub(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+', '', s)
    return s.strip()

# ===============================
# PERSONA GUIDANCE
# ===============================
PERSONA_GUIDE: Dict[str, Dict[str, str]] = {
    "General Public": {
        "expertise": "Low",
        "goal": "Understand what AI is and why it matters.",
        "style": ("Use simple, curiosity-driven language. Avoid jargon and equations. "
                  "Give 1–2 relatable examples or analogies and explain why this matters.")
    },
    "Investor": {
        "expertise": "Low–Medium",
        "goal": "Spot AI trends for business or funding decisions.",
        "style": ("Focus on market potential, differentiation, scalability, and risks. "
                  "Explain technical ideas only when tied to business value.")
    },
    "Student": {
        "expertise": "Medium",
        "goal": "Learn AI fundamentals and expand technical knowledge.",
        "style": ("Use educational tone with short definitions and an intuitive example. "
                  "Highlight motivation, key concepts, and takeaways.")
    },
    "Journalist": {
        "expertise": "Medium",
        "goal": "Report clearly and accurately on AI developments.",
        "style": ("Explain for an informed non-technical audience. "
                  "Emphasize significance, evidence, and societal implications.")
    },
    "Policy Maker": {
        "expertise": "Medium–High",
        "goal": "Assess the social, ethical, and legal implications of AI.",
        "style": ("Prioritize governance, transparency, accountability, risks, and societal impact. "
                  "Avoid deep technical dives unless necessary.")
    },
    "Professor": {
        "expertise": "High",
        "goal": "Teach AI concepts and methods effectively to others.",
        "style": ("Organize clearly: learning objectives, examples, misconceptions and limitations "
                  "to foster critical thinking.")
    },
    "Researchers & Engineers": {
        "expertise": "High",
        "goal": "Produce, implement, and evaluate advanced AI research and systems.",
        "style": ("Be concise, technical, and implementation-aware. Highlight novelty, methodology, datasets, "
                  "metrics, results, engineering tradeoffs, and limitations.")
    },
}

def persona_block(persona: str) -> str:
    spec = PERSONA_GUIDE.get(persona, PERSONA_GUIDE["General Public"])
    return (
        f"TARGET READER PERSONA\n"
        f"- Persona: {persona}\n"
        f"- Expertise level: {spec['expertise']}\n"
        f"- Communication goal: {spec['goal']}\n"
        f"- Writing style to follow: {spec['style']}\n"
        "\n"
        "Very important:\n"
        "- If there is a trade-off, prioritize clarity and usefulness for THIS Persona\n"
        "  over preserving every implementation detail.\n"
    )

# ===============================
# JSON / TEXT HELPERS
# ===============================
def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.I)
    return s.strip()

def extract_first_json_object(s: str) -> str:
    s = _strip_code_fences(s)
    start = s.find("{")
    if start < 0: return s.strip()
    i, depth, in_str, esc = start, 0, False, False
    while i < len(s):
        ch = s[i]
        if in_str:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': in_str = False
        else:
            if ch == '"': in_str = True
            elif ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0: return s[start:i+1]
        i += 1
    return s[start:].strip()

def clean_plain_text(txt: str) -> str:
    t = _strip_code_fences(str(txt))
    t = t.strip().strip('"').strip("'")
    t = re.sub(r'[，。；、]', ' ', t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+', ' ', t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()

def _strip_splitter_markers(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r'<!--\s*\{[^>]*\}\s*-->', '', text)

def _strip_trailing_json_blob(s: str) -> str:
    if not s:
      return s
    m = re.search(r'\{\s*"title"\s*:', s[80:])
    if not m:
        m = re.search(r'\{\s*"text"\s*:', s[80:])
    if m:
        cut_at = 80 + m.start()
        return s[:cut_at].rstrip()
    return s

SENT_STOP_RE = re.compile(r'(?s)(.*?[.!?…](?:["”\']|\))?)\s*(?:$|[A-ZÀ-ÖØ-Ý0-9\n])')
def truncate_to_last_sentence(s: str) -> str:
    s = s.strip()
    if re.search(r'[.!?…]["”\')\]]?\s*$', s):
        return s
    m = list(re.finditer(r'[.!?…](?:["”\')\]]?)(?!.*[.!?…])', s))
    if m:
        return s[:m[-1].end()].strip()
    return s

SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-ZÀ-ÖØ-Ý])')

def dedup_sentences(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return t

    sentences = re.split(r'(?<=[.!?])\s+', t)
    seen = set()
    out = []

    for s in sentences:
        raw = s.strip()
        if not raw:
            continue
        core = re.sub(r'^[\.\-\•\s]+', '', raw).strip()
        if not core:
            continue
        norm = re.sub(r'\s+', ' ', core).lower()
        if norm in seen:
            continue
        seen.add(norm)
        out.append(core)

    return " ".join(out)

def normalize_paragraphs(text: str,
                         target_paras: int = 3,
                         min_sent_per_para: int = 2,
                         max_sent_per_para: int = 4) -> str:
    t = text.strip()
    if not t:
        return t

    sentences = [s.strip() for s in SENT_SPLIT_RE.split(t) if s.strip()]
    if len(sentences) <= min_sent_per_para:
        return " ".join(sentences)

    total = len(sentences)
    paras: List[str] = []
    idx = 0

    avg_sent = max(min_sent_per_para, min(max_sent_per_para, total // target_paras or 1))

    while idx < total:
        end = min(idx + avg_sent, total)
        if total - end < min_sent_per_para and paras:
            last = paras.pop()
            last_sents = last.split("|||")
            new_sents = last_sents + sentences[idx:total]
            paras.append("|||".join(new_sents))
            idx = total
            break
        chunk = sentences[idx:end]
        paras.append("|||".join(chunk))
        idx = end

    normalized = "\n\n".join(" ".join(p.split("|||")).strip() for p in paras if p.strip())
    return normalized.strip()

PARA_BLOCK_RE = re.compile(r"\n{2,}|\r?\n\s*\r?\n")

def split_into_paragraphs_backend(txt: str) -> List[str]:
    if txt is None:
        return []
    s = str(txt).replace("\r\n", "\n").strip()
    if not s:
        return []

    blocks = [b.strip() for b in PARA_BLOCK_RE.split(s) if b.strip()]
    paras: List[str] = []

    for block in blocks:
        if len(block) < 320 or "\n" in block:
            paras.append(block.strip())
            continue

        sentences = [t.strip() for t in SENT_SPLIT_RE.split(block) if t.strip()]

        if len(sentences) <= 1:
            paras.append(block.strip())
            continue

        current = sentences[0]
        for sent in sentences[1:]:
            if len(current) + 1 + len(sent) <= 400:
                current = current + " " + sent
            else:
                paras.append(current.strip())
                current = sent
        if current.strip():
            paras.append(current.strip())

    return paras

def strip_prompt_echo(s: str) -> str:
    s = re.split(r"\bHuman:\b", s, maxsplit=1)[0]
    s = re.split(r"\bUser:\b", s, maxsplit=1)[0]
    s = re.split(r"\bAssistant:\b", s, maxsplit=1)[0]
    s = re.split(r"\bSystem:\b", s, maxsplit=1)[0]
    return s

# ===============================
# LIGHT ANTI-HALLUCINATION
# ===============================
CAP_RE = re.compile(r"\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*)\b")
SAFE_WORDS = {"json","section","introduction","conclusion","authors","framework"}

def cap_entities(text:str):
    return set(CAP_RE.findall(text))

def too_many_new_entities(gen_text:str, ctx:str, max_new_ratio=0.05, min_total=1):
    ctx_cap = cap_entities(ctx)
    out_cap = cap_entities(gen_text)
    if not out_cap: return False
    new_ents = [e for e in out_cap if (e not in ctx_cap and e.lower() not in SAFE_WORDS)]
    return (len(out_cap) >= min_total) and (len(new_ents)/max(1,len(out_cap)) > max_new_ratio)

def strip_unknown_entities(txt, ctx):
    wl = cap_entities(ctx)
    sents = re.split(r'(?<=[.!?])\s+', txt)
    keep = []
    for s in sents:
        ents = cap_entities(s)
        if not ents or all((e in wl) or (e.lower() in SAFE_WORDS) for e in ents):
            keep.append(s)
    return " ".join(keep).strip()

# ===============================
# RETRIEVAL
# ===============================
def segment_text(text:str, max_words:int=180, overlap:int=60) -> List[str]:
    words = (text or "").split()
    if not words: return []
    segs, i = [], 0
    step = max(1, max_words - overlap)
    while i < len(words):
        seg = " ".join(words[i:i+max_words]).strip()
        if seg: segs.append(seg)
        i += step
    return segs

class Retriever:
    def __init__(self, method:str="auto", model_name:str="sentence-transformers/all-MiniLM-L6-v2"):
        self.method_in = method
        self.method = method
        self.model_name = model_name
        self.st_model = None
        self.vec = None
        self.corpus_mat = None
        self.corpus: List[str] = []
        if method in ("auto","emb"):
            try:
                from sentence_transformers import SentenceTransformer
                self.st_model = SentenceTransformer(model_name)
                self.method = "emb"
            except Exception:
                self.method = "tfidf"
        if self.method == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vec = TfidfVectorizer(ngram_range=(1,2), max_features=120000, lowercase=True)

    def fit(self, corpus: List[str]):
        self.corpus = corpus[:]
        if not corpus:
            self.corpus_mat = None
            return
        if self.method == "emb":
            embs = self.st_model.encode(corpus, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
            self.corpus_mat = np.asarray(embs, dtype=np.float32)
        else:
            self.corpus_mat = self.vec.fit_transform(corpus)

    def topk(self, query:str, k:int=5) -> List[Tuple[int,float]]:
        if self.corpus_mat is None or self.corpus_mat.shape[0] == 0:
            return []
        if not self.corpus or self.corpus_mat is None: return []
        if self.method == "emb":
            q = self.st_model.encode([query], normalize_embeddings=True)[0]
            sims = (self.corpus_mat @ q)
            idx = np.argsort(-sims)[:k]
            return [(int(i), float(sims[i])) for i in idx]
        else:
            qv = self.vec.transform([query])
            sims = (qv @ self.corpus_mat.T).toarray().ravel()
            idx = np.argsort(-sims)[:k]
            return [(int(i), float(sims[i])) for i in idx]

def build_section_context(sec_title: str, sec_desc: str, ret: Retriever, k: int = 6, max_chars: int = 2500) -> str:
    query = (sec_title.strip() + " — " + (sec_desc or "")).strip()

    if ret is None or getattr(ret, "corpus_mat", None) is None:
        return ""
    n = getattr(ret.corpus_mat, "shape", (0, 0))[0]
    if n <= 0:
        return ""

    k_eff = min(max(1, int(k)), n)

    hits = ret.topk(query, k=k_eff)
    ctx = ""
    for i, _score in hits:
        frag = ret.corpus[i].strip()
        if not frag:
            continue
        if len(ctx) + len(frag) + 2 > max_chars:
            break
        ctx += frag + "\n\n"
    return ctx.strip()

# ===============================
# PROMPT BUILDERS
# ===============================
BASE_RULES = (
    "- Write ONLY the body of the section.\n"
    "- Do NOT include the section title.\n"
    "- Do NOT use JSON, markdown, bullet points or numbered lists.\n"
    "- Do NOT mention figures, tables, equations, sections, captions, or numbering from the original paper.\n"
    "- If the context contains references to 'Figure', 'Table', or section numbers (e.g., 3.1), ignore them completely.\n"
    "- Rewrite only the meaning, not the formatting or structural references.\n"
    "- Output only 2–4 short paragraphs of plain English text.\n"
    "- Use ONLY entities/terms that appear VERBATIM in the Paper context below.\n"
    "- NEVER invent numbers, hardware brands, dataset names, affiliations, or citations.\n"
    "- If the context does not contain enough details, OMIT them instead of commenting on missing information.\n"
    "- Avoid bullet lists; write smooth narrative prose.\n"
)

def build_section_prompt(
    persona:str, paper_title:str, target_title:str, target_desc:str,
    context_chunk:str, length_preset:str="medium", prev_title:Optional[str]=None
) -> str:
    preset = (length_preset or "medium").lower()
    if preset == "short":
        length_rule = (
            "- Write about 5–7 sentences (~80–120 words) in 1–2 short paragraphs.\n"
        )
    elif preset == "long":
        length_rule = (
            "- Write about 10–14 sentences (~180–250 words) in 3–4 paragraphs.\n"
        )
    else:
        length_rule = (
            "- Write about 7–10 sentences (~120–180 words) in 2–3 paragraphs.\n"
        )

    transition_line = ""
    if prev_title:
        transition_line = (f"- The opening sentence should connect naturally from the previous section "
                           f"('{prev_title}') without repeating it.\n")

    pblock = persona_block(persona)

    prompt = (
        "You are an AI Scientist Storyteller.\n"
        "Your primary objective is to write for the specific Persona described below.\n"
        "Everything you write (tone, level of detail, examples) MUST be adapted to this Persona.\n"
        "\n"
        f"{pblock}\n"
        f"Paper title: {paper_title}\n"
        "\n"
        "Task: Write ONE coherent section of the story, strictly matching the given target title.\n"
        "Rules:\n"
        f"{BASE_RULES}"
        f"{length_rule}"
        f"{transition_line}"
        "- The target section title is fixed; use it as guidance, but do NOT print it.\n"
        "- Paraphrase faithfully instead of copying long passages.\n"
        "- Every explanation, example, and emphasis must be chosen to help THIS Persona reach their Goal above.\n"
        "\nTarget section:\n"
        f"- Title: {target_title}\n"
    )
    if target_desc:
        prompt += f"- Description: {target_desc}\n"
    prompt += (
        "\nPaper context (retrieved, factual grounding):\n"
        f"\"\"\"{context_chunk if context_chunk else '[NO CONTEXT FOUND]'}\"\"\"\n"
        "- If the context is empty, write a VERY brief and generic section.\n"
        "- Use ONLY the section description; do NOT add any details not present.\n"
        "- No invented concepts, datasets, methods, or examples.\n"
    )
    return prompt

def build_title_prompt(persona:str, paper_title:str, outline_titles:List[str], paper_context:str="") -> str:
    avoid = "; ".join(t for t in outline_titles if t)
    pblock = persona_block(persona)
    ctx_block = '"""' + (paper_context or "").replace('"""', '\\"""') + '"""'
    return (
        "You are an AI Scientist Storyteller.\n"
        f"{pblock}\n"
        f"Paper title: {paper_title}\n\n"
        "Task: Propose ONE catchy, persona-tailored story title (<= 10 words).\n"
        "that clearly reflects THIS paper (method, task, dataset, or finding),\n"
        "grounded ONLY in the context below.\n"
        "Constraints:\n"
        "- Do NOT reuse any section title verbatim.\n"
        "- No quotes, no markdown, English only.\n"
        f"- Avoid these exact strings: {avoid}\n"
        "- Prefer including a distinctive keyword from the paper (e.g., model/task/dataset name) if present.\n"
        "\nPaper context (verbatim snippets):\n"
        f"{ctx_block}\n"
        "Return ONLY the title text.\n"
    )

# ===============================
# MODEL LOAD
# ===============================
def load_model_and_tokenizer(adapter_dir: Optional[str]):
    adapter_dir = (adapter_dir or "").strip()
    use_lora = adapter_dir.lower() not in ("", "none", "base")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=(torch.bfloat16 if BF16 else torch.float16),
    )

    cache_dir = os.environ.get("HF_HOME") or None

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, cache_dir=cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=(torch.bfloat16 if BF16 else torch.float16),
        quantization_config=bnb,
        device_map="auto",
        cache_dir=cache_dir,
    )

    if not use_lora:
        trace(
            "story.model.base_only",
            "Using BASE model without LoRA adapter.",
            base_model=BASE_MODEL,
        )
        model = base
    else:
        if not os.path.isdir(adapter_dir):
            raise RuntimeError(f"Adapter dir not found: {adapter_dir}")
        trace(
            "story.model.adapter_load",
            "Loading LoRA adapter.",
            base_model=BASE_MODEL,
            adapter=adapter_dir,
        )
        model = PeftModel.from_pretrained(base, adapter_dir, cache_dir=cache_dir)

    model.eval()
    try:
        model.config.use_cache = True
    except Exception:
        pass
    return tok, model

# ===============================
# GENERATION
# ===============================
@torch.no_grad()
def generate_once(tokenizer, model, prompt: str, gen_cfg: GenerationConfig, safety_margin: int = 128) -> str:
    t0 = time.time()
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    max_pos = int(getattr(model.config, "max_position_embeddings", 8192))
    budget_input = max_pos - int(gen_cfg.max_new_tokens or 0) - safety_margin
    trimmed = False
    if len(ids) > budget_input:
        ids = ids[-budget_input:]
        trimmed = True
    prompt_trim = tokenizer.decode(ids, skip_special_tokens=True)

    trace("gen.budget", "Prompt token budget",
          max_pos=max_pos, safety_margin=safety_margin,
          max_new=gen_cfg.max_new_tokens, in_tokens=len(ids),
          trimmed=trimmed)

    enc = tokenizer(prompt_trim, return_tensors="pt", padding=False, truncation=True, max_length=max_pos - safety_margin)
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        generation_config=gen_cfg,
        use_cache=True,
    )
    gen_ids = out[0][input_ids.shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    trace("gen.done", "Model.generate finished", elapsed_s=round(time.time()-t0,3), out_chars=len(text))
    return text

# ===============================
# MAIN
# ===============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl",  required=True, help="JSONL from splitter (outline + cleaned_text + persona/title)")
    ap.add_argument("--out_jsonl", required=True, help="Output JSONL with the generations")
    ap.add_argument("--adapter",    default=ADAPTER_DIR, help="LoRA adapter dir (final_best)")
    ap.add_argument("--preset", default="medium", help="short|medium|long")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_new_tokens", type=int, default=900)
    ap.add_argument("--per_sec_floor", type=int, default=0, help="min tokens per section")
    ap.add_argument("--retriever", default="auto", help="auto|emb|tfidf")
    ap.add_argument("--retriever_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--k", type=int, default=3, help="top-k paragraphs per section (default 3)")
    ap.add_argument("--max_ctx_chars", type=int, default=1400)
    ap.add_argument("--seg_words", type=int, default=180)
    ap.add_argument("--overlap_words", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_all_seeds(args.seed)

    tok, model = load_model_and_tokenizer(args.adapter)

    trace("story.model_loaded", "Storyteller model loaded.",
          base_model=BASE_MODEL, adapter=args.adapter, preset=args.preset,
          temperature=args.temperature, top_p=args.top_p, max_new_tokens=args.max_new_tokens)
    print("### STORY_VERSION: V3_ANTI_HALLU ###", file=sys.stderr, flush=True)

    ret = Retriever(method=args.retriever, model_name=args.retriever_model)

    temp  = float(args.temperature)
    top_p = float(args.top_p)
    do_sample = temp > 0.0

    base_cfg = dict(
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )

    n_items, ok = 0, 0
    with open(args.in_jsonl, "r", encoding="utf-8") as fin, \
         open(args.out_jsonl, "w", encoding="utf-8") as fout:

        lines = [ln for ln in fin if ln.strip()]
        for ln in tqdm(lines, desc="Inferencing", unit="item"):

            n_items += 1
            item = json.loads(ln)

            persona = item.get("persona") or "General Public"
            paper_title = item.get("paper_title") or "Untitled"
            outline = item.get("sections") or []
            trace("story.item.start", "Starting story generation for item.",
                  persona=persona, paper_title=paper_title,
                  outline_count=len(outline))
            ctx_full = _strip_splitter_markers(item.get("cleaned_text") or "")

            custom_prompt = item.get("prompt")
            if custom_prompt:
                gen_cfg_direct = GenerationConfig(
                    max_new_tokens=384,
                    do_sample=(temp > 0.0),
                    temperature=temp,
                    top_p=top_p,
                    repetition_penalty=(1.10 if temp > 0 else 1.0),
                    **base_cfg,
                )
                gen_full = generate_once(tok, model, custom_prompt, gen_cfg_direct)
                gen_json_text = extract_first_json_object(gen_full)
                if gen_json_text.strip().startswith("{") and not gen_json_text.strip().endswith("}"):
                    gen_json_text = gen_json_text.strip() + "}"
                direct_text = gen_json_text
                try:
                    obj = json.loads(gen_json_text)
                    if isinstance(obj, dict) and isinstance(obj.get("text"), str):
                        direct_text = obj["text"].strip()
                except Exception:
                    pass
                direct_text = clean_plain_text(_strip_trailing_json_blob(direct_text))
                fout.write(json.dumps({
                    "id": item.get("id"),
                    "persona": persona,
                    "generation": {"text": direct_text}
                }, ensure_ascii=False) + "\n")
                continue

            paragraphs = segment_text(ctx_full, max_words=args.seg_words, overlap=args.overlap_words)
            ret.fit(paragraphs)

            SEC_COUNT = 5
            target_n = len(outline) if isinstance(outline, list) and len(outline) > 0 else SEC_COUNT

            raw_per_sec = int(args.max_new_tokens // max(1, target_n))

            if args.preset == "long":
                per_sec_max = int(raw_per_sec * 1.2)
            elif args.preset == "short":
                per_sec_max = int(raw_per_sec * 0.8)
            else:
                per_sec_max = raw_per_sec

            if args.preset == "short":
                per_sec_max = max(80, min(per_sec_max, 140))
            elif args.preset == "medium":
                per_sec_max = max(140, min(per_sec_max, 220))
            elif args.preset == "long":
                per_sec_max = max(220, min(per_sec_max, 380))

            if do_sample:
                gen_cfg = GenerationConfig(
                    max_new_tokens=per_sec_max,
                    do_sample=True,
                    temperature=temp,
                    top_p=top_p,
                    repetition_penalty=1.10,
                    **base_cfg,
                )
            else:
                gen_cfg = GenerationConfig(
                    max_new_tokens=per_sec_max,
                    do_sample=False,
                    **base_cfg,
                )

            sections_out, valid_count = [], 0
            print(json.dumps({ "type": "story_start" }), flush=True)

            for i in range(target_n):
                sec = outline[i] if i < len(outline) else {}
                sec_title = (sec.get("title") or f"Section {i+1}").strip()
                sec_desc  = (sec.get("description") or "").strip()
                prev_title = (
                    outline[i-1]["title"].strip()
                    if i > 0 and isinstance(outline[i-1], dict) and outline[i-1].get("title")
                    else None
                )

                trace("story.retriever.ready", "Retriever ready.",
                      method=ret.method, corpus_segments=len(paragraphs))
                trace("story.section.start", f"Generating story section '{sec_title}'", index=i, title=sec_title)
                print(json.dumps({
                    "type": "story_progress",
                    "index": i,
                    "title": sec_title
                }), flush=True)

                ctx_i = build_section_context(
                    sec_title, sec_desc, ret,
                    k=args.k, max_chars=args.max_ctx_chars
                )
                trace("retrieval.ctx", "Section context built",
                      index=i, title=sec_title, ctx_chars=len(ctx_i), top_k=args.k)
                trace("story.section.retrieval", f"Retrieving context for '{sec_title}'", index=i, title=sec_title)

                if not ctx_i.strip():
                    fallback_ctx = (sec_desc or ctx_full[:1000]).strip()
                    trace("story.section.no_ctx", "No retrieved context, using fallback description/global ctx",
                        index=i, title=sec_title, fallback_len=len(fallback_ctx))
                    ctx_i = fallback_ctx

                prompt = build_section_prompt(
                    persona=persona,
                    paper_title=paper_title,
                    target_title=sec_title,
                    target_desc=sec_desc,
                    context_chunk=ctx_i,
                    length_preset=args.preset,
                    prev_title=prev_title,
                )

                save_artifact(f"/tmp/story_{TRACE_REQ_ID}", f"sec{i:02d}_context.txt", ctx_i)
                save_artifact(f"/tmp/story_{TRACE_REQ_ID}", f"sec{i:02d}_prompt.txt", prompt)

                trace("story.prompt.ready", "Prompt ready",
                      index=i, title=sec_title, prompt_chars=len(prompt))

                gen_full = generate_once(tok, model, prompt, gen_cfg)
                save_artifact(f"/tmp/story_{TRACE_REQ_ID}", f"sec{i:02d}_raw.txt", gen_full)
                trace("story.gen.raw", "Raw generation captured",
                      index=i, chars=len(gen_full))

                raw_text = strip_prompt_echo(gen_full)

                if too_many_new_entities(raw_text, ctx_i, max_new_ratio=0.15, min_total=3):
                    trace("story.gen.entities_filtered", "Filtered unknown entities", index=i)
                    raw_text = strip_unknown_entities(raw_text, ctx_i)

                clean = clean_plain_text(raw_text)
                clean = _strip_trailing_json_blob(clean)
                clean = re.sub(r'[，。；、]+', '.', clean)
                clean = re.sub(r'\.\.+', '.', clean)
                clean = dedup_sentences(clean)
                clean = truncate_to_last_sentence(clean)
                clean = _strip_splitter_markers(clean)

                if args.preset == "short":
                    clean = normalize_paragraphs(clean, target_paras=1, min_sent_per_para=1, max_sent_per_para=3)
                elif args.preset == "medium":
                    clean = normalize_paragraphs(clean, target_paras=2, min_sent_per_para=2, max_sent_per_para=4)
                else:
                    clean = normalize_paragraphs(clean, target_paras=4, min_sent_per_para=3, max_sent_per_para=5)

                if not clean.strip():
                    fallback_source = (sec_desc or ctx_i[:600]).strip()

                    fallback_prompt = (
                        f"You are an AI Scientist Storyteller writing for the following Persona: {persona}.\n"
                        "Rewrite the following text into a smooth, well-structured narrative paragraph adapted to this Persona.\n"
                        "Do NOT include markdown, bullets, or section titles.\n"
                        "Keep the original meaning but improve clarity and flow.\n"
                        "Output only the rewritten paragraph.\n\n"
                        f"Text to rewrite:\n\"\"\"{fallback_source}\"\"\""
                    )

                    gen_cfg_fallback = GenerationConfig(
                        max_new_tokens=180,
                        do_sample=(temp > 0),
                        temperature=temp,
                        top_p=top_p,
                        repetition_penalty=1.05,
                        pad_token_id=tok.pad_token_id,
                        eos_token_id=tok.eos_token_id,
                    )

                    try:
                        fb_raw = generate_once(tok, model, fallback_prompt, gen_cfg_fallback)
                        fb_clean = clean_plain_text(strip_prompt_echo(fb_raw))
                        fb_clean = truncate_to_last_sentence(fb_clean)
                        fb_clean = _strip_splitter_markers(fb_clean)

                        if fb_clean.strip():
                            clean = fb_clean
                        else:
                            clean = fallback_source
                    except Exception:
                        clean = fallback_source

                if args.preset == "short":
                    min_words = 40
                elif args.preset == "medium":
                    min_words = 80
                else:
                    min_words = 150

                words = clean.split()

                if len(words) < min_words and ctx_i:
                    expand_prompt = (
                        prompt
                        + "\n\nThe previous attempt was too short. Now rewrite this section with more detail "
                        "and concrete explanations. Write about 5 more sentences.\n"
                    )
                    gen_full2 = generate_once(tok, model, expand_prompt, gen_cfg)
                    raw_text2 = strip_prompt_echo(gen_full2)
                    clean2 = clean_plain_text(raw_text2)
                    clean2 = _strip_trailing_json_blob(clean2)
                    clean2 = re.sub(r'[，。；、]+', '.', clean2)
                    clean2 = re.sub(r'\.\.+', '.', clean2)
                    clean2 = dedup_sentences(clean2)
                    clean2 = truncate_to_last_sentence(clean2)
                    clean2 = _strip_splitter_markers(clean2)

                    if len(clean2.split()) > len(words):
                        clean = clean2

                paras = split_into_paragraphs_backend(clean)
                if not paras:
                    paras = [clean.strip()]

                out_sec = {
                    "title": sec_title,
                    "text": clean.strip(),
                    "narrative": clean.strip(),
                    "paragraphs": paras,
                }

                save_artifact(f"/tmp/story_{TRACE_REQ_ID}", f"sec{i:02d}_final.txt", out_sec["text"])

                if out_sec["text"].strip():
                    valid_count += 1
                sections_out.append(out_sec)

                trace("story.section.done", f"Section '{sec_title}' generated.",
                      index=i, text_chars=len(out_sec["text"]))

            print(json.dumps({ "type": "story_done" }), flush=True)

            outline_titles = [ (sec.get("title") or "").strip() for sec in outline ]

            title_ctx = build_section_context(
                "Abstract",
                "main contribution, method, task, dataset names",
                ret, k=3, max_chars=600
            ) or ctx_full[:600]

            title_prompt = build_title_prompt(persona, paper_title, outline_titles, paper_context=title_ctx)
            title_cfg = GenerationConfig(max_new_tokens=40, do_sample=False, **base_cfg)
            title_raw = generate_once(tok, model, title_prompt, title_cfg)
            title_line = decode_clean_line(title_raw)
            if outline_titles and title_line.lower() == outline_titles[0].lower():
                title_line = f"{title_line}: An Overview"
            out_obj = {"title": title_line or None, "sections": sections_out}
            save_artifact(f"/tmp/story_{TRACE_REQ_ID}", "title_prompt.txt", title_prompt)
            save_artifact(f"/tmp/story_{TRACE_REQ_ID}", "title_raw.txt", title_raw)
            trace("story.title.raw", "Title raw generated", chars=len(title_raw))

            trace("story.title.generated", f"Story title proposed: '{title_line}'", title=title_line)

            rec = {
                "id": item.get("id"),
                "persona": persona,
                "paper_title": paper_title,
                "outline": outline,
                "paper_context_len": len(ctx_full),
                "generation": out_obj,
                "story_title": None
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            trace("story.item.done", "Story item completed.",
                  sections=len(sections_out))

            if valid_count >= max(1, target_n - 1):
                ok += 1

    print(f"[DONE] items={n_items} ok_ratio={ok/max(1,n_items):.3f}")
    trace("story.output.saved", "Storyteller output saved.", path=args.out_jsonl, items=n_items)
    trace("story.session.done", "Storyteller session completed",
      save_artifacts=SAVE_ARTIFACTS)

if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
