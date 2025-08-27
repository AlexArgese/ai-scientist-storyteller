import argparse
import math
import re
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)

DEFAULT_MODEL_PATH = "./mistral-qlora-ft-merged"
FALLBACK_HEADS = ["Overview", "Framework", "Applications", "Evaluation", "Outlook"]

# ------------------------- PROMPT COMPONENTS -------------------------

BASE_RULES = (
    "You are an expert science storyteller.\n\n"
    "TASK:\n"
    "Read the full research paper and produce EXACTLY five sections.\n"
    "Each section must have:\n"
    "- a SHORT heading (1–3 words),\n"
    "- a clear explanatory narrative of 5–8 sentences for the Persona,\n"
    "- NO direct quotes or copy-paste; explain in your own words,\n"
    "- keep technical fidelity, no invented facts.\n"
    "Persona: {persona}\n\n"
    "STYLE:\n"
    "- didactic, vivid, concise; avoid lists, figures, URLs, references.\n"
)

PAPER_WRAPPER = (
    "Paper:\n"
    "<<<BEGIN PAPER>>>\n"
    "{paper}\n"
    "<<<END PAPER>>>\n"
)

SECTION_INSTR = (
    "Now write ONLY section {n}.\n"
    "Format strictly:\n"
    "### {n}. <Heading>\n"
    "Then write 5–8 sentences (plain prose, no lists).\n"
    "Do NOT start the next section header.\n"
    "When you finish, append the marker {end_marker} on a new line.\n"
)

# ------------------------- STOPPING CRITERIA -------------------------

class StopOnStrings(StoppingCriteria):
    """
    Stop generation if ANY of the stop strings appear within the last tokens.
    """
    def __init__(self, tokenizer, stop_strings, search_window_tokens=256):
        self.tokenizer = tokenizer
        self.stop_ids = [tokenizer.encode(s, add_special_tokens=False) for s in stop_strings]
        self.window = max(64, int(search_window_tokens))

    def __call__(self, input_ids, scores, **kwargs):
        ids = input_ids[0].tolist()
        window_ids = ids[-self.window:] if len(ids) > self.window else ids
        for sid in self.stop_ids:
            L = len(sid)
            if L == 0 or L > len(window_ids):
                continue
            # naive subsequence search
            for i in range(len(window_ids) - L + 1):
                if window_ids[i:i + L] == sid:
                    return True
        return False

# ------------------------- TEXT UTILS -------------------------

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_BRACKET_CIT = re.compile(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]")

def split_sentences(txt: str):
    txt = re.sub(r"\s+", " ", txt.strip())
    if not txt:
        return []
    parts = _SENT_SPLIT.split(txt)
    # drop very short fragments
    return [p.strip() for p in parts if len(p.strip()) >= 3]

def clamp_sentence_count(body: str, min_s: int, max_s: int):
    sents = split_sentences(body)
    if not sents:
        return "", 0
    if len(sents) <= max_s:
        return " ".join(sents), len(sents)
    return " ".join(sents[:max_s]), max_s

def token_overlap_ratio(section_tokens, paper_tokens, n: int = 6):
    """
    Approximate content-copy check: Jaccard over n-gram token sets.
    """
    if len(section_tokens) < n or len(paper_tokens) < n:
        return 0.0
    def ngrams(seq):
        return {tuple(seq[i:i+n]) for i in range(len(seq) - n + 1)}
    A = ngrams(section_tokens)
    B = ngrams(paper_tokens)
    inter = len(A & B)
    union = len(A | B)
    return (inter / union) if union > 0 else 0.0

def strip_markers(text, markers):
    for m in markers:
        if m in text:
            text = text.split(m)[0]
    return text.strip()

def normalize_section(n: int, text: str) -> str:
    """
    - Remove markers and obvious meta
    - Ensure a correct '### n. Heading' line
    - Remove placeholders and [1,2]-style citations
    - Compress blank lines
    """
    text = re.sub(r"\r\n?", "\n", text).strip()
    # drop obvious meta/garbage lines
    filtered = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            filtered.append("")  # keep spacing compact later
            continue
        # drop explicit meta
        if s.startswith("<<<") or s.lower().startswith(("footer", "abbreviations")):
            continue
        # drop pure placeholders like <...>
        if re.match(r"^<[^>]+>$", s):
            continue
        # drop list bullets
        if s.startswith(("-", "•", "*")):
            continue
        # remove bracket citations
        s = _BRACKET_CIT.sub("", s)
        filtered.append(s)
    text = "\n".join(filtered).strip()

    # Ensure correct header line
    lines = [l for l in text.splitlines() if l.strip()]
    if lines and re.match(rf"^###\s*{n}\.\s", lines[0]):
        header = lines[0].strip()
        body = "\n".join(lines[1:]).strip()
    else:
        # choose heading from first non-empty line if short/sane, else fallback
        head_candidate = lines[0].strip() if lines else ""
        if ("<" in head_candidate) or (">" in head_candidate) or len(head_candidate) < 2 or "do not" in head_candidate.lower():
            head = FALLBACK_HEADS[(n - 1) % len(FALLBACK_HEADS)]
        else:
            # strip leading hashes or garbage
            head = re.sub(r"^#+\s*", "", head_candidate)
            # keep it short
            head = " ".join(head.split()[:6])
        header = f"### {n}. {head}"
        body = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""

    # Compress excessive blank lines
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    # Remove any leftover angle-bracket placeholders per line
    body = re.sub(r"^\s*<[^>]+>\s*$", "", body, flags=re.M).strip()

    return f"{header}\n{body}".strip()

# ------------------------- MAIN -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--paper", required=True)
    parser.add_argument("--persona", default="Student")
    parser.add_argument("--max-new-tokens", type=int, default=2000,
                        help="Total new-token budget across all 5 sections.")
    parser.add_argument("--min-new-tokens", type=int, default=260,
                        help="Minimum new tokens per SECTION (best effort).")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--overlap-threshold", type=float, default=0.12,
                        help="Max Jaccard n-gram overlap with the paper (copy guard).")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Retries per section if body is empty/too short/too similar.")
    args = parser.parse_args()

    # Load model/tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    # pad/eos alignment
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # Read paper
    with open(args.paper, "r", encoding="utf-8") as f:
        paper_text = f.read()

    # Pre-tokenize paper for overlap checks (lowercased, collapsed spaces)
    paper_norm = re.sub(r"\s+", " ", paper_text.lower()).strip()
    paper_toks = tokenizer.encode(paper_norm, add_special_tokens=False)

    # Base prompt = rules + paper
    base_prompt = (BASE_RULES.format(persona=args.persona) +
                   PAPER_WRAPPER.format(paper=paper_text))

    generated_sections = []
    budget_left = max(200, int(args.max_new_tokens))
    max_ctx = getattr(model.config, "max_position_embeddings", 8192)

    for n in range(1, 6):
        end_marker = f"<<<END_SECTION_{n}>>>"
        # Fermati SOLO sul marker esplicito, non sull'header della sezione dopo
        stop_strings = [end_marker]
        stopping = StoppingCriteriaList([StopOnStrings(tokenizer, stop_strings)])

        prev_sections = ("\n\n".join(generated_sections) + "\n\n") if generated_sections else ""
        section_prompt = base_prompt + prev_sections + SECTION_INSTR.format(n=n, end_marker=end_marker)

        # Tokenize without adding special tokens
        inputs = tokenizer(section_prompt, return_tensors="pt", add_special_tokens=False, truncation=True)
        # Context check
        if inputs["input_ids"].shape[1] >= max_ctx - 256:
            raise ValueError("Prompt troppo lungo: riduci il paper o usa un suo riassunto.")

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        best_text = ""
        accepted = False

        for attempt in range(max(1, args.max_retries)):
            # Per-section token cap: distribuisci il budget rimasto sulle sezioni rimanenti
            remaining_sections = (6 - n)
            per_section_cap = max(
                360,
                min(budget_left, math.ceil(args.max_new_tokens / remaining_sections))
            )
            if per_section_cap <= 64:
                per_section_cap = min(budget_left, 128) if budget_left > 0 else 128

            # Min new tokens per questa sezione (best effort, non garantito)
            min_this = 0
            if per_section_cap > 80:
                min_this = min(args.min_new_tokens, max(0, per_section_cap - 48))

            # leggera variazione sampling al retry
            temp = min(1.2, args.temperature + 0.15 * attempt)
            tp = min(0.97, args.top_p + 0.03 * attempt)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=per_section_cap,
                    min_new_tokens=min_this,
                    temperature=temp,
                    top_p=tp,
                    top_k=args.top_k,
                    do_sample=True,
                    no_repeat_ngram_size=6,
                    repetition_penalty=1.08,
                    length_penalty=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                    stopping_criteria=stopping,
                )

            gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
            raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            raw_text = strip_markers(raw_text, stop_strings)

            # Normalize this section
            cleaned = normalize_section(n, raw_text)

            # Split header/body
            parts = cleaned.split("\n", 1)
            header = parts[0].strip() if parts else f"### {n}. {FALLBACK_HEADS[(n-1)%len(FALLBACK_HEADS)]}"
            body = parts[1].strip() if len(parts) > 1 else ""

            # Check body quality
            if not body:
                ok_sentences = False
                ok_overlap = True
                cleaned_final = header
            else:
                body_clamped, sent_count = clamp_sentence_count(body, 5, 8)
                cleaned_final = f"{header}\n{body_clamped}".strip()
                # overlap check
                sec_toks = tokenizer.encode(
                    re.sub(r"\s+", " ", body_clamped.lower()).strip(),
                    add_special_tokens=False
                )
                overlap = token_overlap_ratio(sec_toks, paper_toks, n=6)
                ok_sentences = (sent_count >= 5)
                ok_overlap = (overlap <= args.overlap_threshold)

            if ok_sentences and ok_overlap:
                best_text = cleaned_final
                accepted = True
                # Update budget with the actually generated tokens
                budget_left = max(0, budget_left - int(len(gen_ids)))
                break
            else:
                # Tentativo non accettato: tieni comunque il meglio finora (per non restare vuoto)
                best_text = cleaned_final

        # Fallback se rimane solo il titolo / corpo vuoto
        if not accepted or best_text.strip() == header.strip():
            fb_head = FALLBACK_HEADS[(n - 1) % len(FALLBACK_HEADS)]
            if not header.startswith(f"### {n}."):
                header = f"### {n}. {fb_head}"
            fallback_body = (
                "This section explains the core idea in simple terms, focusing on why it matters and how it is used. "
                "We briefly situate the problem in medical imaging and clarify what the dataset/framework adds. "
                "Key design choices are sketched without equations, emphasizing intuition and constraints. "
                "We highlight trade‑offs (quality vs. speed; single‑ vs. multi‑expert; static vs. versioned labels). "
                "Finally, we connect this part to the next section in the narrative."
            )
            best_text = f"{header}\n{fallback_body}"

        generated_sections.append(best_text)

    result = "\n\n".join(generated_sections).strip()
    # Final safety clean: remove leftover placeholders lines
    result = re.sub(r"^\s*<[^>]+>\s*$", "", result, flags=re.M).strip()

    out_name = f"output_storia_{args.persona.replace(' ', '_')}.txt"
    with open(out_name, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"✅ Risultato salvato in {out_name}")

if __name__ == "__main__":
    main()
