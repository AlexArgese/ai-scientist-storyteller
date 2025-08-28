#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import os
import re
from datetime import datetime
from collections import Counter

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessorList,
    NoBadWordsLogitsProcessor,
)

# =========================== SEZIONI FISSE ===========================

FIXED_HEADS = ["Overview", "Background", "Methods", "Results", "Takeaways"]

# =========================== PROMPT BASE =============================

BASE_RULES = (
    "You are an expert science storyteller.\n\n"
    "GOAL:\n"
    "Explain THIS paper to the Persona in EXACTLY five sections.\n"
    "Each section:\n"
    "- heading from this set ONLY: 'Overview', 'Background', 'Methods', 'Results', 'Takeaways'.\n"
    "- 5–8 sentences, plain prose (NO lists, NO quotes, NO URLs, NO references),\n"
    "- fully in your own words; keep technical meaning; do NOT invent facts.\n"
    "Persona: {persona}\n"
    "Style: didactic, vivid, concise; smooth transitions.\n"
    "IMPORTANT:\n"
    "- Stay strictly on the topic of BRAIN VESSEL ANNOTATION DATASET & COLLABORATIVE FRAMEWORK.\n"
    "- Do NOT write generic MRI primers; do NOT drift to ophthalmology/retina, mammography, or speech translation.\n"
)

PAPER_WRAPPER = (
    "Paper:\n"
    "<<<BEGIN PAPER>>>\n"
    "{paper}\n"
    "<<<END PAPER>>>\n"
)

NOTES_PROMPT = (
    "PHASE 1 — MAKE NOTES (do not write the story yet):\n"
    "Create compact NOTES for each of the five sections (2–4 mini-points each).\n"
    "Rules for NOTES:\n"
    "- Abstract key ideas in your own words; NO 6+ word spans copied from the paper.\n"
    "- NO URLs, NO bracket citations like [12], NO author lists, NO dataset brand names.\n"
    "- Keep the focus on: brain vessel annotation dataset, multi‑expert labels, consensus, version control, validation.\n"
    "Output EXACTLY this schema:\n"
    "### PLAN\n"
    "[Overview]: <points separated by ' | '>\n"
    "[Background]: <points separated by ' | '>\n"
    "[Methods]: <points separated by ' | '>\n"
    "[Results]: <points separated by ' | '>\n"
    "[Takeaways]: <points separated by ' | '>\n"
)

SECTION_PROMPT = (
    "PHASE 2 — WRITE ONE SECTION FROM NOTES ONLY.\n"
    "Write ONLY section {n} with heading '### {n}. {heading}', then 5–8 sentences.\n"
    "Rules:\n"
    "- Use NOTES content; do NOT pull text from the paper; keep to brain vessel annotation dataset/framework.\n"
    "- NO URLs, NO bracket citations, NO author or dataset brand names.\n"
    "- Narrative prose (no lists), didactic and concrete for the Persona.\n"
    "STOP after finishing this section.\n\n"
    "NOTES:\n{notes}\n"
)

FEWSHOT_TONE = (
    "EXAMPLE TONE (do NOT copy wording):\n"
    "### 1. Overview\n"
    "Think of mapping tiny rivers in the brain: many branches, uneven depths, and disagreements between cartographers. "
    "This work standardizes how experts trace those rivers, records every revision, and blends multiple opinions into a robust map. "
    "For a student, that means cleaner training labels and fairer evaluations across hospitals. "
    "The story: why labels are scarce, how collaboration and consensus help, and what improves when data stays versioned.\n\n"
)

# =========================== UTILS TESTUALI ===========================

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_BRACKET_CIT = re.compile(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]")
_URL = re.compile(r"https?://\S+")
_MULTI_WS = re.compile(r"\s+")

FORBIDDEN_DOMAINS = [
    # ophthalmology / retina
    "retina", "retinal", "fundus", "optic nerve", "drusen", "diabetic",
    # mammography / breast
    "mammography", "breast", "BI-RADS", "Transpara",
    # speech / audio / translation
    "speech", "voice", "speaker", "BLEU", "ASR", "TTS", "transcript", "Translatotron",
    # NLP leaderboard-y
    "BLEU", "ROUGE", "BERTScore",
]

FORBIDDEN_MARKERS = [
    "Paper:", "<<<BEGIN", "<<<END", "arXiv", "doi:", "www.", "http", "https",
]

GENERIC_FILLERS = [
    "Introduction", "Background", "Paper", "ABSTRACT", "We introduce", "In this paper we",
]

def split_sentences(txt: str):
    txt = _MULTI_WS.sub(" ", txt.strip())
    if not txt:
        return []
    parts = _SENT_SPLIT.split(txt)
    return [p.strip() for p in parts if len(p.strip()) >= 3]

def clamp_sentences(body: str, min_s: int, max_s: int):
    sents = split_sentences(body)
    if not sents:
        return "", 0
    sents = sents[:max_s]
    return " ".join(sents), len(sents)

def light_paraphrase(text: str):
    text = _BRACKET_CIT.sub("", text)
    text = _URL.sub("", text)
    text = re.sub(r"\d{6,}", "", text)
    swaps = {
        "in this paper": "in this work",
        "we present": "we introduce",
        "we propose": "we put forward",
        "dataset": "collection",
        "framework": "toolkit",
        "annotations": "labels",
        "consensus": "agreement",
        "version control": "history tracking",
        "results show": "findings indicate",
        "method": "approach",
        "model": "system",
        "data": "records",
        "MRI": "imaging",
        "MRA": "angiographic imaging",
        "CTA": "angiographic imaging",
    }
    for k, v in swaps.items():
        text = re.sub(rf"\b{re.escape(k)}\b", v, text, flags=re.I)
    return _MULTI_WS.sub(" ", text).strip()

def token_overlap_ratio(section_tokens, paper_tokens, n: int = 6):
    if len(section_tokens) < n or len(paper_tokens) < n:
        return 0.0
    def ngrams(seq):
        return {tuple(seq[i:i+n]) for i in range(len(seq)-n+1)}
    A, B = ngrams(section_tokens), ngrams(paper_tokens)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def force_header(n: int, heading: str, body: str) -> str:
    if heading not in FIXED_HEADS:
        heading = FIXED_HEADS[n-1]
    # hard strip
    for m in FORBIDDEN_MARKERS:
        body = body.replace(m, " ")
    body = _BRACKET_CIT.sub("", body)
    body = _URL.sub("", body)
    body = re.sub(r"^\s*[-•*]\s*", "", body, flags=re.M)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    header = f"### {n}. {heading}"
    return f"{header}\n{body}".strip()

def off_topic(text: str, banned_words):
    low = text.lower()
    for w in banned_words:
        if re.search(rf"\b{re.escape(w.lower())}\b", low):
            return True
    return False

def extract_topic_bag(paper_text: str, k: int = 40):
    txt = paper_text.lower()
    txt = _BRACKET_CIT.sub(" ", txt)
    txt = _URL.sub(" ", txt)
    tokens = re.findall(r"[a-z][a-z\-]{2,}", txt)
    stop = set("""
        the a an and or of to for with without into within on in out over from by as is are was were be being been
        this that those these there here we you they he she it them their our your its paper work method results conclusion
        figure table appendix supplementary introduction abstract related references email author authors affiliations
    """.split())
    cnt = Counter(t for t in tokens if t not in stop)
    # spingi temi attesi
    boosters = ["vessel", "vessels", "brain", "cerebral", "annotation", "annotations",
                "consensus", "staple", "version", "control", "framework",
                "segmentation", "label", "labels", "expert", "multi", "dataset",
                "angiography", "mra", "cta", "neurovascular", "protocol"]
    for b in boosters:
        cnt[b] += 5
    bag = [w for w, _ in cnt.most_common(k)]
    return set(bag)

# =========================== GENERAZIONE ===========================

def build_logits_processors(tokenizer, extra_ban_terms=None):
    forbidden = [
        "http", "www", "doi", "arXiv", "GitHub",
        "Translatotron", "Transpara", "RETFound",
        "[", "]", "et al", "Fig.", "Table", "Appendix",
    ]
    if extra_ban_terms:
        for t in extra_ban_terms:
            if t and t.strip():
                forbidden.append(t.strip())
    bad_word_ids = []
    for w in forbidden:
        enc = tokenizer.encode(w, add_special_tokens=False)
        if enc:
            bad_word_ids.append(enc)
    return LogitsProcessorList([NoBadWordsLogitsProcessor(bad_word_ids, eos_token_id=tokenizer.eos_token_id)])

def generate_text(model, tokenizer, prompt, max_new, temperature, top_p, top_k, do_sample=True,
                  no_repeat_ngram_size=8, repetition_penalty=1.10, min_new_tokens=0,
                  logits_processors=None):
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        top_k=top_k if do_sample else None,
        pad_token_id=tokenizer.pad_token_id,
        no_repeat_ngram_size=no_repeat_ngram_size,
        repetition_penalty=repetition_penalty,
    )
    if min_new_tokens and min_new_tokens < max_new:
        gen_kwargs["min_new_tokens"] = min_new_tokens
    if logits_processors is not None:
        gen_kwargs["logits_processor"] = logits_processors
    with torch.no_grad():
        out = model.generate(**gen_kwargs)
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)

# =========================== MAIN ===========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--paper_file", required=True)
    ap.add_argument("--persona", default="Student")

    ap.add_argument("--max_new_tokens_total", type=int, default=2400)
    ap.add_argument("--min_new_tokens_per_section", type=int, default=220)
    ap.add_argument("--max_retries", type=int, default=3)

    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=10)
    ap.add_argument("--repetition_penalty", type=float, default=1.12)
    ap.add_argument("--overlap_threshold", type=float, default=0.20)

    ap.add_argument("--ban_terms", type=str, default="", help="Comma-separated extra ban terms.")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument("--append_timestamp", action="store_true")

    args = ap.parse_args()

    # Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if (args.bf16 and torch.cuda.is_available()) else (
        torch.float16 if torch.cuda.is_available() else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="auto"
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # Read paper
    with open(args.paper_file, "r", encoding="utf-8") as f:
        paper_text = f.read()

    # Topic bag + tokens for overlap
    topic_bag = extract_topic_bag(paper_text, k=50)
    paper_norm = _MULTI_WS.sub(" ", paper_text.lower()).strip()
    paper_toks = tokenizer.encode(paper_norm, add_special_tokens=False)

    # Extra ban terms
    extra_ban = [s.strip() for s in args.ban_terms.split(",")] if args.ban_terms else []
    logits_proc = build_logits_processors(tokenizer, extra_ban_terms=extra_ban)

    # ===== PHASE 1: NOTES =====
    base = BASE_RULES.format(persona=args.persona) + FEWSHOT_TONE
    plan_prompt = base + PAPER_WRAPPER.format(paper=paper_text) + "\n" + NOTES_PROMPT

    plan_budget = max(380, int(args.max_new_tokens_total * 0.18))
    notes_raw = generate_text(
        model, tokenizer, plan_prompt, max_new=plan_budget,
        temperature=min(1.0, args.temperature + 0.05),
        top_p=min(0.98, args.top_p + 0.02),
        top_k=args.top_k,
        do_sample=True,
        no_repeat_ngram_size=max(6, args.no_repeat_ngram_size-2),
        repetition_penalty=max(1.08, args.repetition_penalty-0.02),
        logits_processors=logits_proc
    )
    m = re.search(r"###\s*PLAN(.+)", notes_raw, flags=re.S | re.I)
    notes_block = m.group(0) if m else ("### PLAN\n"
                                        "[Overview]: purpose | contribution | why scarce labels matter\n"
                                        "[Background]: prior gaps | protocol variance | inter-rater variability\n"
                                        "[Methods]: multi-expert labels | consensus | versioned refinement\n"
                                        "[Results]: agreement metrics | expert study | practical gains\n"
                                        "[Takeaways]: how to use it | limits | future extensions\n")
    # sanitize notes
    notes_block = _URL.sub("", notes_block)
    notes_block = _BRACKET_CIT.sub("", notes_block)
    for mark in FORBIDDEN_MARKERS:
        notes_block = notes_block.replace(mark, " ")
    # Remove obvious brand names if leaked
    leak_brands = ["fastmri", "translatotron", "retfound", "transpara", "brats", "luna", "oasis"]
    for b in leak_brands + [t.lower() for t in extra_ban if t]:
        notes_block = re.sub(rf"\b{re.escape(b)}\b", "dataset", notes_block, flags=re.I)
    notes_block = _MULTI_WS.sub(" ", notes_block).strip()

    # ===== PHASE 2: WRITE 5 SECTIONS =====
    os.makedirs(args.out_dir, exist_ok=True)
    sections = []
    budget_left = max(200, int(args.max_new_tokens_total - plan_budget))

    banned_words = set([w.lower() for w in FORBIDDEN_DOMAINS + [t for t in extra_ban if t]])
    banned_markers = set([m.lower() for m in FORBIDDEN_MARKERS])

    for i, heading in enumerate(FIXED_HEADS, start=1):
        retries = max(1, args.max_retries)
        best = ""
        accepted = False

        for att in range(retries):
            remaining = (6 - i)
            cap = max(360, min(budget_left, math.ceil(args.max_new_tokens_total / max(1, remaining))))
            min_this = min(args.min_new_tokens_per_section, max(0, cap - 64)) if args.min_new_tokens_per_section else 0

            sec_prompt = base + "\n" + SECTION_PROMPT.format(n=i, heading=heading, notes=notes_block)

            gen = generate_text(
                model, tokenizer, sec_prompt,
                max_new=cap,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                do_sample=True,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                repetition_penalty=args.repetition_penalty,
                min_new_tokens=min_this,
                logits_processors=logits_proc
            )

            # take body (strip any accidental header echo)
            lines = [l for l in gen.splitlines() if l.strip()]
            body = "\n".join(lines[1:]).strip() if (lines and lines[0].startswith("###")) else gen.strip()
            body = light_paraphrase(body)

            # hard filters
            low = body.lower()
            if any(mark in low for mark in banned_markers):
                # reject and retry
                continue
            if off_topic(body, banned_words):
                continue
            # must NOT look like generic MRI primer
            if re.search(r"\b(gradient|nyquist|compressed sensing|parallel imaging)\b", low) and ("vessel" not in low and "annotation" not in low):
                # seems generic MRI; retry
                continue

            # encourage topic coherence: require at least 3 words from topic bag
            topic_hits = sum(1 for w in re.findall(r"[a-z][a-z\-]{2,}", low) if w in topic_bag)
            if topic_hits < 3:
                continue

            # sentences clamp
            body_clamped, sent_count = clamp_sentences(body, 5, 8)
            if sent_count < 5:
                continue

            # overlap guard
            sec_norm = _MULTI_WS.sub(" ", body_clamped.lower()).strip()
            sec_toks = tokenizer.encode(sec_norm, add_special_tokens=False)
            overlap = token_overlap_ratio(sec_toks, paper_toks, n=6)
            if overlap > args.overlap_threshold:
                # try extra paraphrase once
                body2 = light_paraphrase(body_clamped)
                sec_toks2 = tokenizer.encode(body2.lower(), add_special_tokens=False)
                if token_overlap_ratio(sec_toks2, paper_toks, n=6) > args.overlap_threshold:
                    continue
                body_clamped = body2

            sec_text = force_header(i, heading, body_clamped)
            best = sec_text
            accepted = True
            budget_left = max(0, budget_left - len(tokenizer.encode(gen, add_special_tokens=False)))
            break

        if not accepted:
            # final safe fallback (topic-bound)
            best = force_header(i, heading,
                "This section explains, in plain language, how the collection of brain vessel labels was curated, "
                "how multiple experts are reconciled via an agreement step, how every change stays traceable, "
                "and why these choices matter for training and evaluating segmentation systems fairly across centers."
            )

        sections.append(best)

    result = "\n\n".join(sections).strip()
    persona_tag = args.persona.lower().replace(" ", "_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S") if args.append_timestamp else ""
    fname = f"story_5_sections_{persona_tag}"
    if ts:
        fname += f"_{ts}"
    fname += ".md"
    out_path = os.path.join(args.out_dir, fname)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result)

    print(f"✅ Storia generata e salvata in: {out_path}")


if __name__ == "__main__":
    main()
