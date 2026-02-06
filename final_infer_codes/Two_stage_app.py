# FILE: two_stage_app.py — Two-stage orchestrator (SPLITTER → STORYTELLER)

import os, re, json, uuid, time, tempfile, subprocess, shutil, sys, fcntl
from typing import Optional, Dict, Any, List, Tuple, Literal
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import requests
import threading
from datetime import datetime

LOG_ROOT = os.getenv("SCI_LOG_ROOT", "/tmp/sci_logs")
os.makedirs(LOG_ROOT, exist_ok=True)

DEBUG_VERBOSE = os.getenv("DEBUG_VERBOSE", "0") == "1"
SAVE_ARTIFACTS = os.getenv("SAVE_ARTIFACTS", "0") == "1"

from fastapi import UploadFile, File, Form
from fastapi.staticfiles import StaticFiles

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROGRESS: dict[str, list[dict]] = {}
PROGRESS_LOCK = threading.Lock()


def _buffer_progress(progress_id: str | None, event: dict):
    if not progress_id:
        return
    try:
        with PROGRESS_LOCK:
            PROGRESS.setdefault(progress_id, []).append(event)
    except Exception:
        pass


def _dbg(trace, event: str, message: str = "", **data):
    if DEBUG_VERBOSE and trace:
        trace(event, message, **data)


def _ts():
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


def _trace_open(req_id: str):
    path = os.path.join(LOG_ROOT, f"trace_{req_id}.ndjson")

    def _write(event: str, message: str = "", **data):
        rec = {"ts": _ts(), "req_id": req_id, "event": event, "message": message, **data}
        line = json.dumps(rec, ensure_ascii=False)
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass
        try:
            print(f"[trace] {event} {message} :: {json.dumps(data, ensure_ascii=False)}", file=sys.stderr, flush=True)
        except Exception:
            pass

    return path, _write


def _save_artifact(base_dir: str, name: str, content: str, trace=None):
    if not SAVE_ARTIFACTS:
        return None
    try:
        os.makedirs(base_dir, exist_ok=True)
        path = os.path.join(base_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content if isinstance(content, str) else str(content))
        if trace:
            trace("artifact.saved", f"Saved artifact {name}", path=path, bytes=os.path.getsize(path))
        return path
    except Exception as e:
        if trace:
            trace("artifact.error", f"Failed saving artifact {name}", error=str(e))
        return None


PY = sys.executable

# =========================
# ENV config
# =========================
API_KEY = os.getenv("API_KEY", "").strip()

SPLITTER_SCRIPT = os.getenv("SPLITTER_SCRIPT", "/docker/argese/clean_dataset/splitting/infer_splitter.py")
SPLITTER_BASE_MODEL = os.getenv("SPLITTER_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
SPLITTER_ADAPTER_PATH = os.getenv("SPLITTER_ADAPTER_PATH", "out-splitter-qwen7b/checkpoint-100")
SPLITTER_MAX_NEW = int(os.getenv("SPLITTER_MAX_NEW", "768"))

STORYTELLER_SCRIPT = os.getenv("STORYTELLER_SCRIPT", "/docker/argese/clean_dataset/storytelling/infer_from_splits.py")
STORYTELLER_ADAPTER = os.getenv("STORYTELLER_ADAPTER", "../qwen32b_storyteller_lora/final_best")

TIMEOUT_SPLITTER = int(os.getenv("TIMEOUT_SPLITTER", "900"))
TIMEOUT_STORY = int(os.getenv("TIMEOUT_STORY", "1800"))

# =========================
# FastAPI
# =========================
app = FastAPI(title="Two-Stage Story Orchestrator", version="1.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/app/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.mount("/papers", StaticFiles(directory=UPLOAD_DIR), name="papers")


@app.get("/health")
def health():
    return {"ok": True, "stage": "split+story", "v": "1.1.0"}


# =========================
# GPU picker + locking
# =========================
def pick_best_gpu(min_free_gb: float = 6.0) -> Optional[Dict[str, Any]]:
    try:
        import pynvml
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        best = None
        for i in range(n):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
            free_gb = mem.free / (1024**3)
            if free_gb >= min_free_gb:
                score = free_gb - (util / 100.0) * 2.0
                if best is None or score > best[0]:
                    best = (score, i, free_gb, util)
        if best:
            return {"index": best[1], "free_gb": round(best[2], 1), "util": int(best[3])}
    except Exception:
        pass

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True
        )
        best = None
        for line in out.strip().splitlines():
            idx_s, free_s, util_s = [x.strip() for x in line.split(",")]
            idx, free_gb, util = int(idx_s), float(free_s) / 1024.0, int(util_s)
            if free_gb >= min_free_gb:
                score = free_gb - (util / 100.0) * 2.0
                if best is None or score > best[0]:
                    best = (score, idx, free_gb, util)
        if best:
            return {"index": best[1], "free_gb": round(best[2], 1), "util": best[3]}
    except Exception:
        pass

    return None


class GPULock:
    def __init__(self, idx: int, timeout_s: int = 10):
        self.idx = idx
        self.timeout_s = timeout_s
        self.fd = None
        self.path = f"/tmp/gpu_{idx}.lock"

    def __enter__(self):
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        start = time.time()
        while True:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                os.ftruncate(self.fd, 0)
                os.write(self.fd, str(os.getpid()).encode())
                return self
            except BlockingIOError:
                if time.time() - start > self.timeout_s:
                    raise HTTPException(503, f"GPU {self.idx} busy (lock timeout)")
                time.sleep(0.2)

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.fd is not None:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
                os.close(self.fd)
        finally:
            self.fd = None


def build_env_with_gpu(min_free_gb: float = 6.0) -> Tuple[Dict[str, str], Optional[GPULock], Optional[int]]:
    chosen = pick_best_gpu(min_free_gb=min_free_gb)
    lock = None
    env = dict(os.environ)
    if chosen:
        lock = GPULock(chosen["index"])
        lock.__enter__()
        env["CUDA_VISIBLE_DEVICES"] = str(chosen["index"])
    env.setdefault("HF_HOME", "/docker/argese/clean_dataset/hf")
    env.setdefault("OFFLOAD_FOLDER", "/docker/argese/offload")
    env.setdefault("GPU_MEM_GB", "40")
    env.setdefault("CPU_MEM_GB", "128")
    return env, lock, (chosen["index"] if chosen else None)


# =========================
# Schemas
# =========================
class SplitterCfg(BaseModel):
    base_model: Optional[str] = None
    adapter_path: Optional[str] = None
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 0.9


class StoryCfg(BaseModel):
    adapter: Optional[str] = None
    length_preset: Optional[str] = Field(default=None, alias="preset")
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_new_tokens: Optional[int] = None
    min_new_tokens: Optional[int] = None
    retriever: Optional[str] = None
    retriever_model: Optional[str] = None
    k: Optional[int] = None
    max_ctx_chars: Optional[int] = None
    seg_words: Optional[int] = None
    overlap_words: Optional[int] = None


class TwoStageRequest(BaseModel):
    persona: str
    paper_title: Optional[str] = None
    markdown: str
    target_sections: int = 5
    splitter: Optional[SplitterCfg] = None
    storyteller: Optional[StoryCfg] = None
    callback_url: Optional[str] = None
    progress_id: Optional[str] = None


class TwoStageResponse(BaseModel):
    persona: str
    paper_title: Optional[str] = None
    outline: List[Dict[str, Any]]
    sections: List[Dict[str, Any]]
    meta: Dict[str, Any]
    title: Optional[str] = None


class OneSectionReq(BaseModel):
    persona: str
    paper_title: Optional[str] = None
    cleaned_text: str
    section: Dict[str, Any]
    storyteller: Optional[StoryCfg] = None


class OneSectionResp(BaseModel):
    title: str
    text: str
    paragraphs: List[str]


class RegenSectionsVMReq(BaseModel):
    persona: str
    paper_title: Optional[str] = None
    cleaned_text: str
    outline: List[Dict[str, Any]]
    targets: List[int]
    temp: Optional[float] = None
    top_p: Optional[float] = None
    length_preset: Optional[str] = None
    retriever: Optional[str] = None
    retriever_model: Optional[str] = None
    k: Optional[int] = None
    max_ctx_chars: Optional[int] = None
    seg_words: Optional[int] = None
    overlap_words: Optional[int] = None
    storyteller: Optional[StoryCfg] = None


class RegenSectionsVMResp(BaseModel):
    persona: str
    paper_title: Optional[str] = None
    sections: Dict[str, Dict[str, Any]]
    meta: Dict[str, Any]


# =========================
# Common helpers
# =========================
def _require_api_key(x_api_key: Optional[str]):
    if API_KEY and (x_api_key or "") != API_KEY:
        raise HTTPException(401, "Unauthorized")


def _normalize_section_text(sec: dict) -> dict:
    if not isinstance(sec, dict):
        return sec

    raw = (sec.get("narrative") or sec.get("text") or "").strip()
    raw = _unwrap_titled_text(raw)
    raw = _maybe_unwrap_json_text(raw, sec.get("title") or "")
    sec["text"] = str(raw).strip()

    paras = sec.get("paragraphs") or []
    if isinstance(paras, list):
        clean_paras = []
        for p in paras:
            if isinstance(p, str):
                p2 = _unwrap_titled_text(p)
                p2 = _maybe_unwrap_json_text(p2, sec.get("title") or "")
                clean_paras.append(p2.strip())
            else:
                clean_paras.append(str(p).strip())
        if len(clean_paras) <= 1 or len(set(clean_paras)) == 1:
            clean_paras = _split_paragraphs(sec["text"])
        sec["paragraphs"] = clean_paras
    else:
        sec["paragraphs"] = _split_paragraphs(sec["text"])

    sec.pop("narrative", None)
    return sec


def _run(cmd, timeout, cwd=None, env=None, trace=None, stage="proc"):
    cmd = [str(c) for c in cmd]
    if trace:
        trace("proc.start", f"Starting {stage}…", cmd=" ".join(cmd), timeout_s=timeout)

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    t0 = time.time()
    stdout_lines = []
    stderr_lines = []

    progress_id = (env or {}).get("PROGRESS_ID") or None
    callback_url = (env or {}).get("CALLBACK_URL") or None

    for line in proc.stdout:
        line = line.rstrip("\n")
        stdout_lines.append(line)

        if line.startswith("{") and line.endswith("}"):
            try:
                ev = json.loads(line)
                if isinstance(ev, dict):
                    _send_progress(callback_url, ev, progress_id=progress_id)
            except Exception:
                pass

        if trace:
            trace("proc.stream", f"{stage}: {line[:160]}", raw=line)

    proc.wait(timeout=timeout)
    dt = round(time.time() - t0, 3)

    if proc.returncode != 0:
        raise HTTPException(500, f"Subprocess failed ({stage})")

    if trace:
        trace("proc.ok", f"{stage} finished", elapsed_s=dt)

    return "\n".join(stdout_lines), "\n".join(stderr_lines)


def _read_first_jsonl(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise HTTPException(500, f"Missing output file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                try:
                    return json.loads(s)
                except Exception:
                    continue
    raise HTTPException(500, f"No valid JSON line in: {path}")


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.I)
    return s.strip()


def _extract_first_balanced_json(s: str) -> str:
    s = _strip_code_fences(s)
    start = s.find("{")
    if start < 0:
        return s.strip()
    depth = 0
    in_str = False
    esc = False
    i = start
    while i < len(s):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i + 1]
        i += 1
    return s[start:].strip()


def _sanitize_title(s: str, max_words: int = 12) -> str:
    if not s:
        return ""
    t = str(s).strip()
    t = t.splitlines()[0].strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I).strip()
    t = t.strip('"').strip("'").strip("`").strip()
    t = re.sub(r'^(?:Human|User|Assistant|System)\s*:\s*', '', t, flags=re.I).strip()
    t = re.split(r'\b(?:Here it is again|It is|The title you provided)\b', t, maxsplit=1, flags=re.I)[0].strip()
    t = re.sub(r'\s{2,}', ' ', t)
    words = t.split()
    if len(words) > max_words:
        t = " ".join(words[:max_words])
    return t


_UNWRAP_TITLED_RE = re.compile(
    r'^\s*\{\s*"title"\s*:\s*"(?:\\.|[^"\\])*"\s*,\s*"text"\s*:\s*"(.*)"\s*\}\s*$',
    re.DOTALL
)


def _unwrap_titled_text(s: str) -> str:
    if not isinstance(s, str):
        return s
    st = s.strip()

    if st.startswith("{") and '"text"' in st:
        try:
            obj = json.loads(st)
            if isinstance(obj, dict) and isinstance(obj.get("text"), str):
                return obj["text"].replace('\\"', '"').replace("\\n", "\n").strip()
        except Exception:
            pass

    m = _UNWRAP_TITLED_RE.match(st)
    if m:
        body = m.group(1)
        body = body.replace('\\"', '"').replace("\\n", "\n")
        return body.strip()

    return s.strip()


_TITLED_TEXT_RE = re.compile(
    r'^\s*\{\s*"title"\s*:\s*"(?:\\.|[^"\\])*"\s*,\s*"text"\s*:\s*"(.*)"\s*\}\s*$',
    re.DOTALL
)


def _force_unwrap_text(s: str, curr_title: str = "") -> str:
    if not isinstance(s, str):
        return s
    st = s.strip()

    if st.startswith("{") and '"text"' in st:
        try:
            obj = json.loads(st)
            if isinstance(obj, dict):
                if isinstance(obj.get("text"), str):
                    return obj["text"].strip()
                if isinstance(obj.get("sections"), list):
                    for it in obj["sections"]:
                        if (
                            isinstance(it, dict)
                            and isinstance(it.get("title"), str)
                            and it["title"].strip() == (curr_title or it["title"]).strip()
                            and isinstance(it.get("text"), str)
                        ):
                            return it["text"].strip()
        except Exception:
            pass

    m = _TITLED_TEXT_RE.match(st)
    if m:
        body = m.group(1)
        body = body.replace('\\"', '"').replace("\\n", "\n")
        return _strip_code_fences(body).strip()

    return _maybe_unwrap_json_text(st, curr_title).strip()


def _maybe_unwrap_json_text(txt: str, curr_title: str) -> str:
    s = (txt or "").strip()
    if not s:
        return s
    try:
        if (s.startswith('"') and s.endswith('"')) or s.startswith('\\"') or s.startswith("\\\""):
            s = json.loads(s)
    except Exception:
        s = s.replace('\\"', '"').replace("\\n", "\n")
    s = _strip_code_fences(s)
    if "{" in s and "}" in s:
        jraw = _extract_first_balanced_json(s)
        try:
            obj = json.loads(jraw)
            if isinstance(obj, dict):
                if isinstance(obj.get("text"), str):
                    return obj["text"].strip()
                if isinstance(obj.get("sections"), list):
                    for it in obj["sections"]:
                        if (
                            isinstance(it, dict)
                            and isinstance(it.get("title"), str)
                            and it["title"].strip() == curr_title
                            and isinstance(it.get("text"), str)
                        ):
                            return it["text"].strip()
                    for it in obj["sections"]:
                        if isinstance(it, dict) and isinstance(it.get("text"), str):
                            return it["text"].strip()
        except Exception:
            pass
    if "}{" in s:
        parts = re.split(r"\}\s*\{", s)
        for p in parts:
            cand = p
            if not cand.startswith("{"):
                cand = "{" + cand
            if not cand.endswith("}"):
                cand = cand + "}"
            try:
                o = json.loads(cand)
                if isinstance(o, dict):
                    if isinstance(o.get("text"), str):
                        return o["text"].strip()
                    if isinstance(o.get("sections"), list):
                        for it in o["sections"]:
                            if (
                                isinstance(it, dict)
                                and isinstance(it.get("title"), str)
                                and it["title"].strip() == curr_title
                                and isinstance(it.get("text"), str)
                            ):
                                return it["text"].strip()
            except Exception:
                continue
    return s


def _deep_unwrap_text(s: str, curr_title: str = "", trace=None, sec_index: Optional[int] = None) -> str:
    if s is None:
        return ""
    original = str(s)
    step = 0
    out = original

    def _log(stage, sample):
        _dbg(
            trace,
            "unwrap.step",
            stage,
            sec_index=sec_index,
            title=curr_title,
            in_len=len(sample),
            in_sample=sample[:260].replace("\n", " ↵ "),
        )

    step += 1
    _log(f"{step}.start", out)
    out = _unwrap_titled_text(out)
    step += 1
    _log(f"{step}.after__unwrap_titled", out)

    out = _maybe_unwrap_json_text(out, curr_title)
    step += 1
    _log(f"{step}.after__maybe_unwrap_json", out)

    out = _force_unwrap_text(out, curr_title)
    step += 1
    _log(f"{step}.after__force_unwrap", out)

    for j in range(3):
        t = (out or "").strip()
        changed = False
        if t.startswith('"') and t.endswith('"'):
            try:
                t2 = json.loads(t)
                out = str(t2)
                changed = True
                _log(f"iter{j}.json.loads_str", out)
            except Exception as e:
                _dbg(trace, "unwrap.warn", "json.loads on quoted failed", sec_index=sec_index, err=str(e))
        if not changed and t.startswith("{") and '"text"' in t:
            try:
                obj = json.loads(t)
                if isinstance(obj, dict) and isinstance(obj.get("text"), str):
                    out = obj["text"].strip()
                    changed = True
                    _log(f"iter{j}.picked_text_field", out)
            except Exception as e:
                _dbg(trace, "unwrap.warn", "json.loads on object failed", sec_index=sec_index, err=str(e))
        if not changed:
            break

    m = _TITLED_TEXT_RE.match((out or "").strip())
    if m:
        body = m.group(1).replace('\\"', '"').replace("\\n", "\n")
        out = _strip_code_fences(body).strip()
        _log("final.regex_unwrap", out)

    final = (out or "").strip()
    _dbg(
        trace,
        "unwrap.done",
        "Fin",
        sec_index=sec_index,
        title=curr_title,
        out_len=len(final),
        changed=(final != original),
        empty=(final == ""),
    )
    return final


def _split_paragraphs(text: str) -> List[str]:
    parts = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not parts:
        parts = re.split(r"(?<=[.!?])\s+(?=[A-ZÀ-ÖØ-Ý])", text)
    return [p.strip() for p in parts if p.strip()]


def _persona_guidance(persona: str) -> str:
    p = (persona or "").lower()
    if any(x in p for x in ["general", "public", "journalist"]):
        return (
            "Use accessible language and add 1–2 short, concrete examples or analogies "
            "to make abstract ideas tangible for non-experts."
        )
    if "student" in p:
        return "Explain with short didactic sentences. Define key terms briefly and include 1 intuitive example."
    if any(x in p for x in ["teacher", "clinician", "product manager", "investor"]):
        return "Highlight implications and practical takeaways. Prefer clear, concrete examples over theory."
    return "Keep technical precision. Examples are optional; focus on faithful paraphrase and clarity."


def _tfidf_topk_fragments(
    cleaned_text: str,
    query: str,
    max_words=180,
    overlap=60,
    k=3,
    max_chars=1200,
) -> List[str]:
    words = (cleaned_text or "").split()
    if not words:
        return []
    segs, i = [], 0
    step = max(1, max_words - overlap)
    while i < len(words):
        seg = " ".join(words[i : i + max_words]).strip()
        if seg:
            segs.append(seg)
        i += step
    if not segs:
        return []

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import linear_kernel

        vec = TfidfVectorizer(ngram_range=(1, 2), max_features=80000, lowercase=True)
        X = vec.fit_transform(segs)
        q = vec.transform([query])
        sims = linear_kernel(q, X).ravel()
        idx = sims.argsort()[::-1][: max(1, k)]
    except Exception:
        qtoks = set(re.findall(r"[a-zA-Z0-9]{3,}", query.lower()))
        scores = []
        for j, s in enumerate(segs):
            stoks = set(re.findall(r"[a-zA-Z0-9]{3,}", s.lower()))
            scores.append((j, len(qtoks & stoks)))
        idx = [j for j, _ in sorted(scores, key=lambda x: x[1], reverse=True)[: max(1, k)]]

    out = []
    budget = 0
    for j in idx:
        frag = segs[j].strip()
        if not frag:
            continue
        if budget + len(frag) + 2 > max_chars:
            break
        out.append(frag)
        budget += len(frag) + 2
    return out


def _strip_splitter_markers(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"<!--\s*\{[^>]*\}\s*-->", "", text)


def _build_story_env_and_lock(min_free_gb: float = 6.0) -> Tuple[Dict[str, str], Optional[GPULock], Optional[int]]:
    return build_env_with_gpu(min_free_gb=min_free_gb)


def _run_splitter(in_split: str, out_split: str, cfg: Dict[str, Any], timeout: int, env: Dict[str, str]):
    cmd = [
        PY,
        SPLITTER_SCRIPT,
        "--base_model",
        cfg["base_model"],
        "--adapter_path",
        cfg["adapter_path"],
        "--input_path",
        in_split,
        "--output_jsonl",
        out_split,
        "--sections",
        str(cfg["sections"]),
        "--max_new_tokens",
        str(cfg["max_new_tokens"]),
        "--temperature",
        str(cfg["temperature"]),
        "--top_p",
        str(cfg["top_p"]),
    ]
    _run(cmd, timeout=timeout, env=env)


def _run_storyteller(in_story: str, out_story: str, cfg: Dict[str, Any], timeout: int, env: Dict[str, str]):
    cmd = [
        PY,
        STORYTELLER_SCRIPT,
        "--in_jsonl",
        in_story,
        "--out_jsonl",
        out_story,
        "--adapter",
        cfg["adapter"],
        "--preset",
        cfg["preset"],
        "--temperature",
        str(cfg["temperature"]),
        "--top_p",
        str(cfg["top_p"]),
        "--max_new_tokens",
        str(cfg["max_new_tokens"]),
        "--per_sec_floor",
        str(cfg.get("per_sec_floor", 480)),
        "--retriever",
        cfg["retriever"],
        "--retriever_model",
        cfg["retriever_model"],
        "--k",
        str(cfg["k"]),
        "--max_ctx_chars",
        str(cfg["max_ctx_chars"]),
        "--seg_words",
        str(cfg["seg_words"]),
        "--overlap_words",
        str(cfg["overlap_words"]),
    ]
    cmd = [str(c) for c in cmd]
    print("USING STORYTELLER:", STORYTELLER_SCRIPT)
    print(f"[trace] storyteller.cmd {' '.join(cmd)}", flush=True)
    _run(cmd, timeout=timeout, env=env)


# =========================
# Progress dispatch
# =========================
def _send_progress(callback_url: Optional[str], event: dict, progress_id: Optional[str] = None):
    _buffer_progress(progress_id, event)
    if not callback_url:
        return
    try:
        headers = {}
        if API_KEY:
            headers["X-API-Key"] = API_KEY
        requests.post(callback_url, json=event, headers=headers, timeout=5)
    except Exception:
        pass


@app.post("/api/two_stage_story", response_model=TwoStageResponse)
def two_stage_story(req: TwoStageRequest, x_api_key: Optional[str] = Header(default=None)):
    _require_api_key(x_api_key)

    t_start = time.time()
    split_base = (req.splitter.base_model if (req.splitter and req.splitter.base_model) else SPLITTER_BASE_MODEL)
    split_adapter = (req.splitter.adapter_path if (req.splitter and req.splitter.adapter_path) else SPLITTER_ADAPTER_PATH)
    split_max_new = (req.splitter.max_new_tokens if (req.splitter and req.splitter.max_new_tokens is not None) else SPLITTER_MAX_NEW)
    split_temp = (float(req.splitter.temperature) if (req.splitter and req.splitter.temperature is not None) else 0.0)
    split_top_p = (float(req.splitter.top_p) if (req.splitter and req.splitter.top_p is not None) else 0.9)

    persona = (req.persona or "General Public").strip()
    paper_title = (req.paper_title or "").strip()
    markdown = req.markdown
    target_sections = max(1, int(req.target_sections or 5))

    st_adapter = (req.storyteller.adapter if (req.storyteller and req.storyteller.adapter) else STORYTELLER_ADAPTER)
    st_preset = ((req.storyteller.length_preset if (req.storyteller and req.storyteller.length_preset) else "medium")).lower()
    st_temp = float(req.storyteller.temperature) if (req.storyteller and req.storyteller.temperature is not None) else 0.0
    st_top_p = float(req.storyteller.top_p) if (req.storyteller and req.storyteller.top_p is not None) else 0.9
    st_max_new = int(req.storyteller.max_new_tokens) if (req.storyteller and req.storyteller.max_new_tokens is not None) else 1500
    st_min_new = int(req.storyteller.min_new_tokens) if (req.storyteller and req.storyteller.min_new_tokens is not None) else 600

    st_retriever = (req.storyteller.retriever if (req.storyteller and req.storyteller.retriever) else "auto")
    st_retriever_model = (
        req.storyteller.retriever_model if (req.storyteller and req.storyteller.retriever_model)
        else "sentence-transformers/all-MiniLM-L6-v2"
    )
    st_k = (int(req.storyteller.k) if (req.storyteller and req.storyteller.k is not None) else 3)
    st_max_ctx_chars = (int(req.storyteller.max_ctx_chars) if (req.storyteller and req.storyteller.max_ctx_chars is not None) else 1400)
    st_seg_words = (int(req.storyteller.seg_words) if (req.storyteller and req.storyteller.seg_words is not None) else 180)
    st_overlap_words = (int(req.storyteller.overlap_words) if (req.storyteller and req.storyteller.overlap_words is not None) else 60)

    rid = str(uuid.uuid4())
    log_path, trace = _trace_open(rid)
    trace("request.received", "Request received.", persona=persona, paper_title=paper_title, target_sections=target_sections)

    workdir = tempfile.mkdtemp(prefix=f"twostage_{rid}_")
    in_split = os.path.join(workdir, "in_splitter.jsonl")
    out_split = os.path.join(workdir, "out_splitter.jsonl")
    out_story = os.path.join(workdir, "out_story.jsonl")

    callback_url = getattr(req, "callback_url", None)
    progress_id = getattr(req, "progress_id", None)

    timings = {}
    env, gpu_lock, gpu_idx = _build_story_env_and_lock(min_free_gb=6.0)
    env = dict(env or {})
    env["TRACE_LOG_FILE"] = log_path
    env["TRACE_REQ_ID"] = rid
    env["PROGRESS_ID"] = progress_id or ""
    trace("gpu.pick", f"Picked GPU {gpu_idx}.", gpu_idx=gpu_idx)
    trace(
        "io.workspace",
        "Workspace prepared.",
        workdir=workdir,
        in_split=in_split,
        out_split=out_split,
        out_story=out_story,
        log_file=log_path,
    )
    _dbg(
        trace,
        "debug.env",
        "Runtime env snapshot",
        CUDA=os.environ.get("CUDA_VISIBLE_DEVICES"),
        HF_HOME=os.environ.get("HF_HOME"),
        SAVE_ARTIFACTS=SAVE_ARTIFACTS,
        DEBUG_VERBOSE=DEBUG_VERBOSE,
    )

    _send_progress(callback_url, {"type": "generation_start", "gpu": gpu_idx}, progress_id=progress_id)

    try:
        record = {"id": rid, "persona": persona, "md": markdown, "title": paper_title or None}
        with open(in_split, "w", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        trace("splitter.input_ready", "Splitter input ready.", bytes=os.path.getsize(in_split))

        t0 = time.time()
        trace(
            "splitter.start",
            f"Splitting into {target_sections} sections…",
            cfg={
                "base_model": split_base,
                "adapter_path": split_adapter,
                "max_new_tokens": split_max_new,
                "temperature": split_temp,
                "top_p": split_top_p,
                "sections": target_sections,
            },
        )

        _send_progress(callback_url, {"type": "outline_start"})

        env["CALLBACK_URL"] = callback_url or ""
        _run_splitter(
            in_split,
            out_split,
            cfg={
                "base_model": split_base,
                "adapter_path": split_adapter,
                "sections": target_sections,
                "max_new_tokens": split_max_new,
                "temperature": split_temp,
                "top_p": split_top_p,
            },
            timeout=TIMEOUT_SPLITTER,
            env=env,
        )
        timings["splitter_s"] = round(time.time() - t0, 3)

        trace("splitter.done", "Splitter finished.", elapsed_s=timings["splitter_s"], out_size=os.path.getsize(out_split))
        try:
            with open(out_split, "r", encoding="utf-8") as f:
                head = "".join([next(f) for _ in range(1)])
            _dbg(trace, "splitter.out.head", "Splitter first JSONL line", head=head.strip()[:1500])
        except Exception as e:
            _dbg(trace, "splitter.out.read_error", "Cannot read splitter head", error=str(e))

        t1 = time.time()
        trace(
            "storyteller.start",
            "Generating narrative sections…",
            cfg={
                "adapter": st_adapter,
                "preset": st_preset,
                "temperature": st_temp,
                "top_p": st_top_p,
                "max_new_tokens": st_max_new,
                "retriever": st_retriever,
                "k": st_k,
            },
        )

        _run_storyteller(
            in_story=out_split,
            out_story=out_story,
            cfg={
                "adapter": st_adapter,
                "preset": st_preset,
                "temperature": st_temp,
                "top_p": st_top_p,
                "max_new_tokens": st_max_new,
                "retriever": st_retriever,
                "retriever_model": st_retriever_model,
                "k": st_k,
                "max_ctx_chars": st_max_ctx_chars,
                "seg_words": st_seg_words,
                "overlap_words": st_overlap_words,
            },
            timeout=TIMEOUT_STORY,
            env=env,
        )
        timings["storyteller_s"] = round(time.time() - t1, 3)

        trace(
            "storyteller.done",
            "Storyteller finished.",
            elapsed_s=timings["storyteller_s"],
            out_size=os.path.getsize(out_story),
        )
        try:
            with open(out_story, "r", encoding="utf-8") as f:
                head = "".join([next(f) for _ in range(1)])
            _dbg(trace, "storyteller.out.head", "Storyteller first JSONL line", head=head.strip()[:1500])
        except Exception as e:
            _dbg(trace, "storyteller.out.read_error", "Cannot read storyteller head", error=str(e))

        try:
            split_obj = _read_first_jsonl(out_split)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"Cannot read splitter output JSONL: {e}")

        outline = split_obj.get("sections", []) or []
        titles = []
        for i, sec in enumerate(outline):
            if isinstance(sec, dict):
                titles.append(sec.get("title") or f"Section {i+1}")
            else:
                titles.append(f"Section {i+1}")

        _send_progress(callback_url, {"type": "outline_ready", "titles": titles}, progress_id=progress_id)

        story_obj = None
        try:
            story_obj = _read_first_jsonl(out_story)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"Cannot read storyteller output JSONL: {e}")

        gen = story_obj.get("generation", {}) or {}
        sections = gen.get("sections", []) or gen.get("Sections", []) or []

        norm_sections = []
        for i, sec in enumerate(sections):
            if not isinstance(sec, dict):
                continue
            title = sec.get("title") or f"Section {i+1}"

            raw = sec.get("text") or sec.get("narrative") or ""
            _dbg(
                trace,
                "postprocess.sec.raw",
                "Section raw payload",
                i=i,
                title=title,
                has_text=bool(sec.get("text")),
                has_narr=bool(sec.get("narrative")),
                text_head=str(sec.get("text") or "")[:180],
                narr_head=str(sec.get("narrative") or "")[:180],
            )

            txt = _deep_unwrap_text(raw, title, trace=trace, sec_index=i)

            for _ in range(3):
                t = (txt or "").strip()
                if t.startswith('"') and t.endswith('"'):
                    try:
                        t = json.loads(t)
                        txt = str(t)
                        continue
                    except Exception:
                        pass
                if t.startswith("{") and '"text"' in t:
                    try:
                        obj = json.loads(t)
                        if isinstance(obj, dict):
                            inner = obj.get("text")
                            if isinstance(inner, str):
                                txt = inner.strip()
                                continue
                    except Exception:
                        pass
                break

            paras_in = sec.get("paragraphs") or []
            paragraphs = []
            if isinstance(paras_in, list):
                for p in paras_in:
                    paragraphs.append(_deep_unwrap_text(str(p), title))

            if not paragraphs or len(set(paragraphs)) == 1:
                paragraphs = _split_paragraphs(txt)

            _dbg(
                trace,
                "postprocess.sec.final",
                "Section normalized",
                i=i,
                title=title,
                text_len=len(txt),
                para_count=len(paragraphs),
                text_head=txt[:220],
            )

            norm_sections.append({"title": title, "text": txt, "paragraphs": paragraphs})

        sections = norm_sections

        for i, sec in enumerate(sections):
            _send_progress(callback_url, {"type": "section_start", "index": i, "title": sec.get("title") or f"Section {i+1}"})
            _send_progress(callback_url, {"type": "section_done", "index": i}, progress_id=progress_id)

        trace(
            "postprocess.sections_parsed",
            f"Parsed {len(sections)} sections.",
            titles=[s.get("title") for s in sections],
        )
        sections = [s for s in sections if s.get("text", "").strip()]

        story_title = gen.get("title") if isinstance(gen, dict) else None
        story_title = _sanitize_title(story_title) if story_title else None
        if not story_title:
            if sections and isinstance(sections[0], dict):
                story_title = sections[0].get("title")
        if not story_title:
            base = split_obj.get("paper_title") or paper_title or "Untitled"
            story_title = f"{base} — {persona}"

        if not isinstance(sections, list) or len(sections) == 0:
            outline = split_obj.get("sections", []) or []
            sections = [
                {"title": s.get("title", f"Section {i+1}"), "text": s.get("description", "")}
                for i, s in enumerate(outline[:5])
            ]

        resp = {
            "persona": persona,
            "paper_title": split_obj.get("paper_title") or paper_title or None,
            "title": story_title,
            "outline": outline,
            "sections": sections,
            "meta": {
                "req_id": rid,
                "timings": {**timings, "total_s": round((time.time() - t_start), 3)},
                "trace": {"id": rid, "log_file": log_path},
                "gpu": {"idx": gpu_idx},
                "paths": {"in_splitter": in_split, "out_splitter": out_split, "out_story": out_story},
                "title_source": (
                    "storyteller.title"
                    if gen.get("title")
                    else "sections[0].title"
                    if (sections and sections[0].get("title"))
                    else "fallback(persona+paper_title)"
                ),
                "splitter_params": {
                    "base_model": split_base,
                    "adapter_path": split_adapter,
                    "max_new_tokens": split_max_new,
                    "temperature": split_temp,
                    "top_p": split_top_p,
                    "target_sections": target_sections,
                },
                "storyteller_params": {
                    "adapter": st_adapter,
                    "length_preset": st_preset,
                    "temperature": st_temp,
                    "top_p": st_top_p,
                    "max_new_tokens": st_max_new,
                    "retriever": st_retriever,
                    "retriever_model": st_retriever_model,
                    "k": st_k,
                    "max_ctx_chars": st_max_ctx_chars,
                    "seg_words": st_seg_words,
                    "overlap_words": st_overlap_words,
                },
            },
        }
        if SAVE_ARTIFACTS:
            _save_artifact(os.path.join(workdir, "artifacts"), "outline.json", json.dumps(outline, ensure_ascii=False, indent=2), trace)
            _save_artifact(os.path.join(workdir, "artifacts"), "sections.json", json.dumps(sections, ensure_ascii=False, indent=2), trace)
            _save_artifact(
                os.path.join(workdir, "artifacts"),
                "meta.json",
                json.dumps({"split": split_obj, "story": story_obj}, ensure_ascii=False)[:200000],
                trace,
            )
        resp["meta"]["debug_tag"] = "V3_ANTI_HALLU"
        return resp

    finally:
        try:
            _buffer_progress(progress_id, {"type": "all_done"})
        except Exception:
            pass

        try:
            shutil.rmtree(workdir, ignore_errors=True)
        except Exception:
            pass
        if gpu_lock:
            gpu_lock.__exit__(None, None, None)


class TwoStageFromOutlineReq(BaseModel):
    persona: str
    paper_title: str | None = None
    cleaned_text: str
    outline: List[Dict[str, Any]]
    storyteller: Optional[StoryCfg] = None


@app.post("/api/two_stage_story_from_outline")
def two_stage_story_from_outline(req: TwoStageFromOutlineReq, x_api_key: Optional[str] = Header(default=None)):
    _require_api_key(x_api_key)

    persona = (req.persona or "General Public").strip()
    paper_title = (req.paper_title or "Paper").strip()
    cleaned = req.cleaned_text or ""
    outline = req.outline or []

    st = req.storyteller or StoryCfg()
    story_adapter = st.adapter or STORYTELLER_ADAPTER
    st_preset = ((st.length_preset or "medium")).lower()
    st_temp = float(st.temperature) if st.temperature is not None else 0.0
    st_top_p = float(st.top_p) if st.top_p is not None else 0.9
    st_max_new = int(st.max_new_tokens) if st.max_new_tokens is not None else 1500
    st_min_new = int(st.min_new_tokens) if st.min_new_tokens is not None else 600

    st_retriever = (getattr(st, "retriever", None) or "auto")
    st_retriever_model = (getattr(st, "retriever_model", None) or "sentence-transformers/all-MiniLM-L6-v2")
    st_k = int(getattr(st, "k", 3) or 3)
    st_max_ctx_chars = int(getattr(st, "max_ctx_chars", 1400) or 1400)
    st_seg_words = int(getattr(st, "seg_words", 180) or 180)
    st_overlap_words = int(getattr(st, "overlap_words", 60) or 60)

    rid = str(uuid.uuid4())
    workdir = tempfile.mkdtemp(prefix=f"twostage2_{rid}_")
    in_story = os.path.join(workdir, "in_story.jsonl")
    out_story = os.path.join(workdir, "out_story.jsonl")

    env, gpu_lock, gpu_idx = _build_story_env_and_lock(min_free_gb=6.0)

    def _filter_unsupported(text: str, source: str) -> str:
        src = source.lower()
        sents = re.split(r"(?<=[.!?])\s+", text.strip())
        kept = []
        for s in sents:
            toks = re.findall(r"\b[A-Z][a-zA-Z\-]{2,}\b", s)
            if any(t.lower() not in src for t in toks):
                continue
            kept.append(s)
        return " ".join(kept).strip()

    try:
        record = {
            "id": rid,
            "persona": persona,
            "paper_title": paper_title,
            "cleaned_text": cleaned,
            "sections": outline,
        }
        with open(in_story, "w", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        _run_storyteller(
            in_story,
            out_story,
            cfg={
                "adapter": story_adapter,
                "preset": st_preset,
                "temperature": st_temp,
                "top_p": st_top_p,
                "max_new_tokens": st_max_new,
                "retriever": st_retriever,
                "retriever_model": st_retriever_model,
                "k": st_k,
                "max_ctx_chars": st_max_ctx_chars,
                "seg_words": st_seg_words,
                "overlap_words": st_overlap_words,
            },
            timeout=TIMEOUT_STORY,
            env=env,
        )

        story_obj = _read_first_jsonl(out_story)
        gen = story_obj.get("generation", {}) or {}
        sections = gen.get("sections", []) or []

        norm = []
        for i, sec in enumerate(sections):
            if not isinstance(sec, dict):
                continue
            title = sec.get("title") or (outline[i]["title"] if i < len(outline) else f"Section {i+1}")
            raw = sec.get("text") or sec.get("narrative") or ""
            text = _unwrap_titled_text(str(raw)) or (outline[i].get("description", "") if i < len(outline) else str(raw))
            text = _maybe_unwrap_json_text(text, title)
            text = _force_unwrap_text(str(text), title)

            if cleaned:
                text = _filter_unsupported(text, cleaned)
            norm.append({"title": title, "text": text, "paragraphs": _split_paragraphs(text)})

        ret_title = _sanitize_title(gen.get("title")) if isinstance(gen, dict) else None

        return {
            "persona": persona,
            "paper_title": paper_title,
            "title": paper_title,
            "outline": outline,
            "sections": norm,
            "meta": {
                "req_id": rid,
                "title_locked": True,
                "aiTitle": ret_title,
                "title_source": "paper_title_locked",
                "gpu": {"idx": gpu_idx},
                "storyteller_params": {
                    "length_preset": st_preset,
                    "temperature": st_temp,
                    "top_p": st_top_p,
                    "max_new_tokens": st_max_new,
                    "retriever": st_retriever,
                    "retriever_model": st_retriever_model,
                    "k": st_k,
                    "max_ctx_chars": st_max_ctx_chars,
                    "seg_words": st_seg_words,
                    "overlap_words": st_overlap_words,
                },
            },
        }
    finally:
        try:
            shutil.rmtree(workdir, ignore_errors=True)
        except Exception:
            pass
        if gpu_lock:
            gpu_lock.__exit__(None, None, None)


@app.post("/api/regen_sections_vm", response_model=RegenSectionsVMResp)
def regen_sections_vm(req: RegenSectionsVMReq, x_api_key: Optional[str] = Header(default=None)):
    _require_api_key(x_api_key)

    persona = (req.persona or "General Public").strip()
    paper_title = (req.paper_title or "").strip() or None
    cleaned_text = _strip_splitter_markers((req.cleaned_text or "").strip())
    outline = req.outline or []
    targets_in = req.targets or []

    if not outline or not isinstance(outline, list):
        raise HTTPException(422, "outline missing or invalid")
    if not cleaned_text:
        raise HTTPException(422, "cleaned_text missing")
    if not targets_in:
        raise HTTPException(400, "no valid targets")

    st = req.storyteller or StoryCfg()
    story_adapter = st.adapter or STORYTELLER_ADAPTER

    st_preset = (
        getattr(st, "length_preset", None)
        or getattr(st, "preset", None)
        or getattr(req, "length_preset", None)
        or "medium"
    )
    st_preset = str(st_preset).lower()

    raw_temp = None
    if getattr(st, "temperature", None) is not None:
        raw_temp = st.temperature
    elif req.temp is not None:
        raw_temp = req.temp
    else:
        raw_temp = 0.0
    st_temp = float(raw_temp)

    raw_top_p = None
    if getattr(st, "top_p", None) is not None:
        raw_top_p = st.top_p
    elif req.top_p is not None:
        raw_top_p = req.top_p
    else:
        raw_top_p = 0.9
    st_top_p = float(raw_top_p)

    st_max_new = int(st.max_new_tokens) if st.max_new_tokens is not None else 1500
    st_min_new = int(st.min_new_tokens) if st.min_new_tokens is not None else 600

    st_retriever = (getattr(st, "retriever", None) or req.retriever or "auto")
    st_retriever_model = (getattr(st, "retriever_model", None) or req.retriever_model or "sentence-transformers/all-MiniLM-L6-v2")
    st_k = int((getattr(st, "k", None) if getattr(st, "k", None) is not None else (req.k if req.k is not None else 3)))
    st_max_ctx_chars = int(
        (getattr(st, "max_ctx_chars", None) if getattr(st, "max_ctx_chars", None) is not None else (req.max_ctx_chars if req.max_ctx_chars is not None else 1400))
    )
    st_seg_words = int((getattr(st, "seg_words", None) if getattr(st, "seg_words", None) is not None else (req.seg_words if req.seg_words is not None else 180)))
    st_overlap_words = int(
        (getattr(st, "overlap_words", None) if getattr(st, "overlap_words", None) is not None else (req.overlap_words if req.overlap_words is not None else 60))
    )

    print(
        f"[/api/regen_sections_vm] resolved knobs: "
        f"preset={st_preset}, temp={st_temp}, top_p={st_top_p}, "
        f"max_new={st_max_new}, retriever={st_retriever}, "
        f"k={st_k}, max_ctx_chars={st_max_ctx_chars}, "
        f"seg_words={st_seg_words}, overlap_words={st_overlap_words}",
        flush=True,
    )

    uniq_targets = sorted({int(t) for t in targets_in if isinstance(t, (int,))})
    valid_targets: List[int] = []
    for t in uniq_targets:
        if 0 <= t < len(outline):
            valid_targets.append(t)
    if not valid_targets:
        raise HTTPException(400, "no valid targets in range")

    env, gpu_lock, gpu_idx = _build_story_env_and_lock(min_free_gb=6.0)

    sparse_sections: Dict[str, Dict[str, Any]] = {}
    timings: Dict[str, Any] = {"per_section_s": {}}
    rid = str(uuid.uuid4())

    try:
        for t in valid_targets:
            start_t = time.time()
            workdir = tempfile.mkdtemp(prefix=f"regen_{rid}_{t}_")
            in_story = os.path.join(workdir, "in_story.jsonl")
            out_story = os.path.join(workdir, "out_story.jsonl")

            target_section = outline[t] or {}
            fixed_title = (target_section.get("title") or f"Section {t+1}").strip()
            section_record = {
                "id": f"{rid}_{t}",
                "persona": persona,
                "paper_title": paper_title or "",
                "cleaned_text": cleaned_text,
                "sections": [{"title": fixed_title, "description": (target_section.get("description") or "")}],
            }

            with open(in_story, "w", encoding="utf-8") as f:
                f.write(json.dumps(section_record, ensure_ascii=False) + "\n")

            _run_storyteller(
                in_story,
                out_story,
                cfg={
                    "adapter": story_adapter,
                    "preset": st_preset,
                    "temperature": st_temp,
                    "top_p": st_top_p,
                    "max_new_tokens": st_max_new,
                    "retriever": st_retriever,
                    "retriever_model": st_retriever_model,
                    "k": st_k,
                    "max_ctx_chars": st_max_ctx_chars,
                    "seg_words": st_seg_words,
                    "overlap_words": st_overlap_words,
                },
                timeout=TIMEOUT_STORY,
                env=env,
            )

            story_obj = _read_first_jsonl(out_story)
            gen = story_obj.get("generation", {}) or {}
            out_sections = gen.get("sections", []) or []

            text = ""
            if out_sections and isinstance(out_sections[0], dict):
                raw = out_sections[0].get("text") or out_sections[0].get("narrative") or ""
                text = _unwrap_titled_text(str(raw))
                text = _maybe_unwrap_json_text(text, fixed_title).strip()
                text = _force_unwrap_text(str(text), fixed_title)
                text = _strip_splitter_markers(text)

            if not text:
                continue

            sparse_sections[str(t)] = {"title": fixed_title, "text": text, "paragraphs": _split_paragraphs(text)}
            timings["per_section_s"][str(t)] = round(time.time() - start_t, 3)
            shutil.rmtree(workdir, ignore_errors=True)

        meta = {
            "req_id": rid,
            "timings": {**timings},
            "gpu": {"idx": gpu_idx},
            "storyteller_params": {
                "adapter": story_adapter,
                "length_preset": st_preset,
                "temperature": st_temp,
                "top_p": st_top_p,
                "max_new_tokens": st_max_new,
                "retriever": st_retriever,
                "retriever_model": st_retriever_model,
                "k": st_k,
                "max_ctx_chars": st_max_ctx_chars,
                "seg_words": st_seg_words,
                "overlap_words": st_overlap_words,
            },
            "targets": valid_targets,
        }

        return {"persona": persona, "paper_title": paper_title, "sections": sparse_sections, "meta": meta}
    finally:
        if gpu_lock:
            gpu_lock.__exit__(None, None, None)


class ParagraphOps(BaseModel):
    paraphrase: bool
    simplify: bool
    length_op: Literal["keep", "shorten", "lengthen"]


class ParagraphOpsReq(BaseModel):
    persona: str
    paper_title: str
    cleaned_text: str
    section: Dict[str, Any]
    paragraph_index: int
    ops: ParagraphOps
    temperature: float = 0.3
    top_p: float = 0.9
    n: int = 2
    k_ctx: int = 3
    max_ctx_chars: int = 1200
    length_preset: Optional[Literal["short", "medium", "long"]] = None
    seed: Optional[int] = None


def _snap_creativity_to_step10(temp_in: float) -> float:
    base = temp_in
    if base > 1.2:
        base = base / 100.0
    base = max(0.10, min(1.20, base))
    snapped = round(base, 1)
    if snapped < 0.10:
        snapped = 0.10
    return snapped


def _build_temperature_schedule(base_temp: float, n: int) -> List[float]:
    out = []
    for i in range(max(1, int(n))):
        t = min(1.20, base_temp + 0.04 * i)
        out.append(round(t, 2))
    return out


def _normalize_text_for_dedup(s: str) -> str:
    return re.sub(r"[^\w\s]+", "", (s or "")).lower().strip()


@app.post("/api/regen_paragraph_vm")
def regen_paragraph_vm(req: ParagraphOpsReq, x_api_key: Optional[str] = Header(default=None)):
    _require_api_key(x_api_key)

    persona = req.persona or "General Public"
    paper_title = req.paper_title or "Paper"
    cleaned_text = req.cleaned_text or ""
    sec = req.section or {}
    paragraphs = sec.get("paragraphs") or []
    title = sec.get("title") or "Section"
    idx = int(req.paragraph_index)
    if not paragraphs or idx < 0 or idx >= len(paragraphs):
        raise HTTPException(422, "invalid paragraph_index or section.paragraphs")

    target_text = paragraphs[idx]
    context_local = "\n\n".join(paragraphs[max(0, idx - 1) : idx + 2])

    ctx_frags = _tfidf_topk_fragments(
        cleaned_text=cleaned_text,
        query=target_text,
        max_words=180,
        overlap=60,
        k=max(1, int(req.k_ctx)),
        max_chars=max(256, int(req.max_ctx_chars)),
    )
    context_retrieved = "\n\n".join(ctx_frags).strip() if ctx_frags else ""

    ops = req.ops

    simplify_instruction = "Use simpler vocabulary and sentences." if ops.simplify else ""
    paraphrase_instruction = "Paraphrase faithfully without changing facts." if ops.paraphrase else "Keep most phrasing unchanged."
    persona_block = _persona_guidance(persona)

    length_op = getattr(req.ops, "length_op", "keep")
    length_preset = (req.length_preset or {"shorten": "short", "keep": "medium", "lengthen": "long"}.get(length_op, "medium")).lower()

    if length_preset == "short":
        length_instruction = "Write a shorter version, target 20–45 words (max ~60)."
        max_new_tokens = 100
        min_new_tokens = 20
    elif length_preset == "long":
        length_instruction = "Write a longer version, target 120–190 words (max ~200)."
        max_new_tokens = 300
        min_new_tokens = 160
    else:
        length_instruction = "Keep similar length, target 50–100 words (max ~110)."
        max_new_tokens = 200
        min_new_tokens = 90

    base_temp = _snap_creativity_to_step10(float(req.temperature))
    temps = _build_temperature_schedule(base_temp, req.n)
    top_p = float(req.top_p or 0.9)

    prompt = f"""
You are an AI Scientist Storyteller writing for a {persona}.
Paper title: {paper_title}
Section: {title}

Guidance for the persona:
- {persona_block}

Context (nearby paragraphs):
\"\"\"{context_local}\"\"\"

Retrieved paper fragments (factual grounding):
\"\"\"{context_retrieved if context_retrieved else "[NO CONTEXT FOUND]"}\"\"\"

Target paragraph to rewrite:
\"\"\"{target_text}\"\"\"

Rewrite this paragraph following these operations:
- {paraphrase_instruction}
- {simplify_instruction}
- {length_instruction}
- Target length must fit the range stated above; do not exceed it.
- Use English only. No lists; write fluent prose.
- Never invent facts, names, or numbers. If something is not in context, do not add it.
- If audience is non-expert, prefer a brief concrete example or analogy when helpful.

Return ONLY JSON with this schema:
{{ "text": "..." }}
""".strip()

    rid = str(uuid.uuid4())
    env, gpu_lock, gpu_idx = _build_story_env_and_lock(min_free_gb=6.0)

    outputs: List[Dict[str, Any]] = []
    seen = set()
    MAX_ATTEMPTS_PER_ALT = 5

    try:
        for i, t in enumerate(temps):
            workdir = tempfile.mkdtemp(prefix=f"parops_{rid}_{i}_")
            in_json = os.path.join(workdir, "in.jsonl")
            out_json = os.path.join(workdir, "out.jsonl")

            attempt = 0
            curr_temp = t
            while True:
                seed = (req.seed if req.seed is not None else None)
                if seed is None:
                    seed = int.from_bytes(os.urandom(4), "big")

                record = {
                    "id": f"{rid}_{i}_{attempt}",
                    "persona": persona,
                    "paper_title": paper_title,
                    "cleaned_text": cleaned_text,
                    "prompt": prompt,
                }
                with open(in_json, "w", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

                cmd = [
                    PY, STORYTELLER_SCRIPT,
                    "--in_jsonl", in_json,
                    "--out_jsonl", out_json,
                    "--adapter", STORYTELLER_ADAPTER,
                    "--preset", length_preset,
                    "--temperature", str(curr_temp),
                    "--top_p", str(top_p),
                    "--max_new_tokens", str(max_new_tokens),
                    "--seed", str(seed),
                ]

                _run(cmd, timeout=min(TIMEOUT_STORY, 900), env=env, stage=f"parregen_alt{i}_try{attempt}")

                story_obj = _read_first_jsonl(out_json)
                gen = story_obj.get("generation") or story_obj.get("output") or {}

                candidate_text = None
                if isinstance(gen, dict) and isinstance(gen.get("text"), str):
                    try:
                        obj = json.loads(gen["text"])
                        if isinstance(obj, dict) and isinstance(obj.get("text"), str):
                            candidate_text = obj["text"].strip()
                        else:
                            candidate_text = gen["text"].strip()
                    except Exception:
                        candidate_text = gen["text"].strip()
                elif isinstance(gen, dict) and isinstance(gen.get("alternatives"), list):
                    for a in gen["alternatives"]:
                        if isinstance(a, dict) and isinstance(a.get("text"), str):
                            candidate_text = a["text"].strip()
                            break
                elif isinstance(gen, str) and gen.strip():
                    candidate_text = gen.strip()

                if not candidate_text:
                    candidate_text = target_text

                norm = _normalize_text_for_dedup(candidate_text)
                if norm not in seen:
                    seen.add(norm)
                    outputs.append({"text": candidate_text, "temperature": curr_temp, "seed": seed})
                    break

                if attempt >= MAX_ATTEMPTS_PER_ALT:
                    outputs.append({"text": candidate_text, "temperature": curr_temp, "seed": seed, "dup": True})
                    break

                curr_temp = round(min(1.20, curr_temp + 0.02), 2)
                attempt += 1

            try:
                shutil.rmtree(workdir, ignore_errors=True)
            except Exception:
                pass

    finally:
        if gpu_lock:
            gpu_lock.__exit__(None, None, None)

    alts = [{"text": o["text"]} for o in outputs[: max(1, int(req.n))]]

    return {
        "alternatives": alts,
        "meta": {
            "applied_ops": {"paraphrase": ops.paraphrase, "simplify": ops.simplify, "length_op": ops.length_op},
            "section_title": title,
            "paragraph_index": idx,
            "gpu": {"idx": gpu_idx},
            "ctx_used": bool(context_retrieved),
            "creativity": {"base_temperature": base_temp, "schedule": temps, "top_p": top_p},
        },
    }


def _public_file_url(paper_id: str) -> str:
    return f"/papers/{paper_id}.pdf"


@app.post("/api/papers/upload")
def api_papers_upload(file: UploadFile = File(...), paper_id: str = Form(...)):
    safe_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", paper_id).strip("_")
    if not safe_id:
        raise HTTPException(400, "invalid paper_id")

    out_path = os.path.join(UPLOAD_DIR, f"{safe_id}.pdf")
    try:
        with open(out_path, "wb") as f:
            f.write(file.file.read())
    finally:
        try:
            file.file.close()
        except Exception:
            pass

    return {"ok": True, "file_url": f"/papers/{safe_id}.pdf"}


@app.post("/papers/upload")
def upload_paper_compat(
    paper_id: str = Form(...),
    file: UploadFile = File(...),
    x_api_key: str | None = Header(default=None),
):
    return api_papers_upload(file=file, paper_id=paper_id)


@app.get("/api/progress/{progress_id}")
def get_progress(progress_id: str):
    with PROGRESS_LOCK:
        events = list(PROGRESS.get(progress_id, []))
        if any(isinstance(e, dict) and e.get("type") == "all_done" for e in events):
            PROGRESS.pop(progress_id, None)
    return {"events": events}
