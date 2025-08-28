# Mistral — Training 3

**Model**: mistralai/Mistral-7B-Instruct-v0.2  
**Method**: LoRA SFT on full papers + stories.  
**Early stopping**: based on loss **and** semantic probe (ROUGE/BLEU/BERTScore).  
**Goal**: improve coherence, relevance, and section-level length allocation.

### Artifacts
- `config.json` — training config (high level)
- `metrics.json` — aggregate metrics (validation/test)
- `samples/story_5_sections_Student.md` — qualitative sample
- Scripts used: `scripts/mistral_train3.py`, `scripts/mistral_infer5_v4_train3.py`

### Notes
Generations occasionally match the desired narrative style but inconsistently (~1 useful out of 10–15). Frequent concept repetition and occasional drift from section focus.
