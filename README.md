# AI Scientist Storyteller — Experiments & Prototype

This repository hosts the code and minimal artifacts for the **AI Scientist Storyteller** project:
- training & evaluation scripts for Flan-T5 and Mistral,
- prompt templates and inference utilities,
- curated experiment metadata (no large checkpoints).

## Repository layout
- `scripts/` — training/inference utilities.
- `experiments/` — per-run metadata (config, tiny samples, metrics).
- `final_infer_codes/` - Final infer scripts (SPLITTER + STORYTELLER)
- `datasets/` — **local** datasets (ignored by git).
- `data-samples/` — tiny text snippets for reproducibility.
- `outputs/` — demo outputs (small).
- `.gitignore` — excludes large/private files.

## Datasets
Real datasets live in `datasets/` (not versioned). Small heads are provided under `data-samples/` to illustrate format.

## Experiments (summary)

### Flan-T5
**Training 1 (full finetune, flan-t5-base)**  
- Epochs: 3, LR=5e-5, eff. batch=8, max seq len: 1024.  
- Metrics (test): ROUGE-1 0.120, ROUGE-2 0.023, ROUGE-L 0.079, BLEU ≈ 0.0035.  
- Artifacts: `experiments/flan/train1/` (config, metrics, sample eval).

**Training 2 (QLoRA+LoRA, flan-t5-xl)**  
- Epochs: 3, LR=1e-4, eff. batch=8, max seq len: 1024.  
- Metrics (test): ROUGE-1 0.1239, ROUGE-2 0.0246, ROUGE-L 0.0817, BLEU ≈ 0.0039.  
- Artifacts: `experiments/flan/train2/ (script: scripts/flan_t5_train2.py)`.

**Training 3 (QLoRA+LoRA, flan-t5-large, chunked)**  
- Epochs: 5, LR=2e-4, eff. batch=8, max seq len: 1024 (chunked).  
- Metrics (test): ROUGE-1 0.134, ROUGE-2 0.024, ROUGE-L 0.090, BLEU ≈ 0.00285.  
- Artifacts: `experiments/flan/train3/`.

**Training 4 (QLoRA, LongT5-large)**  
- Epochs: 30, LR=2e-4, eff. batch=8, max seq len: 4096→1024.  
- Metrics: ROUGE/BLEU = 0, eval loss ≈ 10.37 (collapse).  
- Artifacts: `experiments/flan/train4/` (scripts + samples).

**Training 5 (QLoRA+LoRA, flan-t5-large; training==inference prompts)**  
- Epochs: ~20 (early-stopped ≈19), LR=2e-4, eff. batch=16, src/tgt=768/384.  
- Metrics (test): ROUGE-1 0.197, ROUGE-2 0.035, ROUGE-L 0.126, BLEU 1.90, BERTScore-F1 −0.189.  
- Artifacts: `experiments/flan/train5/` (scripts, config, metrics, eval, outputs).

**Raw baseline (no fine-tuning, flan-t5-large)**  
- Decoding: temp=0.0, top_p=1.0, max_in=480, max_new=360, min_new=140.  
- Paper: VesselVerse; sections: Intro/Methods/Results/Conclusion.  
- Macro-avg metrics: ROUGE-1 0.483, ROUGE-2 0.334, ROUGE-L 0.297, BLEU 18.04, BERTScore-F1 0.318.  
- Artifacts: `experiments/flan/raw/`.


### Mistral
**Training 1 (7B Instruct v0.2, LoRA)**  
- Metrics (test): ROUGE-1 0.2179, ROUGE-2 0.0385, ROUGE-L 0.1191, BLEU 0.0195, BERTScore-F1 −0.1280.  
- Artifacts: `experiments/mistral/train1/` (config, metrics, notebook).

**Training 2 (LoRA, v0.2, full-doc inputs)**  
- Metrics (test): ROUGE-1 0.2218, ROUGE-2 0.0413, ROUGE-L 0.1133, BLEU 0.0189, BERTScore-F1 −0.1524, Avg FKGL 13.29.  
- Early stopping at epoch 11 (loss only).  
- Artifacts: `experiments/mistral/train2/`.

**Training 3 (LoRA, v0.2, full dataset)**  
- Stopping criterion: early stopping on **loss + semantic metrics** (ROUGE, BLEU, BERTScore).  
- Goal: improve coherence, relevance, and section length allocation.  
- Results: ≈1 useful output every 10–15 generations; section allocation poor, coherence still inconsistent.  
- Artifacts: `experiments/mistral/train3/` (training script + inference variants: `scripts/mistral_train3.py`, `scripts/mistral_infer5_v4_train3.py`, `scripts/mistral_infer_new_train3.py`).

**Training 4 (LoRA+QLoRA, v0.3, dataset reformulation)**  
- Dataset: reformulated into two tasks — (1) logical section segmentation, (2) persona-conditioned story generation.  
- Epochs: up to 12 with patience=3 (early stopping on loss + semantic metrics).  
- LR=2e-4, eff. batch=64 (2×8 grad accum × 4 GPUs), max seq len: 4096.  
- Metrics: composite ≈ 0.07 (ROUGE-L ~0.064, title overlap ~0.105, persona alignment ~0.05).  
- Notes: best Mistral run so far — coherent, persona-aware stories; still some repetition and occasional drift.  
- Artifacts: `experiments/mistral/train4/` (scripts, config, metrics, samples).


## Notes
- Large models/checkpoints are not stored here.
- Full experiment log is kept on Notion.
