# AI Scientist Storyteller — Experiments & Prototype

This repository hosts the code and minimal artifacts for the **AI Scientist Storyteller** project:
- training & evaluation scripts for Flan-T5 and Mistral,
- prompt templates and inference utilities,
- curated experiment metadata (no large checkpoints).

## Repository layout
- `scripts/` — training/inference utilities.
- `experiments/` — per-run metadata (config, tiny samples, metrics).
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

## Notes
- Large models/checkpoints are not stored here.
- Full experiment log is kept on Notion.
