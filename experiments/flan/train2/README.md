# Flan-T5 — Training 2
- Base model: `google/flan-t5-xl`
- Method: **QLoRA (4b) + LoRA** — r=16, alpha=32, dropout=0.1
- Epochs: 3, LR=1e-4, eff. batch=8, max len 1024/1024
- Dataset: `../../datasets/flan_story_v1/` (ignored in git), samples in `../../data-samples/flan_story_v1/`
- Metrics: see `metrics.json`; sample outputs in `samples/`
