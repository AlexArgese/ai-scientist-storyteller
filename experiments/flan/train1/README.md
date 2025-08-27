# Flan-T5 — Training 1

- Base model: `google/flan-t5-base`
- Finetuning: full finetune (no LoRA)
- Epochs: 3, LR=5e-5, eff. batch=8
- Dataset: `../../datasets/flan_story_v1/` (local, ignored), samples in `../../data-samples/flan_story_v1/`
- Metrics: see `metrics.json`; sample eval rows in `samples/evaluation_results_head.csv`
