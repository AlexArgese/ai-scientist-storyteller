# Scripts index

## Flan-T5
- `flan_t5_train1.py` — Training 1 (flan-t5-base, full finetune)
- `flan_t5_train2.py` — Training 2 (flan-t5-xl, QLoRA+LoRA)
- `flan_t5_train3.py` — Training 3 (flan-t5-large, QLoRA+LoRA, chunked 1024)
- `flan_t5_train4.py` — Training 4 (long-t5-large, QLoRA+LoRA)
- `flan_t5_eval_longt5_lora.py` — Eval utilities for Training 4
- `flan_t5_infer_longt5.py` — Inference (story generation) for Training 4

For per-run configs/metrics/samples see `experiments/<model family>/trainX/`.
