# Flan-T5 — Training 3 (Large, QLoRA+LoRA, chunked 1024)

- **Model:** google/flan-t5-large  
- **Method:** QLoRA (4-bit nf4, bf16 compute) + LoRA — r=32, α=64, dropout=0.1  
- **Epochs:** 5 — **LR:** 2e-4 — **Eff. batch:** 8  
- **Max len:** 1024 in/out con **chunking** (no truncation)  
- **Split:** ~70/10/20 (val ricavata dal train 80%)  
- **Eval:** ROUGE/BLEU (vedi `metrics.json`); sample predictions in `samples/preds_test_head.json`.

Script: `scripts/flan_t5_train3.py` — Notebook: `experiments/flan/train3/T5_3.ipynb`
