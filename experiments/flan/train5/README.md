# Flan-T5 — Training 5

The model was based on **google/flan-t5-large** and adapted with **QLoRA (4-bit) + LoRA**.

- **LoRA r:** 16 — **α:** 32 — **dropout:** 0.05  
- **Target modules:** Q, K, V, O, wi, wo  
- **Quantization:** NF4 4-bit (double quant), bf16 compute  
- **Gradient checkpointing:** enabled  
- **Dataset:** 1662 train / 204 val (split by `id_story`, 10%)  
- **Epochs:** ~20 (early stopped ≈19, patience=3; composite metric `0.7 * ROUGE-L + 0.3 * BLEU/100`)  
- **LR:** 2e-4 — **Weight decay:** 0.01 — **Scheduler:** cosine — **Warmup:** 3%  
- **Batching:** train_bs=2, grad_accum=8 (effective 16)  
- **Tokenization:** source 768, target 384  
- **Prompt template:** explicit “AI Scientist Storyteller”; **training == inference**.

## Results on the Test Set
- **ROUGE-1:** 0.1967  
- **ROUGE-2:** 0.0349  
- **ROUGE-L:** 0.1262  
- **BLEU:** 1.90  
- **BERTScore F1:** −0.189

## Files
- `config.json` — training configuration
- `metrics.json` — test metrics (summary)
- `eval/` — raw eval artifacts (`eval_test_metrics.json`, `eval_test_preds.jsonl`)
- `outputs/` — section-wise prompts/refs/generations and `metrics_flan_sft_promptExact.json`
- Scripts: `scripts/flan_t5_train5.py`, `scripts/flan_t5_eval_train5.py`, `scripts/flan_t5_gen_vesselverse_train5.py`
