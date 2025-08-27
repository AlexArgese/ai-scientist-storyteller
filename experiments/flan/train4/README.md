# Flan-T5 — Training 4 (LongT5-large, QLoRA 4-bit)

- **Model:** `google/long-t5-large`
- **Method:** QLoRA (4-bit nf4, bf16 compute) + LoRA (r=32, α=64, dropout=0.1)
- **Epochs:** 30 — **Eff. batch:** 8 — **Max len:** 4096 in / 1024 out
- **Outcome:** collapse (all metrics ~0; eval_loss ≈ 10.37).  
  Likely due to too many epochs on small data + quantization instability. Early stopping not effective.

**Scripts:**  
- training `scripts/flan_t5_train4.py`  
- evaluation `scripts/flan_t5_eval_longt5_lora.py`  
- inference `scripts/flan_t5_infer_longt5.py`  

**Artifacts:** see `samples/` (train log head, paper head, one generated sample).
