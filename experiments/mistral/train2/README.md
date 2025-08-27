# Mistral — Training 2

**Base model:** `mistralai/Mistral-7B-Instruct-v0.2` 
 
**Method:** LoRA fine-tuning (full papers + complete stories, no section splits)  

**Stopping:** Early stopping triggered at epoch 11 (monitoring loss only).

## Rationale
Use full-document inputs to exploit the larger context window and test if end-to-end story generation improves without section splits.

## Results (qualitative)
- Slight improvements in fluency.
- Semantic alignment remained weak (no structured section focus).
- ROUGE-1 **0.2218** • ROUGE-2 **0.0413** • ROUGE-L **0.1133** • BLEU **0.0189** • BERTScore-F1 **−0.1524** • Avg FKGL **13.29**

## Artifacts

Sample outputs included (student/teacher/general_public). Some drift (citations/acknowledgments) is deliberately kept to document limitations observed in T2.
- Notebook: `Mistral_2.ipynb`
- Samples: `samples/predictions_head.json`

## Config & Metrics
See `config.json` and `metrics.json`.
