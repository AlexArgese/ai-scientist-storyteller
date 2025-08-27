# Mistral — Training 1 (7B Instruct v0.2, LoRA)

- **Model:** `mistralai/Mistral-7B-Instruct-v0.2`
- **Method:** LoRA (PEFT), tokenizer `trust_remote_code=True`
- **Data:** section-based splits (same preparation as Flan experiments)
- **Generation (eval):** `max_new_tokens=256`, greedy (`do_sample=False`)

**Results (466 test samples):**
- ROUGE-1 0.2179 • ROUGE-2 0.0385 • ROUGE-L 0.1191 • BLEU 0.0195 • BERTScore-F1 −0.1280

**Notes:** Outputs were short and often uninformative; this served as a baseline for subsequent runs.  
No prediction samples were stored for this experiment.
