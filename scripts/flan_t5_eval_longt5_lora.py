import os, numpy as np, torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          DataCollatorForSeq2Seq)
from peft import PeftModel
from evaluate import load as load_metric

DATA_DIR   = "/docker/argese/volumes/final_data"
TEST_JSON  = os.path.join(DATA_DIR, "test_ready_trunc.json")
ADAPTER_DIR = "./longt5_local_base_lora_4bit"
BASE = "google/long-t5-local-base"
MAX_INPUT_LEN, MAX_TARGET_LEN = 1024, 1024
OUT_DIR = "./eval_out"

def tok(batch, tokzr):
    x = tokzr(batch["input"], max_length=MAX_INPUT_LEN, truncation=True, padding=False)
    y = tokzr(text_target=batch["output"], max_length=MAX_TARGET_LEN, truncation=True, padding=False)
    x["labels"] = y["input_ids"]
    return x

def main():
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    ds = load_dataset("json", data_files={"test": TEST_JSON})["test"]
    test_tok = ds.map(lambda b: tok(b, tokenizer), batched=True, remove_columns=ds.column_names)

    base = AutoModelForSeq2SeqLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.eval()

    rouge = load_metric("rouge")
    sacrebleu = load_metric("sacrebleu")

    def safe_decode(batch):
        return tokenizer.batch_decode(batch, skip_special_tokens=True)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        pred_txt = safe_decode(preds)
        lab_txt  = safe_decode(labels)
        r = rouge.compute(predictions=pred_txt, references=lab_txt)
        sb = sacrebleu.compute(predictions=pred_txt, references=[[t] for t in lab_txt])

        def _get(x):
            try: return float(x)
            except: return float(getattr(x, "mid", x).fmeasure)
        return {
            "rouge1": round(_get(r["rouge1"])*100, 2),
            "rouge2": round(_get(r["rouge2"])*100, 2),
            "rougeL": round(_get(r["rougeL"])*100, 2),
            "sacrebleu": round(sb["score"], 2),
        }

    collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, model=model)

    args = Seq2SeqTrainingArguments(
        output_dir=OUT_DIR,
        do_train=False,
        do_eval=True,
        per_device_eval_batch_size=1,     # alza se hai VRAM
        predict_with_generate=True,
        generation_max_length=256,        # accelera l’eval
        generation_num_beams=1,           # veloce
        bf16=True,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        eval_dataset=test_tok,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    metrics = trainer.evaluate()
    print("=== TEST METRICS ===")
    for k,v in metrics.items():
        if isinstance(v, float): print(f"{k}: {v}")

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    main()
