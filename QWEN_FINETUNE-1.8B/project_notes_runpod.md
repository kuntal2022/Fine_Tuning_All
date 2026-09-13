# Pharma Fine-Tuning Project — Master Notes

One-stop reference for the whole pipeline: environment setup, data sources,
the 3-stage training pipeline, and fixes for errors already hit once so
they don't need to be re-solved.

---

## 0. Environment setup

Run `setup.sh` on a fresh RunPod pod (see that file). It installs `uv`,
creates a venv (`ktl_env`), installs all dependencies pinned to versions
known to work together (`transformers==4.46.0` is pinned — `5.x` broke
safetensors loading with bitsandbytes quantization).

Recommended GPU: **RTX PRO 4500, 32GB VRAM, $0.72/hr** (not the SE variant —
same price, no meaningful benefit for this workload since it's GPU-bound,
not RAM/vCPU-bound).

**IMPORTANT — RunPod data persistence:** If a pod is *terminated* (not just
stopped), all data on it is permanently lost. Always push finished
artifacts (datasets, merged models) to Hugging Face Hub or download them
*immediately* after each stage completes — don't leave them sitting only
on the pod.

---

## 1. Data sources used

### Stage 1 (Domain Adaptation) — raw text corpus
- **OCR'd textbook**: `Textbook of Pharmacology` by B.C. Bose (archive.org,
  public domain). It's a **scanned PDF** — no embedded text layer, so it
  needed OCR (`pymupdf` to render pages to images + `pytesseract` to read
  them). ~953 usable pages. Saved as `pharma_ocr_text.jsonl`
  (`{"page": i, "text": "..."}` per line).
- **Katzung's Basic & Clinical Pharmacology** — clean digital text, no OCR
  needed. Source: `cogbuji/medqa_corpus_en` on Hugging Face (a dataset
  repo using the old script-based loader, so it must be downloaded via
  `hf_hub_download` + manually unzipped, NOT `load_dataset()` directly —
  that fails with `RuntimeError: Dataset scripts are no longer supported`).
  File inside the zip: `textbooks/en/Pharmacology_Katzung.jsonl`
  (4,505 records, field name is `"text"`).
- **Combined total: ~5,458 records (~1.35M tokens)**.

### Stage 2 (Instruction Tuning) — Q&A pairs
Generated synthetically from the Stage 1 corpus using an LLM (see
"Synthetic data generation" below), rather than hand-written. Target was
~1,500-2,600 pairs (1,300 text chunks × ~2 pairs each, allowing for some
generation failures).

### Stage 3 (DPO) — preference pairs
Built from the Stage 2 output: the existing `output` field becomes
`chosen`; a deliberately worse answer to the same question is generated
to serve as `rejected`.

---

## 2. Synthetic data generation — what actually worked

**Groq (free tier) was tried first and hit two walls:**
1. `llama-3.1-70b-versatile` was decommissioned mid-project — Groq
   deprecates models with little notice. Check
   `https://api.groq.com/openai/v1/models` for what's currently live
   before assuming a model name from any past conversation or doc.
2. Even `openai/gpt-oss-20b` hit a **daily token limit (TPD)** on the free
   tier after ~220 chunks (200,000 tokens/day cap) — this stops the whole
   run, not just slows it, and resets only after ~24h.

**Switched to LangChain + OpenAI (`gpt-4o-mini`)** — reliable, and the
entire generation run (Q&A + DPO pairs, ~2,800 API calls total) cost
**under $1**. This is the recommended path going forward; don't burn more
time on Groq's free-tier limits for a run this size.

Both generation scripts (`qna_langchain_gpt.py`,
`create_dpo_data_langchain_gpt.py`) write output in **append mode** and
count existing lines in the output file to auto-resume — safe to stop and
restart without duplicating work or losing progress.

---

## 3. The 3-stage pipeline

```
Base model (Qwen1.5-1.8B, non-instruct)
    │  Stage 1: LoRA fine-tune on raw text (next-token prediction)
    │  Data: combined_pharma_text.jsonl (~5,458 records)
    ▼
merge_and_unload() → merged_for_stage2/  (new "base" for next stage)
    │  Stage 2: LoRA fine-tune on instruction/output pairs
    │  Data: generated_instruction_data.jsonl
    ▼
merge_and_unload() → merged_for_stage3/
    │  Stage 3: DPO fine-tune on chosen/rejected pairs
    │  Data: dpo_preference_data.jsonl
    ▼
Final model
```

**Core rule:** each stage's LoRA adapter (a delta on top of frozen
weights) must be **merged into the base weights before starting the next
stage** — never stack multiple un-merged LoRA adapters on the same base.
Each new stage's adapter needs to train against a "solid," already-updated
base, not float independently on top of the original one.

### Model choice

- **Qwen1.5-1.8B** (base, non-instruct) — the model actually used and
  successfully trained on. Chosen because it's small enough to iterate on
  fast, and `target_modules` for LoRA (`q_proj`, `k_proj`, `v_proj`,
  `o_proj` — Qwen doesn't fuse these like Phi does) are simple and
  well-documented.
- Attempted a switch to **Mistral-7B-v0.1** and then **Qwen2.5-7B** for a
  more impressive portfolio size, but hit reliability issues (OOM, and a
  `transformers==5.17.0` incompatibility that silently reinitialized all
  weights as MISSING during 4-bit quantized loading — confirmed by
  downgrading to `transformers==4.46.0`, which is now pinned in
  requirements). **Decision: stick with the smaller, already-working
  Qwen1.5-1.8B pipeline** rather than keep debugging a bigger model under
  time pressure — a complete pipeline on a small model beats a half-broken
  one on a big model, especially for a portfolio/interview story.

### Training config that worked (Stage 1, Qwen1.5-1.8B, RTX PRO 4500 32GB)

```python
training_arg = TrainingArguments(
    output_dir=model_dir,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,      # effective batch = 32
    learning_rate=2e-4,
    fp16=True,
    logging_steps=20,
    save_total_limit=1,
    weight_decay=0.05,
    save_strategy='epoch',
    warmup_steps=50,                    # use this, not warmup_ratio —
                                         # warmup_ratio caused a TypeError
                                         # on this transformers install
    lr_scheduler_type="cosine",
    report_to="none",
)
```

Actual run: completed in ~22-25 minutes (~0.38 it/s), loss dropped from
5.49 → ~1.34 and plateaued there — a healthy, expected curve, not
overfitting (small fluctuations of ~0.01 around a plateau are normal
noise, not a warning sign).

### LoRA config used

```python
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen/Llama-style
    # NOTE: Phi-family models use FUSED projections instead —
    # ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"] — different
    # naming if that model family is used later.
)
```

---

## 4. Evaluation — Stage 1 result (honest finding, not just a checkbox)

Ran a **perplexity comparison** (original base model vs. Stage-1
fine-tuned model) on 5 held-out pharma sentences:

```
                                    Original PPL   Fine-tuned PPL
ACE inhibitors mechanism                4.22            4.32
Beta-blockers mechanism                 6.73            8.27
Pharmacokinetics definition             4.94            8.96
Epilepsy characterization               4.10            6.07
b.i.d. abbreviation                    27.65           29.28
AVERAGE                                 9.53           11.38
```

**Result: fine-tuned model was WORSE (higher perplexity) than the
original base model on every single test sentence.** Likely causes:
Qwen's own pretraining already covered general medical text well, and the
~1.35M-token corpus (much of it noisy OCR output) was too small/noisy to
improve on that — it may have nudged the model toward its own training
distribution's quirks rather than genuinely improving pharma
understanding.

This is a legitimate, worth-reporting finding, not a failure to hide:
domain-adaptation on a small, partly-noisy corpus doesn't automatically
help, and perplexity is what actually caught it (generation samples alone
looked superficially fine — they used pharma vocabulary correctly even
while getting specific facts wrong, e.g. a hallucinated demographic claim
about epilepsy, and one generation randomly output Chinese characters for
the b.i.d. question).

**Decision:** proceed to Stage 2 anyway rather than re-do Stage 1 — the
instruction-tuning stage gives the model direct correct-answer supervision,
which may correct course more effectively than more raw-text pretraining
would. If Stage 2 results are also weak, revisit Stage 1 with a cleaner
corpus (drop the noisy OCR text, keep only Katzung) and a lower learning
rate.

---

## 5. Errors already hit and fixed (don't re-debug these)

| Error | Cause | Fix |
|---|---|---|
| `pip: No module named pip` | `pip` itself missing from a `uv`-created venv | `python -m ensurepip --upgrade` |
| `TrainingArguments.__init__() got an unexpected keyword argument 'warmup_ratio'` | transformers version mismatch | Use `warmup_steps=50` instead |
| `RuntimeError: Dataset scripts are no longer supported` (loading `cogbuji/medqa_corpus_en`) | HF deprecated script-based dataset loaders | Use `hf_hub_download` + manual `zipfile` extraction instead of `load_dataset()` |
| Groq `rate_limit_exceeded` / `tokens per day (TPD)` | Free-tier daily token cap hit | Switch to a lighter model, or better: switch to OpenAI (`gpt-4o-mini`, ~$1 total cost, no daily cap) |
| `SafetensorError: incomplete metadata, file not fully covered` | Model file transfer from RunPod to local was incomplete (partial download) | Verify file size matches expected (~3.6GB for a 1.8B bf16 model) before trying to load; re-transfer as a `.zip` rather than a raw multi-GB file |
| All model weights show as `MISSING` in a `[transformers] ... LOAD REPORT` during 4-bit load | `transformers==5.17.0` incompatible with `bitsandbytes` quantized loading path | Downgrade to `transformers==4.46.0` (now pinned in requirements.txt) |
| `CUDA out of memory` when loading a 7B model | Previous model (e.g. Qwen1.5-1.8B) still occupying GPU memory from an earlier cell | `del model_var; gc.collect(); torch.cuda.empty_cache()`, or just restart the kernel |
| `Pipeline`/`AutoModelForCausalLM` instantiated directly (`Pipeline(...)`, `AutoModelForCausalLM(...)`) | These are factory classes, not directly instantiable | Always use `.from_pretrained(...)` |

---

## 6. Next steps (where this project picks back up)

1. Re-provision a RunPod pod (previous one was terminated — all data on
   it is gone; local files like `pharma_ocr_text.jsonl` and
   `generated_instruction_data.jsonl`, if downloaded before termination,
   are the only remaining artifacts — verify what actually survived
   locally before redoing anything).
2. Re-run `combined_pharma_text.jsonl` creation (OCR text + Katzung) if
   the OCR'd file survived locally — OCR itself doesn't need to be
   redone, only the combine + Stage 1 retrain (~25 min).
3. Complete Stage 2 (instruction tuning) using
   `generated_instruction_data.jsonl` — same LoRA-then-merge pattern as
   Stage 1.
4. Generate DPO data (`create_dpo_data_langchain_gpt.py`) from the Stage 2
   output.
5. Run Stage 3 (DPO) using `trl`'s `DPOTrainer`.
6. Push each stage's merged model to Hugging Face Hub immediately after
   merging — don't rely on the pod surviving until the whole pipeline is
   done.