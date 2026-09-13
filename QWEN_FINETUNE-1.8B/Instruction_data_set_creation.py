"""
qna_langchain_gpt.py

Generates instruction-style Q&A pairs from raw pharma text chunks using
LangChain + OpenAI's gpt-4o-mini. Replaces the Groq-based version to avoid
free-tier daily token-limit interruptions — gpt-4o-mini is cheap enough
(~$0.35-0.40 for this whole run) that reliability is worth the small cost.

Setup:
    pip install langchain langchain-openai --break-system-packages

    Get an API key from https://platform.openai.com/api-keys
    (requires adding a small prepaid balance, e.g. $5)

Run:
    python qna_langchain_gpt.py
"""

import json
import re
import random
import time
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GPT_MODEL = "gpt-4o-mini"

INPUT_JSONL = r"C:\Users\csc\Desktop\Pharma_fine_tune\combined_pharma_text.jsonl"
OUTPUT_JSONL = "generated_instruction_data.jsonl"   # append-safe, resumable

QA_PER_CHUNK = 2
MAX_CHUNKS = 1300
MIN_CHUNK_LENGTH = 200
RANDOM_SEED = 42

SLEEP_BETWEEN_CALLS = 0.3   # OpenAI's rate limits are generous — light throttle is enough

# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

llm = ChatOpenAI(model=GPT_MODEL, temperature=0.7, max_tokens=512, timeout=20)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------
def build_messages(text_chunk, n_pairs=2):
    system_msg = (
        "You are a dataset-generation assistant for pharmacology education. "
        "Given a passage from a pharmacology textbook, generate factual, specific "
        "instruction-style question-answer pairs based ONLY on the passage's content. "
        "Do not invent facts not present in the passage. "
        "Respond with ONLY a JSON array, no extra text, no markdown fences."
    )
    user_msg = (
        f"Passage:\n\"\"\"\n{text_chunk}\n\"\"\"\n\n"
        f"Generate exactly {n_pairs} question-answer pairs from this passage. "
        f'Format: [{{"instruction": "...", "output": "..."}}, ...]'
    )
    return [SystemMessage(content=system_msg), HumanMessage(content=user_msg)]


# ---------------------------------------------------------------------------
# JSON extraction — robust to stray text around the JSON array
# ---------------------------------------------------------------------------
def extract_json_array(raw_output):
    match = re.search(r"\[.*\]", raw_output, re.DOTALL)
    if not match:
        return []
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return []


# ---------------------------------------------------------------------------
# One LLM call per chunk, with retry
# ---------------------------------------------------------------------------
def generate_qa_for_chunk(text_chunk, n_pairs=2, max_retries=3):
    messages = build_messages(text_chunk, n_pairs)

    for attempt in range(max_retries):
        try:
            response = llm.invoke(messages)
            return extract_json_array(response.content)

        except Exception as e:
            error_str = str(e)
            if "rate_limit" in error_str.lower() or "429" in error_str:
                print(f"  Rate limit hit. Waiting 30s before retry {attempt + 1}/{max_retries}...")
                time.sleep(30)
            else:
                print(f"  Retry {attempt + 1}/{max_retries} after error: {e}")
                time.sleep(5 * (attempt + 1))

    return None


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------
def count_existing_lines(path):
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    with open(INPUT_JSONL, "r", encoding="utf-8") as f_in:
        all_chunks = []
        for line in f_in:
            row = json.loads(line)
            text_chunk = row.get("text", "").strip()
            if len(text_chunk) >= MIN_CHUNK_LENGTH:
                all_chunks.append(text_chunk)

    print(f"Total usable chunks available: {len(all_chunks)}")

    random.seed(RANDOM_SEED)
    random.shuffle(all_chunks)
    selected_chunks = all_chunks[:MAX_CHUNKS]

    existing_pairs_count = count_existing_lines(OUTPUT_JSONL)
    approx_chunks_done = int(existing_pairs_count / QA_PER_CHUNK)

    if approx_chunks_done > 0:
        print(f"Found {existing_pairs_count} existing Q&A pairs "
              f"(~{approx_chunks_done} chunks already done). Resuming from there.")
        selected_chunks = selected_chunks[approx_chunks_done:]

    print(f"Chunks remaining to process: {len(selected_chunks)}")
    print(f"Estimated time: ~{len(selected_chunks) * (SLEEP_BETWEEN_CALLS + 1.5) / 60:.0f} minutes")

    total_generated = existing_pairs_count
    start_time = time.time()

    with open(OUTPUT_JSONL, "a", encoding="utf-8") as f_out:
        for i, text_chunk in enumerate(selected_chunks):
            qa_pairs = generate_qa_for_chunk(text_chunk, n_pairs=QA_PER_CHUNK)

            if qa_pairs is None:
                print(f"  Skipping chunk {i} after repeated failures.")
                continue

            for qa in qa_pairs:
                if "instruction" in qa and "output" in qa:
                    f_out.write(json.dumps(qa, ensure_ascii=False) + "\n")
                    total_generated += 1

            f_out.flush()

            if i % 10 == 0:
                elapsed_min = (time.time() - start_time) / 60
                print(f"Processed {i}/{len(selected_chunks)} chunks — "
                      f"{total_generated} total Q&A pairs — {elapsed_min:.1f} min elapsed")

            time.sleep(SLEEP_BETWEEN_CALLS)

    total_min = (time.time() - start_time) / 60
    print(f"\nDone in {total_min:.1f} minutes. "
          f"{total_generated} total Q&A pairs saved to {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()