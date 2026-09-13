"""
create_dpo_data_langchain_gpt.py

Converts your instruction Q&A data (instruction + output) into DPO-ready
chosen/rejected pairs, using LangChain + OpenAI's gpt-4o-mini.

Input:  generated_instruction_data.jsonl   (instruction, output)
Output: dpo_preference_data.jsonl          (prompt, chosen, rejected)

Setup:
    pip install langchain langchain-openai --break-system-packages
"""
from dotenv import load_dotenv
load_dotenv()
import json
import time
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


GPT_MODEL = "gpt-4o-mini"

INPUT_JSONL = "generated_instruction_data.jsonl"
OUTPUT_JSONL = "dpo_preference_data.jsonl"

SLEEP_BETWEEN_CALLS = 0.3



llm = ChatOpenAI(model=GPT_MODEL, temperature=0.9, max_tokens=256, timeout=20)


# ---------------------------------------------------------------------------
# Prompt builder — asks for a deliberately worse answer to the same question
# ---------------------------------------------------------------------------
def build_rejected_messages(instruction, good_answer):
    system_msg = (
        "You are helping build a training dataset that teaches a model to prefer "
        "good answers over bad ones. Given a question and a correct, high-quality "
        "answer, write a WORSE version of the answer to the SAME question — "
        "vague, incomplete, or slightly inaccurate, as a poorly-trained model might "
        "produce. It should still be plausible-sounding, not nonsensical. "
        "Respond with ONLY the worse answer text, nothing else — no labels, no quotes."
    )
    user_msg = (
        f"Question: {instruction}\n\n"
        f"Correct answer (for reference, do not copy): {good_answer}\n\n"
        f"Write a noticeably worse answer to this same question."
    )
    return [SystemMessage(content=system_msg), HumanMessage(content=user_msg)]


# ---------------------------------------------------------------------------
# One LLM call, with retry
# ---------------------------------------------------------------------------
def generate_rejected(instruction, good_answer, max_retries=3):
    messages = build_rejected_messages(instruction, good_answer)

    for attempt in range(max_retries):
        try:
            response = llm.invoke(messages)
            return response.content.strip()

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
        qa_pairs = [json.loads(line) for line in f_in]

    print(f"Loaded {len(qa_pairs)} instruction/output pairs")

    already_done = count_existing_lines(OUTPUT_JSONL)
    if already_done > 0:
        print(f"Found {already_done} existing DPO pairs. Resuming from there.")
        qa_pairs = qa_pairs[already_done:]

    print(f"Remaining pairs to process: {len(qa_pairs)}")
    print(f"Estimated time: ~{len(qa_pairs) * (SLEEP_BETWEEN_CALLS + 1.2) / 60:.0f} minutes")

    total_generated = already_done
    start_time = time.time()

    with open(OUTPUT_JSONL, "a", encoding="utf-8") as f_out:
        for i, row in enumerate(qa_pairs):
            instruction = row.get("instruction", "").strip()
            chosen = row.get("output", "").strip()

            if not instruction or not chosen:
                continue

            rejected = generate_rejected(instruction, chosen)

            if rejected:
                f_out.write(json.dumps({
                    "prompt": instruction,
                    "chosen": chosen,
                    "rejected": rejected
                }, ensure_ascii=False) + "\n")
                total_generated += 1
            else:
                print(f"  Skipping pair {i} after repeated failures.")

            f_out.flush()

            if i % 10 == 0:
                elapsed_min = (time.time() - start_time) / 60
                print(f"Processed {i}/{len(qa_pairs)} — "
                      f"{total_generated} total DPO pairs — {elapsed_min:.1f} min elapsed")

            time.sleep(SLEEP_BETWEEN_CALLS)

    total_min = (time.time() - start_time) / 60
    print(f"\nDone in {total_min:.1f} minutes. "
          f"{total_generated} total DPO pairs saved to {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()