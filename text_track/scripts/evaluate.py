import json
import os
import re
import argparse
import time
from dotenv import load_dotenv

load_dotenv()

import anthropic
from openai import OpenAI

# ---------------- CONFIGURATION ---------------- #
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "..", "data", "dataset.jsonl")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "..", "data", "evaluation_results_3pass.jsonl")

# Delay (seconds) inserted after every model call to respect provider rate limits.
REQUEST_DELAY = 1.0

# Model IDs
# NOTE: This is the current Claude Sonnet. If your experiment specifically requires the
# frozen "claude-3-5-sonnet-20240620" checkpoint for reproducibility, change this one line.
CLAUDE_MODEL = "claude-sonnet-4-6"
# HuggingFace router model IDs (OpenAI-compatible endpoint). The provider suffix after ':'
# pins which serverless provider serves the model.
LLAMA_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
# --- SeaLLM disabled ---
# featherless-ai is the ONLY HF provider serving this model, and it hard-caps output at
# ~55 tokens regardless of max_tokens (finish_reason='length'), so SeaLLM never completes
# its reasoning or reaches the <answer> tag. Re-enable this line AND the MODELS entry below
# if a higher-limit provider (or a local deployment) becomes available.
# SEALLM_MODEL = "SeaLLMs/SeaLLM-7B-v2.5:featherless-ai"

# Initialize Clients (both SDKs handle their own retry/backoff on transient errors)
anthropic_client = anthropic.Anthropic(max_retries=3)  # Uses ANTHROPIC_API_KEY
hf_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ.get("HF_TOKEN"),
    max_retries=3,
    timeout=60.0,
)

# ---------------- 3-PASS SYSTEM PROMPTS ---------------- #
# Shared answer-format rule appended to every pass so grading is automatic and parseable.
_ANSWER_RULE = (
    "After your reasoning, end your response with the final answer enclosed in "
    "<answer></answer> tags. The tags must contain ONLY the final value — a single "
    "number, letter, or short word (such as a multiple-choice option) — with no units, "
    "no explanation, and no extra text.\n"
    "Examples: <answer>109</answer>, <answer>A</answer>, <answer>Yes</answer>."
)

# Pass 1 — English Baseline: English question -> English CoT.
PASS1_SYSTEM = (
    "You are an expert problem solver. Carefully read the problem the user provides and "
    "solve it. Show your complete step-by-step reasoning in English.\n" + _ANSWER_RULE
)

# Pass 2 — Native Track: Ilokano question -> reasoning ENTIRELY in Ilokano.
PASS2_SYSTEM = (
    "You are an expert problem solver who is fluent in Ilokano. The user will give you a "
    "problem written in Ilokano. Solve it and show your complete step-by-step reasoning "
    "ENTIRELY IN ILOKANO. Do not write your reasoning in English.\n" + _ANSWER_RULE
)

# Pass 3 — English Pivot Track: Ilokano question -> translate to English first, then reason.
PASS3_SYSTEM = (
    "You are an expert problem solver and translator. The user will give you a problem "
    "written in Ilokano. Follow these steps in order:\n"
    "1. First, translate the problem from Ilokano into English.\n"
    "2. Then solve it, showing your complete step-by-step reasoning in English.\n"
    + _ANSWER_RULE
)

# Each pass declares its system prompt and which language's question to feed in.
PASSES = [
    {"key": "pass1_english_baseline", "system": PASS1_SYSTEM, "lang_input": "english"},
    {"key": "pass2_native_ilokano",   "system": PASS2_SYSTEM, "lang_input": "ilokano"},
    {"key": "pass3_english_pivot",     "system": PASS3_SYSTEM, "lang_input": "ilokano"},
]

# ---------------- HELPER FUNCTIONS ---------------- #
def parse_expected_answer(answer_text):
    """Extract the final answer. GSM8K answers end with '#### <value>'; bbh/mmlu answers
    are already the bare final value, so we return them as-is."""
    match = re.search(r'####\s*(.+)', answer_text)
    if match:
        return match.group(1).strip()
    return answer_text.strip()

def extract_answer(text):
    """Extract the text inside the LAST well-formed <answer>...</answer> tag pair.
    The negative lookahead prevents a capture from spanning a stray <answer> opener that
    a model may echo from the instructions (e.g. 'enclosed in <answer> tags ... <answer>109</answer>');
    taking the last non-empty match then picks the real final answer over any earlier echo."""
    matches = re.findall(r'<answer>((?:(?!<answer>).)*?)</answer>', text, re.IGNORECASE | re.DOTALL)
    for m in reversed(matches):
        if m.strip():
            return m.strip()
    return None

def normalize_answer(text):
    """Normalize for comparison: lowercase, drop commas, strip surrounding
    brackets/parens/punctuation, and remove common answer prefixes."""
    if text is None:
        return None
    t = text.strip().lower().replace(",", "")
    t = t.strip("().[]:;")
    for prefix in ("the answer is ", "final answer ", "answer ", "option "):
        if t.startswith(prefix):
            t = t[len(prefix):]
    return t.strip("().[]:; ")

def _first_number(text):
    """Return the first numeric token in text (as a string), or None."""
    if text is None:
        return None
    m = re.search(r'-?\d+(?:\.\d+)?', text.replace(",", ""))
    return m.group(0) if m else None

def _matches(extracted, gold):
    """True if extracted answer matches a single gold answer (string or numeric)."""
    if extracted is None or gold is None:
        return False
    if normalize_answer(extracted) == normalize_answer(gold):
        return True
    # Numeric fallback so "109 years" or "109." still matches gold "109".
    ne, ng = _first_number(extracted), _first_number(gold)
    if ne is not None and ng is not None:
        try:
            return float(ne) == float(ng)
        except ValueError:
            return False
    return False

def grade(extracted, gold_en, gold_ilo):
    """Correct if the extracted answer matches EITHER the English or the Ilokano gold.
    This handles the native pass where e.g. a correct 'Yes' may be answered as 'Wen'."""
    return _matches(extracted, gold_en) or _matches(extracted, gold_ilo)

def query_claude(system_prompt, user_prompt):
    response = anthropic_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2048,
        temperature=0.0,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    # The safety classifier can decline a prompt (HTTP 200, stop_reason='refusal')
    # with an EMPTY content array — indexing content[0] then raises IndexError.
    # Concatenate all text blocks; if there are none, surface the stop_reason so the
    # caller records a clean "[REFUSAL]" marker instead of crashing.
    text = "".join(getattr(b, "text", "") for b in response.content if b.type == "text")
    if not text:
        return f"[{(response.stop_reason or 'EMPTY').upper()}] Claude returned no text content."
    return text

def query_openai_model(model_id, system_prompt, user_prompt, supports_system=True):
    """Query an OpenAI-compatible chat model (HF router for Llama / SeaLLM).
    Some providers (e.g. featherless-ai serving SeaLLM) reject a 'system' role, so we
    fold the system prompt into the user message when supports_system is False."""
    if supports_system:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    else:
        messages = [
            {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"},
        ]
    response = hf_client.chat.completions.create(
        model=model_id,
        messages=messages,
        max_tokens=2048,
        temperature=0.0,
    )
    return response.choices[0].message.content

# Model registry: each entry knows how to run a (system, user) prompt pair.
MODELS = [
    {"key": "claude_sonnet_4_6", "name": "Claude Sonnet 4.6",
     "fn": lambda system, user: query_claude(system, user)},
    {"key": "llama_3_8b", "name": "Llama 3 8B",
     "fn": lambda system, user: query_openai_model(LLAMA_MODEL, system, user)},
    # --- SeaLLM disabled (see note at SEALLM_MODEL above). Uncomment to restore. ---
    # {"key": "seallm_7b", "name": "SeaLLM 7B",
    #  "fn": lambda system, user: query_openai_model(SEALLM_MODEL, system, user, supports_system=False)},
]

# ---------------- MAIN PIPELINE ---------------- #
def evaluate_dataset(limit=None):
    processed_count = 0
    accuracy = {
        m["key"]: {p["key"]: {"correct": 0, "total": 0} for p in PASSES}
        for m in MODELS
    }

    with open(INPUT_FILE, 'r', encoding='utf-8') as infile, \
         open(OUTPUT_FILE, 'a', encoding='utf-8') as outfile:

        for line in infile:
            if not line.strip():
                continue

            if limit and processed_count >= limit:
                print(f"\n--- Reached the limit of {limit} rows. Stopping. ---")
                break

            data = json.loads(line)
            english_question   = data.get("question_en", "")
            ilokano_question   = data.get("question_ilo", "")
            expected_answer    = parse_expected_answer(data.get("answer_en", ""))
            ilokano_answer_ref = parse_expected_answer(data.get("answer_ilo", ""))

            # Per-language user prompts reused across the relevant passes.
            prompts = {
                "english": f"Problem:\n{english_question}",
                "ilokano": f"Problem:\n{ilokano_question}",
            }

            print(f"\nEvaluating Q{processed_count + 1} [{data.get('id', '?')}] "
                  f"(expected: {expected_answer})...")

            row_results = {
                "id":                 data.get("id", ""),
                "source":             data.get("source", ""),
                "english_question":   english_question,
                "ilokano_question":   ilokano_question,
                "expected_answer":    expected_answer,
                "ilokano_answer_ref": ilokano_answer_ref,
                "evaluations": {},
            }

            for model in MODELS:
                print(f"  -> Querying {model['name']}...")
                model_block = {}
                for p in PASSES:
                    user_prompt = prompts[p["lang_input"]]
                    try:
                        raw = model["fn"](p["system"], user_prompt)
                        extracted = extract_answer(raw)
                        is_correct = grade(extracted, expected_answer, ilokano_answer_ref)
                        model_block[p["key"]] = {
                            "chain_of_thought": raw,
                            "extracted_answer": extracted,
                            "is_correct":       is_correct,
                        }
                    except Exception as e:
                        # Catch + log API timeouts, connection errors, bad requests, etc.
                        print(f"     !! {p['key']} failed: {type(e).__name__}: {e}")
                        model_block[p["key"]] = {
                            "chain_of_thought": f"Error: {type(e).__name__}: {e}",
                            "extracted_answer": None,
                            "is_correct":       False,
                        }
                    time.sleep(REQUEST_DELAY)
                row_results["evaluations"][model["key"]] = model_block

            # Update running accuracy (per model, per pass).
            for model in MODELS:
                for p in PASSES:
                    cell = accuracy[model["key"]][p["key"]]
                    cell["total"] += 1
                    if row_results["evaluations"][model["key"]][p["key"]]["is_correct"]:
                        cell["correct"] += 1

            outfile.write(json.dumps(row_results) + '\n')
            outfile.flush()
            processed_count += 1

    # ---------------- ACCURACY SUMMARY ---------------- #
    print(f"\n{'=' * 60}")
    print(f"EVALUATION COMPLETE — {processed_count} questions")
    print(f"{'=' * 60}")
    for model in MODELS:
        print(f"\n{model['name']} ({model['key']}):")
        for p in PASSES:
            c = accuracy[model["key"]][p["key"]]
            pct = (c["correct"] / c["total"] * 100) if c["total"] else 0
            print(f"  {p['key']:24s}: {c['correct']}/{c['total']} ({pct:.1f}%)")
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3-pass cross-lingual LLM evaluation.")
    parser.add_argument("--limit", type=int, help="Limit the number of questions to evaluate.")
    args = parser.parse_args()

    evaluate_dataset(limit=args.limit)
