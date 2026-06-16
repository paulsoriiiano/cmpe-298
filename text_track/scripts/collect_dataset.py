import os
import json
import random
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from openai import OpenAI

# Initialize the Translation API (Assuming GPT-4o for the first-pass translation)
# The RA will need to set their OPENAI_API_KEY as an environment variable

# 1. Load the .env file into the system's environment memory
load_dotenv() 

# 2. Initialize the client
# Because the variable is named OPENAI_API_KEY, the library 
# automatically finds it. You don't need to pass api_key="..."

# # Verification (Optional: delete this after testing)
# if os.getenv("OPENAI_API_KEY"):
#     print("API Key loaded successfully.")
# else:
#     print("Error: API Key not found. Check your .env file.")

# Access the token
token = os.getenv("HF_TOKEN")

# Use it to authenticate
if token:
    login(token=token)
 
# Initialize the OpenAI client
client = OpenAI()


def translate_to_ilokano(english_text):
    """
    Drafts an Ilokano translation using an LLM.
    The RA should log these outputs for your human-in-the-loop review.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.0, # Zero temperature to avoid creative variations
            messages=[
                {"role": "system", "content": "You are an expert translator. Translate the following English text into grammatically precise, native-sounding Ilokano. Preserve all mathematical numbers, logical structures, and constraints exactly as they are."},
                {"role": "user", "content": english_text}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Translation Error: {e}")
        return ""

def build_dataset():
    final_dataset = []
    
    print("1. Extracting GSM8K...")
    gsm8k = load_dataset("openai/gsm8k", "main", split="test")
    # Sample 400 items
    gsm8k_sample = gsm8k.shuffle(seed=42).select(range(400))
    for i, row in enumerate(gsm8k_sample):
        final_dataset.append({
            "id": f"gsm8k_{i}",
            "source": "gsm8k",
            "question_en": row["question"],
            "answer_en": row["answer"], # Note: GSM8K includes the CoT in the answer string
            "question_ilo": "", # To be filled
            "answer_ilo": ""    # To be filled
        })

    print("2. Extracting BigBench Hard (Logical Deduction & Causal Judgement)...")
    bbh_count = 0
    bbh_target = 300
    
    # Load logical_deduction_five_objects
    bbh_logic = load_dataset("maveriq/bigbenchhard", "logical_deduction_five_objects", split="train")
    bbh_logic_sample = bbh_logic.shuffle(seed=42)
    for i, row in enumerate(bbh_logic_sample):
        if bbh_count >= bbh_target:
            break
        final_dataset.append({
            "id": f"bbh_logic_{bbh_count}",
            "source": "bbh_logical_deduction",
            "question_en": row["input"],
            "answer_en": row["target"],
            "question_ilo": "",
            "answer_ilo": ""
        })
        bbh_count += 1
    
    # If we need more, load causal_judgement
    if bbh_count < bbh_target:
        print(f"  -> Got {bbh_count} from logical_deduction, adding from causal_judgement...")
        bbh_causal = load_dataset("maveriq/bigbenchhard", "causal_judgement", split="train")
        bbh_causal_sample = bbh_causal.shuffle(seed=42)
        for i, row in enumerate(bbh_causal_sample):
            if bbh_count >= bbh_target:
                break
            final_dataset.append({
                "id": f"bbh_causal_{bbh_count}",
                "source": "bbh_causal_judgement",
                "question_en": row["input"],
                "answer_en": row["target"],
                "question_ilo": "",
                "answer_ilo": ""
            })
            bbh_count += 1

    print("3. Extracting MMLU (Formal Logic & Conceptual Physics)...")
    mmlu_count = 0
    mmlu_target = 300
    
    # Load formal_logic
    mmlu_logic = load_dataset("cais/mmlu", "formal_logic", split="test")
    mmlu_logic_sample = mmlu_logic.shuffle(seed=42)
    for i, row in enumerate(mmlu_logic_sample):
        if mmlu_count >= mmlu_target:
            break
        # MMLU has multiple choices, so we combine the prompt and options into one string for translation
        options_string = f"\nA) {row['choices'][0]}\nB) {row['choices'][1]}\nC) {row['choices'][2]}\nD) {row['choices'][3]}"
        full_question = row["question"] + options_string
        
        final_dataset.append({
            "id": f"mmlu_logic_{mmlu_count}",
            "source": "mmlu_formal_logic",
            "question_en": full_question,
            "answer_en": str(row["answer"]),
            "question_ilo": "",
            "answer_ilo": ""
        })
        mmlu_count += 1
    
    # If we need more, load conceptual_physics
    if mmlu_count < mmlu_target:
        print(f"  -> Got {mmlu_count} from formal_logic, adding from conceptual_physics...")
        mmlu_physics = load_dataset("cais/mmlu", "conceptual_physics", split="test")
        mmlu_physics_sample = mmlu_physics.shuffle(seed=42)
        for i, row in enumerate(mmlu_physics_sample):
            if mmlu_count >= mmlu_target:
                break
            # MMLU has multiple choices, so we combine the prompt and options into one string for translation
            options_string = f"\nA) {row['choices'][0]}\nB) {row['choices'][1]}\nC) {row['choices'][2]}\nD) {row['choices'][3]}"
            full_question = row["question"] + options_string
            
            final_dataset.append({
                "id": f"mmlu_physics_{mmlu_count}",
                "source": "mmlu_conceptual_physics",
                "question_en": full_question,
                "answer_en": str(row["answer"]),
                "question_ilo": "",
                "answer_ilo": ""
            })
            mmlu_count += 1

    print(f"Total dataset size before translation: {len(final_dataset)} items.")
    return final_dataset


def load_completed_ids(output_file):
    completed = set()
    if not os.path.exists(output_file):
        return completed
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    if 'id' in item:
                        completed.add(item['id'])
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Warning: could not read existing output file: {e}")
    return completed


def process_translations_and_save(dataset, output_file="ilokano_reasoning_benchmark_draft.jsonl", resume=True):
    print("Starting Machine Translation Phase (This will take a while)...")

    completed_ids = load_completed_ids(output_file) if resume else set()
    mode = 'a' if resume and completed_ids else 'w'
    skipped = 0
    translated = 0

    with open(output_file, mode, encoding='utf-8') as f:
        for item in dataset:
            if item['id'] in completed_ids:
                skipped += 1
                continue

            print(f"Translating {item['id']}...")
            item["question_ilo"] = translate_to_ilokano(item["question_en"])

            if item["source"] == "gsm8k":
                item["answer_ilo"] = translate_to_ilokano(item["answer_en"])
            else:
                item["answer_ilo"] = item["answer_en"]

            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            translated += 1

    print(f"Success! Dataset saved to {output_file}")
    print(f"Skipped {skipped} already-translated items.")
    print(f"Translated {translated} new items.")


def save_raw_dataset(dataset, output_file="ilokano_reasoning_benchmark_raw.jsonl"):
    """Save the raw dataset before any translations."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Raw dataset saved to {output_file}")


# --- Execute the Pipeline ---
if __name__ == "__main__":
    raw_dataset = build_dataset()
    save_raw_dataset(raw_dataset)
    process_translations_and_save(raw_dataset, resume=True)