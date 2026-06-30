import os
import json
import time
import argparse
import anthropic

from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Initialize the client (ensure your ANTHROPIC_API_KEY is set in your environment variables)
client = anthropic.Anthropic()

# Configuration
INPUT_FILE = "not_translated.json" # Read from raw dataset with id and source
OUTPUT_FILE = "retranslated_dataset_with_metadata.jsonl" # Keeping output as JSONL for safety
MODEL_NAME = "claude-sonnet-4-5-20250929" 

# The Upgraded System Prompt
SYSTEM_PROMPT = """You are an expert Ilokano linguist and mathematician. Your task is to translate English math word problems and their answers into grammatically correct Ilokano while STRICTLY PRESERVING the exact mathematical logic.

CRITICAL CONSTRAINTS:
1. NO TAGALOG BLEED-THROUGH: Do not use Tagalog words under any circumstance. Pay special attention to colors (e.g., use 'asul' for blue, 'puraw' for white, 'nalabaga' for red).
2. PRECISE COMPARATIVES: You must accurately translate mathematical operators. Use proper morphological reduplication for comparatives (e.g., use 'basbassit' for 'less/fewer', do not just use the root word 'bassit').
3. DIMENSIONAL VERBS: Translate English catch-all verbs strictly based on their physical context. Do not translate 'increase [pressure/amount]' as 'mapapardas' (which means to speed up).
4. UNTRANSLATABLE TERMS: If a specific math term (like 'probability', 'x-axis', or 'fraction') lacks a widely accepted Ilokano equivalent, keep the English term rather than inventing a confusing word.
"""

def process_dataset(limit=None, dry_run=False):
    # Step 1: Read the entire JSON file into a Python list
    with open(INPUT_FILE, 'r', encoding='utf-8') as infile:
        dataset = json.load(infile)
        
    processed_count = 0
    
    # Open the output file safely (append mode if file exists, create if not)
    outfile = None
    if not dry_run:
        outfile = open(OUTPUT_FILE, 'a', encoding='utf-8')
        
    try:
        # Step 2: Iterate over the parsed JSON list
        for data in dataset:
            
            # Check if we hit the limit
            if limit and processed_count >= limit:
                print(f"\n--- Reached the limit of {limit} rows. Stopping. ---")
                break
                
            # Isolate only the English Q&A and metadata
            id_value = data.get("id", "")
            source_value = data.get("source", "")
            english_question = data.get("question_en", "")
            english_answer = data.get("answer_en", "")
            
            prompt = f"Translate the following problem and answer into Ilokano.\n\nQuestion: {english_question}\nAnswer: {english_answer}"

            # Step 3: Handle Dry Run
            if dry_run:
                print(f"[DRY RUN {processed_count + 1}] Prepared to translate: {english_question[:60]}...")
                processed_count += 1
                continue

            # Step 4: Query Claude with the Anthropic API syntax
            try:
                # Translate question
                question_prompt = f"Translate the following math problem into Ilokano:\n\nQuestion: {english_question}"
                
                response_question = client.messages.create(
                    model=MODEL_NAME,
                    max_tokens=512,
                    temperature=0.2,
                    system=SYSTEM_PROMPT,
                    messages=[
                        {"role": "user", "content": question_prompt}
                    ]
                )
                
                question_ilo = response_question.content[0].text
                
                # Translate answer
                answer_prompt = f"Translate the following math answer into Ilokano:\n\nAnswer: {english_answer}"
                
                response_answer = client.messages.create(
                    model=MODEL_NAME,
                    max_tokens=512,
                    temperature=0.2,
                    system=SYSTEM_PROMPT,
                    messages=[
                        {"role": "user", "content": answer_prompt}
                    ]
                )
                
                answer_ilo = response_answer.content[0].text
                
                # Create the clean dictionary structure
                new_row = {
                    "id": id_value,
                    "source": source_value,
                    "question_en": english_question,
                    "answer_en": english_answer,
                    "question_ilo": question_ilo,
                    "answer_ilo": answer_ilo
                }
                
                # Write to the new file instantly (as JSONL)
                outfile.write(json.dumps(new_row) + '\n')
                
                # Small sleep to respect API rate limits
                time.sleep(0.5)
                print(f"Successfully re-translated [{processed_count + 1}]: {english_question[:50]}...")

            except Exception as e:
                print(f"API Error on question [{processed_count + 1}]: {english_question[:50]}... | Error: {e}")
            
            processed_count += 1

    finally:
        if outfile:
            outfile.close()
            
    if dry_run:
        print("\nDry run complete. No API calls were made and no files were written.")
    else:
        print(f"\nTranslation complete. Clean data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate dataset using Claude Sonnet 4.5.")
    
    # Add the command line flags
    parser.add_argument("--limit", type=int, help="Limit the number of rows to process.")
    parser.add_argument("--dry", action="store_true", help="Perform a dry run without calling the API or saving data.")
    
    args = parser.parse_args()
    
    # Pass the flags into the function
    process_dataset(limit=args.limit, dry_run=args.dry)