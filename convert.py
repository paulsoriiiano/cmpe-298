import pandas as pd
import json

# Define the audit fields you want to see in Excel
AUDIT_COLUMNS = [
    "linguistic_drift",
    "tagalog_interference",
    "focus_system_error",
    "lexical_hallucination",
    "naturalness_score",
    "researcher_notes"
]

def json_to_csv_with_bootstrap(json_input, csv_output):
    """
    Reads a standard JSON file (list of dicts) and converts to CSV.
    """
    with open(json_input, 'r', encoding='utf-8') as f:
        # We load the WHOLE file at once using json.load()
        data = json.load(f)
    
    # Flatten the data (handles nested audit_metadata if it exists)
    df = pd.json_normalize(data)
    
    # Add audit columns if they are missing
    for col in AUDIT_COLUMNS:
        full_col_name = f"audit_metadata.{col}"
        if full_col_name not in df.columns:
            df[full_col_name] = ""
    
    df.to_csv(csv_output, index=False, encoding='utf-8-sig')
    print(f"✅ Successfully exported to {csv_output}.")

def csv_to_json(csv_input, json_output):
    """
    Converts edited CSV back to a standard JSON list.
    """
    df = pd.read_csv(csv_input).fillna("")
    
    final_records = []
    for _, row in df.iterrows():
        record = row.to_dict()
        metadata = {}
        keys_to_remove = []
        
        for key, value in record.items():
            if key.startswith('audit_metadata.'):
                clean_key = key.replace('audit_metadata.', '')
                # Boolean conversion logic
                if str(value).lower() in ['true', '1', '1.0']:
                    metadata[clean_key] = True
                elif str(value).lower() in ['false', '0', '0.0']:
                    metadata[clean_key] = False
                else:
                    metadata[clean_key] = value
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            record.pop(key)
            
        record['audit_metadata'] = metadata
        final_records.append(record)
    
    # Save as a standard JSON list
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(final_records, f, ensure_ascii=False, indent=4)
            
    print(f"🚀 Standard JSON reconstructed at {json_output}")

# --- EXECUTION ---
json_to_csv_with_bootstrap("ilokano_reasoning_benchmark_draft.json", "audit_sheet.csv")
# csv_to_json("audit_sheet.csv", "verified_dataset.json")