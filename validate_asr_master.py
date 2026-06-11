import re
from pathlib import Path
import pandas as pd

EXPECTED = {
    "MAT": [25,23,17,25,48,34,29,34,38,42,30,50,58,36,39,28,27,35,30,34,46,46,39,51,46,75,66,20],
    "MRK": [45,28,35,41,43,56,37,38,50,52,33,44,37,72,47,20],
    "LUK": [80,52,38,44,39,49,50,56,62,42,54,59,35,35,32,31,37,43,48,47,38,71,56,53],
}

OMITTED = {
    "en": {
        "MAT_12_47", "MAT_17_21", "MAT_18_11", "MAT_23_14",
        "MRK_07_16", "MRK_09_44", "MRK_09_46", "MRK_11_26", "MRK_15_28",
        "LUK_17_36", "LUK_23_17",
    },
    "ilo": {
        "MAT_12_47", "MAT_17_21", "MAT_18_11", "MAT_23_14",
        "MRK_07_16", "MRK_09_44", "MRK_09_46", "MRK_11_26", "MRK_15_28",
        "LUK_17_36", "LUK_23_17",
        "LUK_22_43", "LUK_22_44", "LUK_24_12", "LUK_24_40",
    },
}

df = pd.read_csv("asr_evaluation_master.csv")
errors = []

def parse_verse_id(vid):
    m = re.match(r"^([A-Z]{3})_(\d{2})_(\d{2})(?:_(\d{2}))?$", str(vid))
    if not m:
        return None
    book = m.group(1)
    chapter = int(m.group(2))
    start = int(m.group(3))
    end = int(m.group(4)) if m.group(4) else start
    return book, chapter, start, end

def omitted_for_chapter(lang, book, chapter):
    omitted = set()

    for vid in OMITTED.get(lang, set()):
        parsed = parse_verse_id(vid)
        if parsed is None:
            continue

        b, ch, start, end = parsed
        if b == book and ch == chapter:
            omitted.update(range(start, end + 1))

    return omitted

for lang in ["en", "ilo"]:
    lang_df = df[df["language"] == lang]

    for book, chapter_counts in EXPECTED.items():
        for chapter, expected_last_verse in enumerate(chapter_counts, start=1):
            rows = lang_df[
                lang_df["verse_id"].str.startswith(f"{book}_{chapter:02d}_", na=False)
            ]

            covered = set()

            for vid in rows["verse_id"]:
                parsed = parse_verse_id(vid)
                if parsed is None:
                    errors.append(f"Bad verse_id format: {vid}")
                    continue

                _, _, start, end = parsed
                covered.update(range(start, end + 1))

            expected = set(range(1, expected_last_verse + 1))
            expected -= omitted_for_chapter(lang, book, chapter)

            missing = sorted(expected - covered)
            extra = sorted(covered - set(range(1, expected_last_verse + 1)))

            if missing:
                errors.append(f"{lang} {book}_{chapter:02d} missing verses: {missing}")
            if extra:
                errors.append(f"{lang} {book}_{chapter:02d} has impossible verses: {extra}")

bad_markers = df[df["raw_text"].astype(str).str.contains(r"※|†|\*", regex=True, na=False)]
if len(bad_markers):
    errors.append(f"{len(bad_markers)} rows still contain footnote markers.")

empty = df[
    df["raw_text"].isna()
    | df["normalized_text"].isna()
    | (df["normalized_text"].astype(str).str.strip() == "")
]
if len(empty):
    errors.append(f"{len(empty)} rows have empty text.")

digits = df[df["normalized_text"].astype(str).str.contains(r"\d", regex=True, na=False)]
if len(digits):
    errors.append(f"{len(digits)} rows still contain digits in normalized_text.")

missing_audio = []
for path in df["audio_file_path"].unique():
    if not Path(path).exists():
        missing_audio.append(path)

if missing_audio:
    errors.append(f"{len(missing_audio)} audio files referenced in CSV do not exist.")

if errors:
    print("VALIDATION FAILED")
    print("=" * 80)
    for e in errors:
        print(e)
else:
    print("VALIDATION PASSED")
    print("Every expected non-omitted verse is covered by either a verse row or grouped-verse row.")
    print(f"Total CSV rows: {len(df)}")
    print(df.groupby("language").size())