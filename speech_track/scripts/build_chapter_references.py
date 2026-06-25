"""
Build chapter-level reference transcripts from asr_evaluation_master.csv.

Input:
    asr_evaluation_master.csv

Chapter text comes from the earlier Bible.is scraping pipeline.
This script uses the `normalized_text` column.

The scraping step omitted headings because they are not part of the verse text,
but the headings are spoken in the audio. Therefore this script prepends the
spoken heading to each chapter-level reference.

Heading convention:
    English:
        Chapter 1: "Luke Luke 1"
        Other chapters: "Luke 2"

    Ilokano:
        Chapter 1:
            "Ti Naimbag a Damag kas Insurat ni San Lucas San Lucas Capitulo Uno"
        Other chapters:
            "San Lucas Capitulo Dos"

Output:
    references/chapter_level/<language>/<chapter>.txt
"""

from pathlib import Path
import re
import string

import pandas as pd


MASTER_CSV = "asr_evaluation_master.csv"
OUT_DIR = Path("references/chapter_level")


ENGLISH_BOOK_NAMES = {
    "MAT": "Matthew",
    "MRK": "Mark",
    "LUK": "Luke",
}

ILOKANO_BOOK_NAMES = {
    "MAT": "San Mateo",
    "MRK": "San Marcos",
    "LUK": "San Lucas",
}

ILOKANO_FIRST_CHAPTER_TITLES = {
    "MAT": "Ti Naimbag a Damag kas Insurat ni San Mateo",
    "MRK": "Ti Naimbag a Damag kas Insurat ni San Marcos",
    "LUK": "Ti Naimbag a Damag kas Insurat ni San Lucas",
}

# English chapter headings are normalized to English number words
# Only 1-28 are needed because Matthew has 28 chapters
ENGLISH_HEADER_NUMBERS = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
    20: "twenty",
    21: "twenty one",
    22: "twenty two",
    23: "twenty three",
    24: "twenty four",
    25: "twenty five",
    26: "twenty six",
    27: "twenty seven",
    28: "twenty eight",
}

# Ilokano audio uses Spanish-style chapter numbers in the heading
# This applies only to the spoken chapter heading, not to scripture body text,
# which is assumed to use native Ilokano numbering.
SPANISH_HEADER_NUMBERS = {
    1: "uno",
    2: "dos",
    3: "tres",
    4: "cuatro",
    5: "cinco",
    6: "seis",
    7: "siete",
    8: "ocho",
    9: "nueve",
    10: "diez",
    11: "once",
    12: "doce",
    13: "trece",
    14: "catorce",
    15: "quince",
    16: "dieciseis",
    17: "diecisiete",
    18: "dieciocho",
    19: "diecinueve",
    20: "veinte",
    21: "veintiuno",
    22: "veintidos",
    23: "veintitres",
    24: "veinticuatro",
    25: "veinticinco",
    26: "veintiseis",
    27: "veintisiete",
    28: "veintiocho",
}


def chapter_id_from_audio(path):
    # audio_dataset/en/MAT_01.mp3 -> MAT_01
    return Path(path).stem


def parse_chapter_id(chapter_id):
    # MAT_01 -> ("MAT", 1)
    book, chapter = chapter_id.split("_")
    return book, int(chapter)


def normalize_heading(text):
    """
    Normalize only the added heading.

    The body text is already normalized in asr_evaluation_master.csv.
    This function makes the heading match that style:
        - lowercase
        - remove punctuation
        - collapse whitespace
    """
    text = str(text).lower()

    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = text.replace("‘", "'")
    text = text.replace("’", "'")
    text = text.replace("—", " ")
    text = text.replace("–", " ")
    text = text.replace("…", " ")

    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def make_spoken_heading(language, chapter_id):
    book, chapter = parse_chapter_id(chapter_id)

    if language == "en":
        book_name = ENGLISH_BOOK_NAMES[book]
        chapter_word = ENGLISH_HEADER_NUMBERS[chapter]

        if chapter == 1:
            heading = f"{book_name} {book_name} {chapter_word}"
        else:
            heading = f"{book_name} {chapter_word}"

    elif language == "ilo":
        book_name = ILOKANO_BOOK_NAMES[book]
        chapter_word = SPANISH_HEADER_NUMBERS[chapter]

        if chapter == 1:
            title = ILOKANO_FIRST_CHAPTER_TITLES[book]
            heading = f"{title} {book_name} Capitulo {chapter_word}"
        else:
            heading = f"{book_name} Capitulo {chapter_word}"

    else:
        raise ValueError(f"Unexpected language: {language}")

    return normalize_heading(heading)


def main():
    df = pd.read_csv(MASTER_CSV)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for (language, audio_file_path), group in df.groupby(["language", "audio_file_path"]):
        chapter_id = chapter_id_from_audio(audio_file_path)

        heading = make_spoken_heading(language, chapter_id)

        body = " ".join(group["normalized_text"].astype(str))

        reference_text = f"{heading} {body}".strip()

        lang_dir = OUT_DIR / language
        lang_dir.mkdir(parents=True, exist_ok=True)

        out_path = lang_dir / f"{chapter_id}.txt"
        out_path.write_text(reference_text, encoding="utf-8")

        print(f"Wrote {out_path}")

    print("Done building normalized chapter-level references with spoken headings.")


if __name__ == "__main__":
    main()