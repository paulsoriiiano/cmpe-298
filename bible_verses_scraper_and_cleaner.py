#!/usr/bin/env python3
"""
Bible.is scraper + cleaner for ASR evaluation.

Output:
    asr_evaluation_master.csv

Columns:
    verse_id
    language
    audio_file_path
    raw_text
    normalized_text

Run:
python bible_verses_scraper_and_cleaner.py --delay 5
python validate_asr_master.py

"""

import argparse
import re
import string
import time
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright


BOOKS = {
    "MAT": 28,
    "MRK": 16,
    "LUK": 24,
}

BIBLEIS_CODES = {
    "en": "EN1ESV",
    "ilo": "ILONGN",
}

ENGLISH_NUMBERS = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
    5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
    10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen",
    14: "fourteen", 15: "fifteen", 16: "sixteen",
    17: "seventeen", 18: "eighteen", 19: "nineteen",
    20: "twenty", 30: "thirty", 40: "forty", 50: "fifty",
    60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety",
}

ILOKANO_NUMBERS = {
    0: "awan", 1: "maysa", 2: "dua", 3: "tallo", 4: "uppat",
    5: "lima", 6: "innem", 7: "pito", 8: "walo", 9: "siam",
    10: "sangapulo", 11: "sangapulo ket maysa",
    12: "sangapulo ket dua", 13: "sangapulo ket tallo",
    14: "sangapulo ket uppat", 15: "sangapulo ket lima",
    16: "sangapulo ket innem", 17: "sangapulo ket pito",
    18: "sangapulo ket walo", 19: "sangapulo ket siam",
    20: "duapulo",
}


def bibleis_id_to_verse_id(data_id):
    """
    Convert Bible.is data-id values into verse_id format.

    Examples:
        MAT1_1      -> MAT_01_01
        MAT1_2-6a   -> MAT_01_02_06
        MAT1_6b-11  -> MAT_01_06_11
    """
    match = re.match(r"^([A-Z]{3})(\d+)_(.+)$", data_id)

    if not match:
        raise ValueError(f"Unexpected data-id format: {data_id}")

    book = match.group(1)
    chapter = int(match.group(2))
    verse_part = match.group(3)

    nums = re.findall(r"\d+", verse_part)
    if not nums:
        raise ValueError(f"No verse number found in data-id: {data_id}")

    start_verse = int(nums[0])
    end_verse = int(nums[-1])

    if start_verse == end_verse:
        return f"{book}_{chapter:02d}_{start_verse:02d}"

    return f"{book}_{chapter:02d}_{start_verse:02d}_{end_verse:02d}"


def chapter_from_verse_id(verse_id):
    parts = verse_id.split("_")
    return parts[0], parts[1]


def clean_raw_text(text):
    """
    Preserve original readable verse text, but remove HTML artifacts and footnote markers.
    """
    text = BeautifulSoup(str(text), "html.parser").get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()

    # Remove Bible.is footnote/reference markers
    text = text.replace("※", " ")
    text = text.replace("†", " ")
    text = text.replace("*", " ")
    text = re.sub(r"[※†*]+", " ", text)

    # Clean empty punctuation leftovers
    text = re.sub(r"\[\s*\]", " ", text)
    text = re.sub(r"\(\s*\)", " ", text)

    # Fix spacing around punctuation
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def english_number_to_words(n):
    if n in ENGLISH_NUMBERS:
        return ENGLISH_NUMBERS[n]

    if n < 100:
        tens = (n // 10) * 10
        ones = n % 10
        return ENGLISH_NUMBERS[tens] + (" " + ENGLISH_NUMBERS[ones] if ones else "")

    if n < 1000:
        words = ENGLISH_NUMBERS[n // 100] + " hundred"
        if n % 100:
            words += " " + english_number_to_words(n % 100)
        return words

    if n < 10000:
        words = english_number_to_words(n // 1000) + " thousand"
        if n % 1000:
            words += " " + english_number_to_words(n % 1000)
        return words

    return " ".join(ENGLISH_NUMBERS[int(d)] for d in str(n))


def ilokano_number_to_words(n):
    if n in ILOKANO_NUMBERS:
        return ILOKANO_NUMBERS[n]

    # For larger Ilokano numbers, use digit-by-digit spelling to avoid bad morphology
    return " ".join(ILOKANO_NUMBERS[int(d)] for d in str(n))


def replace_digits(text, language):
    def repl(match):
        n = int(match.group(0))
        if language == "en":
            return english_number_to_words(n)
        return ilokano_number_to_words(n)

    return re.sub(r"\d+", repl, text)


def normalize_text(text, language):
    """
    ASR / WER normalization.

    Steps:
        1. lowercase
        2. remove Bible.is footnote markers
        3. convert digits to words
        4. remove punctuation
        5. collapse whitespace
    """

    text = str(text).lower()

    # Remove Bible.is footnote/reference markers
    text = text.replace("※", " ")
    text = text.replace("†", " ")
    text = text.replace("*", " ")

    # Normalize Unicode punctuation
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = text.replace("‘", "'")
    text = text.replace("’", "'")
    text = text.replace("—", " ")
    text = text.replace("–", " ")
    text = text.replace("…", " ")

    # Convert digits to words
    text = replace_digits(text, language)

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Remove any remaining non-word symbols
    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
        flags=re.UNICODE,
    )

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_rows_from_page(html, language, book, chapter, audio_root):
    soup = BeautifulSoup(html, "html.parser")

    rows = []

    for span in soup.select("span.v[data-id]"):
        data_id = span.get("data-id", "").strip()

        if not data_id:
            continue

        try:
            verse_id = bibleis_id_to_verse_id(data_id)
        except ValueError:
            continue

        parsed_book, parsed_chapter = chapter_from_verse_id(verse_id)

        if parsed_book != book or int(parsed_chapter) != chapter:
            continue

        raw_text = clean_raw_text(span.get_text(" ", strip=True))

        if not raw_text:
            continue

        rows.append({
            "verse_id": verse_id,
            "language": language,
            "audio_file_path": f"{audio_root}/{language}/{parsed_book}_{parsed_chapter}.mp3",
            "raw_text": raw_text,
            "normalized_text": normalize_text(raw_text, language),
        })

    seen = set()
    final_rows = []

    for row in rows:
        key = (row["verse_id"], row["language"], row["raw_text"])
        if key in seen:
            continue
        seen.add(key)
        final_rows.append(row)

    return final_rows


def scrape_chapter(page, language, book, chapter, audio_root):
    bibleis_code = BIBLEIS_CODES[language]
    url = f"https://live.bible.is/bible/{bibleis_code}/{book}/{chapter}"

    print(f"Scraping {language}: {url}")

    page.goto(url, wait_until="domcontentloaded", timeout=60000)

    # Wait specifically for Bible.is verse spans
    try:
        page.wait_for_selector("span.v[data-id]", timeout=15000)
    except Exception:
        pass

    # Extra wait for chapters that render slowly
    page.wait_for_timeout(4000)

    html = page.content()
    rows = extract_rows_from_page(html, language, book, chapter, audio_root)

    # Retry once if page was not fully rendered
    if not rows:
        page.reload(wait_until="domcontentloaded", timeout=60000)
        try:
            page.wait_for_selector("span.v[data-id]", timeout=15000)
        except Exception:
            pass
        page.wait_for_timeout(5000)

        html = page.content()
        rows = extract_rows_from_page(html, language, book, chapter, audio_root)

    if not rows:
        debug_path = Path(f"debug_{language}_{book}_{chapter:02d}.html")
        debug_path.write_text(html, encoding="utf-8")
        print(
            f"WARNING: No verse/grouped-verse rows found for "
            f"{language} {book}_{chapter:02d}. Saved {debug_path}"
        )
        return []

    print(f"  kept {len(rows)} verse/grouped rows")
    return rows


def build_master(audio_root, delay, show_browser):
    all_rows = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not show_browser)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 Chrome/120 Safari/537.36"
            )
        )
        page = context.new_page()

        for language in ["en", "ilo"]:
            for book, chapter_count in BOOKS.items():
                for chapter in range(1, chapter_count + 1):
                    rows = scrape_chapter(page, language, book, chapter, audio_root)
                    all_rows.extend(rows)
                    time.sleep(delay)

        browser.close()

    df = pd.DataFrame(all_rows)

    df = df[[
        "verse_id",
        "language",
        "audio_file_path",
        "raw_text",
        "normalized_text",
    ]]

    sort_parts = df["verse_id"].str.extract(
        r"(?P<book>[A-Z]+)_(?P<chapter>\d+)_(?P<start>\d+)(?:_(?P<end>\d+))?"
    )

    df["_book_order"] = sort_parts["book"].map({"MAT": 1, "MRK": 2, "LUK": 3})
    df["_chapter"] = sort_parts["chapter"].astype(int)
    df["_start"] = sort_parts["start"].astype(int)

    df = df.sort_values(["language", "_book_order", "_chapter", "_start"])

    return df[[
        "verse_id",
        "language",
        "audio_file_path",
        "raw_text",
        "normalized_text",
    ]]


def validate_chapter_coverage(df):
    """
    Ensure every requested language/book/chapter has at least one verse/grouped-verse row.
    """
    missing = []

    for language in ["en", "ilo"]:
        for book, chapter_count in BOOKS.items():
            for chapter in range(1, chapter_count + 1):
                prefix = f"{book}_{chapter:02d}_"
                exists = (
                    (df["language"] == language)
                    & df["verse_id"].str.startswith(prefix)
                ).any()

                if not exists:
                    missing.append(f"{language} {book}_{chapter:02d}")

    if missing:
        raise RuntimeError("Missing chapter coverage:\n" + "\n".join(missing))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-root", default="audio_dataset")
    parser.add_argument("--out", default="asr_evaluation_master.csv")
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--show-browser", action="store_true")
    args = parser.parse_args()

    master = build_master(
        audio_root=args.audio_root.rstrip("/"),
        delay=args.delay,
        show_browser=args.show_browser,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(out, index=False, encoding="utf-8")

    print(f"Wrote {out} with {len(master)} rows")
    print("Done: verse-level/grouped-verse rows extracted from Bible.is span.v[data-id].")


if __name__ == "__main__":
    main()