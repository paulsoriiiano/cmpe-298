"""Validate an ASR master CSV using shared book configuration."""

import argparse
import re
from pathlib import Path

import pandas as pd

from book_config import BOOKS, resolve_books


OMITTED = {
    "en": {
        "MAT_12_47", "MAT_17_21", "MAT_18_11", "MAT_23_14",
        "MRK_07_16", "MRK_09_44", "MRK_09_46", "MRK_11_26", "MRK_15_28",
        "LUK_17_36", "LUK_23_17",

        "JHN_05_04",
        "ACT_08_37", "ACT_15_34", "ACT_24_07", "ACT_28_29",
        "ROM_16_24",
    },
    "ilo": {
        "MAT_12_47", "MAT_17_21", "MAT_18_11", "MAT_23_14",
        "MRK_07_16", "MRK_09_44", "MRK_09_46", "MRK_11_26", "MRK_15_28",
        "LUK_17_36", "LUK_23_17",
        "LUK_22_43", "LUK_22_44", "LUK_24_12", "LUK_24_40",

        "JHN_07_53",
        "ACT_08_37", "ACT_15_34", "ACT_24_07", "ACT_28_29",
        "ROM_16_24",
    },
}


# Last verse counts are needed for strict verse-level completeness validation
EXPECTED_LAST_VERSE = {
    "MAT": [
        25, 23, 17, 25, 48, 34, 29, 34, 38, 42, 30, 50, 58, 36,
        39, 28, 27, 35, 30, 34, 46, 46, 39, 51, 46, 75, 66, 20,
    ],
    "MRK": [
        45, 28, 35, 41, 43, 56, 37, 38, 50, 52, 33, 44, 37, 72, 47, 20,
    ],
    "LUK": [
        80, 52, 38, 44, 39, 49, 50, 56, 62, 42, 54, 59, 35, 35,
        32, 31, 37, 43, 48, 47, 38, 71, 56, 53,
    ],
    "JHN": [
        51, 25, 36, 54, 47, 71, 53, 59, 41, 42, 57, 50, 38, 31,
        27, 33, 26, 40, 42, 31, 25,
    ],
    "ACT": [
        26, 47, 26, 37, 42, 15, 60, 40, 43, 48, 30, 25, 52, 28,
        41, 40, 34, 28, 41, 38, 40, 30, 35, 27, 27, 32, 44, 31,
    ],
    "ROM": [
        32, 29, 31, 25, 21, 23, 25, 39, 33, 21, 36, 21, 14, 23, 33, 27,
    ],
}


def parse_verse_id(vid: str):
    match = re.match(r"^([A-Z]{3})_(\d{2})_(\d{2})(?:_(\d{2}))?$", str(vid))

    if not match:
        return None

    book = match.group(1)
    chapter = int(match.group(2))
    start = int(match.group(3))
    end = int(match.group(4)) if match.group(4) else start

    return book, chapter, start, end


def omitted_for_chapter(lang: str, book: str, chapter: int) -> set[int]:
    omitted = set()

    for vid in OMITTED.get(lang, set()):
        parsed = parse_verse_id(vid)

        if parsed is None:
            continue

        b, ch, start, end = parsed

        if b == book and ch == chapter:
            omitted.update(range(start, end + 1))

    return omitted


def validate(
    master_csv: Path,
    books: list[str],
    strict_verse_coverage: bool = True,
) -> list[str]:
    df = pd.read_csv(master_csv)
    errors = []

    required_cols = {
        "verse_id",
        "language",
        "audio_file_path",
        "raw_text",
        "normalized_text",
    }

    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        return [f"Missing required columns: {sorted(missing_cols)}"]

    for vid in df["verse_id"]:
        if parse_verse_id(vid) is None:
            errors.append(f"Bad verse_id format: {vid}")

    for lang in sorted(df["language"].dropna().unique()):
        lang_df = df[df["language"] == lang]

        for book in books:
            for chapter in range(1, BOOKS[book].chapters + 1):
                prefix = f"{book}_{chapter:02d}_"

                rows = lang_df[
                    lang_df["verse_id"].str.startswith(prefix, na=False)
                ]

                if rows.empty:
                    errors.append(f"{lang} {book}_{chapter:02d} has no text rows")
                    continue

                if strict_verse_coverage and book in EXPECTED_LAST_VERSE:
                    expected_last = EXPECTED_LAST_VERSE[book][chapter - 1]
                    covered = set()

                    for vid in rows["verse_id"]:
                        parsed = parse_verse_id(vid)

                        if parsed:
                            _, _, start, end = parsed
                            covered.update(range(start, end + 1))

                    expected = set(range(1, expected_last + 1))
                    expected -= omitted_for_chapter(lang, book, chapter)

                    missing = sorted(expected - covered)
                    extra = sorted(covered - set(range(1, expected_last + 1)))

                    if missing:
                        errors.append(f"{lang} {book}_{chapter:02d} missing verses: {missing}")

                    if extra:
                        errors.append(f"{lang} {book}_{chapter:02d} has impossible verses: {extra}")

    bad_markers = df[
        df["raw_text"].astype(str).str.contains(r"※|†|\*", regex=True, na=False)
    ]

    if len(bad_markers):
        errors.append(f"{len(bad_markers)} rows still contain footnote markers")

    empty = df[
        df["raw_text"].isna()
        | df["normalized_text"].isna()
        | (df["normalized_text"].astype(str).str.strip() == "")
    ]

    if len(empty):
        errors.append(f"{len(empty)} rows have empty text")

    digits = df[
        df["normalized_text"].astype(str).str.contains(r"\d", regex=True, na=False)
    ]

    if len(digits):
        errors.append(f"{len(digits)} rows still contain digits in normalized_text")

    missing_audio = [
        path for path in df["audio_file_path"].unique()
        if not Path(path).exists()
    ]

    if missing_audio:
        errors.append(f"{len(missing_audio)} audio files referenced in CSV do not exist")

    dupes = df[
        df.duplicated(subset=["verse_id", "language"], keep=False)
    ]

    if len(dupes):
        errors.append(f"{len(dupes)} rows have duplicate verse_id/language pairs")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-csv", default="asr_evaluation_master.csv")
    parser.add_argument("--books", nargs="+", default=["MAT", "MRK", "LUK"])
    parser.add_argument("--no-strict-verse-coverage", action="store_true")
    args = parser.parse_args()

    books = resolve_books(args.books)

    errors = validate(
        Path(args.master_csv),
        books,
        strict_verse_coverage=not args.no_strict_verse_coverage,
    )

    if errors:
        print("VALIDATION FAILED")
        print("=" * 80)

        for error in errors:
            print(error)

        raise SystemExit(1)

    df = pd.read_csv(args.master_csv)

    print("VALIDATION PASSED")
    print(f"Books: {', '.join(books)}")
    print(f"Total CSV rows: {len(df)}")
    print(df.groupby("language").size())


if __name__ == "__main__":
    main()