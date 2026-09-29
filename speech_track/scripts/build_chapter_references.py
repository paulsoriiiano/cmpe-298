"""
Build normalized chapter-level reference transcripts from a master CSV.

The chapter body comes from the already-normalized `normalized_text` column.
Only the added spoken heading is generated here, then normalized using the
shared text_normalization file.
"""

import argparse
from pathlib import Path

import pandas as pd

from book_config import BOOKS
from text_normalization import (
    ENGLISH_HEADER_NUMBERS,
    SPANISH_HEADER_NUMBERS,
    normalize_heading,
)


def chapter_id_from_audio(path: str) -> str:
    return Path(path).stem


def parse_chapter_id(chapter_id: str) -> tuple[str, int]:
    book, chapter = chapter_id.split("_")
    return book, int(chapter)


def make_spoken_heading(language: str, chapter_id: str) -> str:
    book_code, chapter = parse_chapter_id(chapter_id)
    book = BOOKS[book_code]

    if chapter not in ENGLISH_HEADER_NUMBERS or chapter not in SPANISH_HEADER_NUMBERS:
        raise ValueError(f"Missing spoken chapter number mapping for chapter {chapter}")

    if language == "en":
        template = (
            book.english_first_heading
            if chapter == 1
            else book.english_other_heading
        )

        heading = template.format(
            chapter=chapter,
            chapter_en=ENGLISH_HEADER_NUMBERS[chapter],
        )

    elif language == "ilo":
        template = (
            book.ilokano_first_heading
            if chapter == 1
            else book.ilokano_other_heading
        )

        heading = template.format(
            chapter=chapter,
            chapter_es=SPANISH_HEADER_NUMBERS[chapter],
        )

    else:
        raise ValueError(f"Unexpected language: {language}")

    return normalize_heading(heading, language)


def build_references(master_csv: Path, out_root: Path) -> None:
    df = pd.read_csv(master_csv)
    out_root.mkdir(parents=True, exist_ok=True)

    for (language, audio_file_path), group in df.groupby(
        ["language", "audio_file_path"],
        sort=False,
    ):
        chapter_id = chapter_id_from_audio(audio_file_path)
        heading = make_spoken_heading(language, chapter_id)
        body = " ".join(group["normalized_text"].astype(str))

        reference_text = f"{heading} {body}".strip()

        lang_dir = out_root / language
        lang_dir.mkdir(parents=True, exist_ok=True)

        out_path = lang_dir / f"{chapter_id}.txt"
        out_path.write_text(reference_text, encoding="utf-8")

        print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-csv", default="asr_evaluation_master.csv")
    parser.add_argument("--out-root", default="references/chapter_level")
    args = parser.parse_args()

    build_references(Path(args.master_csv), Path(args.out_root))


if __name__ == "__main__":
    main()