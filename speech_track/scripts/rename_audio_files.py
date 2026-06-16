#!/usr/bin/env python3
"""
rename_audio_files.py

Copies English and Ilokano MP3 files into folders:
  audio_dataset/en/
  audio_dataset/ilo/

and renames them consistently as:
  MAT_01.mp3
  MRK_01.mp3
  LUK_01.mp3
  JHN_01.mp3

Run:
  python rename_audio_files.py --src ~/Downloads/ENGESVN1DA --out audio_dataset/en
  python rename_audio_files.py --src ~/Downloads/ILORPVN2DA --out audio_dataset/ilo

Preview without copying:
  python rename_audio_files.py --src ~/Downloads/ILORPVN2DA --out audio_dataset/ilo --dry-run
"""

import argparse
import re
import shutil
from pathlib import Path


BOOK_NUM_TO_CODE = {
    1: "MAT",
    2: "MRK",
    3: "LUK",
}

BOOK_NAME_TO_CODE = {
    "matthew": "MAT",
    "mat": "MAT",
    "mark": "MRK",
    "mrk": "MRK",
    "luke": "LUK",
    "luk": "LUK"
}

BOOK_CHAPTER_COUNTS = {
    "MAT": 28,
    "MRK": 16,
    "LUK": 24,
}


def parse_audio_filename(path: Path):
    """
    Returns (book_code, chapter_number) or None.
    """

    name = path.name

    # English style:
    # ENGESVN1DA_B01_MAT_001.mp3
    match = re.search(
        r"_B(?P<book_num>\d{2})_(?P<book_code>MAT|MRK|LUK|JHN)_(?P<chapter>\d{3})\.mp3$",
        name,
        flags=re.IGNORECASE,
    )
    if match:
        book_code = match.group("book_code").upper()
        chapter = int(match.group("chapter"))
        return book_code, chapter

    # Ilokano style:
    # B03___24_Luke___________ILORPVN2DA.mp3
    # This regex allows any number of underscores/spaces between parts.
    match = re.search(
        r"^B(?P<book_num>\d{2})[_\s]+(?P<chapter>\d{1,3})[_\s]+(?P<book_name>Matthew|Mark|Luke|John)[_\s]+.*\.mp3$",
        name,
        flags=re.IGNORECASE,
    )
    if match:
        book_num = int(match.group("book_num"))
        chapter = int(match.group("chapter"))
        book_name = match.group("book_name").lower()

        # Prefer the written book name because it is clearest.
        book_code = BOOK_NAME_TO_CODE.get(book_name)

        # Fall back to B01/B02/B03/B04 if needed.
        if book_code is None:
            book_code = BOOK_NUM_TO_CODE.get(book_num)

        if book_code:
            return book_code, chapter

    # More flexible fallback for Ilokano if underscores display oddly:
    # Pull book number, chapter, and book name from anywhere in the filename.
    match = re.search(r"B(?P<book_num>\d{2})", name, flags=re.IGNORECASE)
    book_num = int(match.group("book_num")) if match else None

    chapter_match = re.search(r"B\d{2}\D+(?P<chapter>\d{1,3})\D+", name, flags=re.IGNORECASE)
    chapter = int(chapter_match.group("chapter")) if chapter_match else None

    lower_name = name.lower()
    book_code = None
    for book_name, code in BOOK_NAME_TO_CODE.items():
        if book_name in lower_name:
            book_code = code
            break

    if book_code is None and book_num is not None:
        book_code = BOOK_NUM_TO_CODE.get(book_num)

    if book_code is not None and chapter is not None:
        return book_code, chapter

    return None


def copy_rename(src: Path, out: Path, dry_run: bool = False):
    src = src.expanduser().resolve()
    out = out.expanduser().resolve()

    if not src.exists():
        raise FileNotFoundError(f"Source folder does not exist: {src}")

    out.mkdir(parents=True, exist_ok=True)

    copied = []
    skipped = []

    for mp3 in sorted(src.glob("*.mp3")):
        parsed = parse_audio_filename(mp3)

        if parsed is None:
            skipped.append((mp3.name, "could not parse filename"))
            continue

        book_code, chapter = parsed

        if book_code not in BOOK_CHAPTER_COUNTS:
            skipped.append((mp3.name, f"not wanted book: {book_code}"))
            continue

        if chapter < 1 or chapter > BOOK_CHAPTER_COUNTS[book_code]:
            skipped.append((mp3.name, f"chapter out of range for {book_code}: {chapter}"))
            continue

        new_name = f"{book_code}_{chapter:02d}.mp3"
        destination = out / new_name

        if dry_run:
            print(f"WOULD COPY: {mp3.name} -> {new_name}")
        else:
            shutil.copy2(mp3, destination)
            print(f"COPIED: {mp3.name} -> {new_name}")

        copied.append(new_name)

    print()
    print(f"Source: {src}")
    print(f"Output: {out}")
    print(f"Copied: {len(copied)}")
    print(f"Skipped: {len(skipped)}")

    if skipped:
        print()
        print("Skipped files:")
        for filename, reason in skipped:
            print(f"  {filename}  ({reason})")

    print()
    print("Expected for 3 books:")
    print("  Matthew 28 + Mark 16 + Luke 24 = 68 files")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Source folder containing MP3 files")
    parser.add_argument("--out", required=True, help="Output folder for renamed MP3 files")
    parser.add_argument("--dry-run", action="store_true", help="Preview without copying")
    args = parser.parse_args()

    copy_rename(Path(args.src), Path(args.out), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
