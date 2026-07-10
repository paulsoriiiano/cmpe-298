"""
Copy and rename Bible.is MP3 files into a stable chapter-level layout.

Examples:
  python rename_audio_files.py --src ~/Downloads/ENGESVN2DA --out audio_dataset/en --books JHN ACT ROM
  python rename_audio_files.py --src ~/Downloads/ILORPVN2DA --out audio_dataset/ilo --books JHN ACT ROM

Output names:
  JHN_01.mp3, ACT_01.mp3, ROM_01.mp3, etc.
"""

import argparse
import re
import shutil
from pathlib import Path

from book_config import BOOKS, BOOK_NUMBER_TO_CODE, resolve_books, normalize_book_code


def parse_audio_filename(path: Path) -> tuple[str, int] | None:
    name = path.name

    # Common English Bible.is style, e.g. ENGESVN2DA_B04_JHN_001.mp3
    match = re.search(
        r"_B(?P<book_num>\d{2})_(?P<book_code>[A-Z0-9]{3})_(?P<chapter>\d{3})\.mp3$",
        name,
        flags=re.IGNORECASE,
    )

    if match:
        raw_code = match.group("book_code").upper()
        chapter = int(match.group("chapter"))

        try:
            return normalize_book_code(raw_code), chapter
        except ValueError:
            book_num = int(match.group("book_num"))
            if book_num in BOOK_NUMBER_TO_CODE:
                return BOOK_NUMBER_TO_CODE[book_num], chapter

    # Common Ilokano style, e.g. B04___01_John___________ILORPVN2DA.mp3
    match = re.search(
        r"^B(?P<book_num>\d{2})[_\s]+(?P<chapter>\d{1,3})[_\s]+(?P<book_name>[A-Za-z ]+?)[_\s]+.*\.mp3$",
        name,
        flags=re.IGNORECASE,
    )

    if match:
        book_num = int(match.group("book_num"))
        chapter = int(match.group("chapter"))
        book_name = match.group("book_name").strip().lower()

        try:
            return normalize_book_code(book_name), chapter
        except ValueError:
            if book_num in BOOK_NUMBER_TO_CODE:
                return BOOK_NUMBER_TO_CODE[book_num], chapter

    lower_name = name.lower()

    for code, book in BOOKS.items():
        candidates = (
            code.lower(),
            book.english_name.lower(),
            book.ilokano_name.lower(),
            *book.aliases,
        )

        if any(alias and alias in lower_name for alias in candidates):
            chapter_match = re.search(
                r"B\d{2}\D+(?P<chapter>\d{1,3})\D+",
                name,
                flags=re.IGNORECASE,
            )

            if chapter_match:
                return code, int(chapter_match.group("chapter"))

    return None


def copy_rename(src: Path, out: Path, books: list[str], dry_run: bool = False) -> None:
    src = src.expanduser().resolve()
    out = out.expanduser().resolve()
    wanted = set(resolve_books(books))

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

        if book_code not in wanted:
            skipped.append((mp3.name, f"not requested book: {book_code}"))
            continue

        if chapter < 1 or chapter > BOOKS[book_code].chapters:
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

    expected = sum(BOOKS[code].chapters for code in wanted)

    print(f"\nSource: {src}")
    print(f"Output: {out}")
    print(f"Requested books: {', '.join(sorted(wanted, key=lambda c: BOOKS[c].order))}")
    print(f"Copied: {len(copied)} / expected {expected}")
    print(f"Skipped: {len(skipped)}")

    if skipped:
        print("\nSkipped files:")
        for filename, reason in skipped:
            print(f"  {filename} ({reason})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--books", nargs="+", default=["MAT", "MRK", "LUK"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    copy_rename(Path(args.src), Path(args.out), args.books, dry_run=args.dry_run)


if __name__ == "__main__":
    main()