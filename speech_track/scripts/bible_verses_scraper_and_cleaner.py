"""
Scrape Bible.is verse/grouped-verse text and build an ASR master CSV.

This script is book-agnostic. Choose books with --books, e.g.:
  python bible_verses_scraper_and_cleaner.py --books MAT MRK LUK --delay 5
  python bible_verses_scraper_and_cleaner.py --books JHN ACT ROM --out fine_tune_master.csv

Output columns are preserved:
  verse_id, language, audio_file_path, raw_text, normalized_text
"""

import argparse
import re
import time
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

from book_config import BIBLEIS_TEXT_CODES, LANGUAGES, BOOKS, resolve_books, book_order
from text_normalization import clean_raw_text, normalize_text


def bibleis_id_to_verse_id(data_id: str) -> str:
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


def chapter_from_verse_id(verse_id: str) -> tuple[str, int]:
    parts = verse_id.split("_")
    return parts[0], int(parts[1])


def extract_rows_from_page(
    html: str,
    language: str,
    book: str,
    chapter: int,
    audio_root: str,
) -> list[dict]:
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

        if parsed_book != book or parsed_chapter != chapter:
            continue

        raw_text = clean_raw_text(span.get_text(" ", strip=True))

        if not raw_text:
            continue

        rows.append({
            "verse_id": verse_id,
            "language": language,
            "audio_file_path": f"{audio_root.rstrip('/')}/{language}/{parsed_book}_{parsed_chapter:02d}.mp3",
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


def scrape_chapter(
    page,
    language: str,
    book: str,
    chapter: int,
    audio_root: str,
) -> list[dict]:
    bibleis_code = BIBLEIS_TEXT_CODES[language]
    url = f"https://live.bible.is/bible/{bibleis_code}/{book}/{chapter}"

    print(f"Scraping {language}: {url}")

    page.goto(url, wait_until="domcontentloaded", timeout=60000)

    try:
        page.wait_for_selector("span.v[data-id]", timeout=15000)
    except Exception:
        pass

    page.wait_for_timeout(4000)

    html = page.content()
    rows = extract_rows_from_page(html, language, book, chapter, audio_root)

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
        print(f"WARNING: no rows for {language} {book}_{chapter:02d}; saved {debug_path}")
        return []

    print(f"  kept {len(rows)} verse/grouped rows")
    return rows


def build_master(
    audio_root: str,
    books: list[str],
    languages: list[str],
    delay: float,
    show_browser: bool,
) -> pd.DataFrame:
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

        for language in languages:
            if language not in BIBLEIS_TEXT_CODES:
                raise ValueError(f"Unsupported language: {language}")

            for book in books:
                for chapter in range(1, BOOKS[book].chapters + 1):
                    all_rows.extend(scrape_chapter(page, language, book, chapter, audio_root))
                    time.sleep(delay)

        browser.close()

    df = pd.DataFrame(all_rows)

    if df.empty:
        raise RuntimeError("No rows scraped.")

    sort_parts = df["verse_id"].str.extract(
        r"(?P<book>[A-Z]+)_(?P<chapter>\d+)_(?P<start>\d+)(?:_(?P<end>\d+))?"
    )

    df["_book_order"] = sort_parts["book"].map(book_order)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--books", nargs="+", default=["MAT", "MRK", "LUK"], help="Book codes or aliases")
    parser.add_argument("--languages", nargs="+", default=list(LANGUAGES), help="Languages to scrape")
    parser.add_argument("--audio-root", default="audio_dataset")
    parser.add_argument("--out", default="asr_evaluation_master.csv")
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--show-browser", action="store_true")
    args = parser.parse_args()

    books = resolve_books(args.books)

    master = build_master(
        audio_root=args.audio_root,
        books=books,
        languages=args.languages,
        delay=args.delay,
        show_browser=args.show_browser,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(out, index=False, encoding="utf-8")

    print(f"Wrote {out} with {len(master)} rows")
    print(f"Books: {', '.join(books)}")


if __name__ == "__main__":
    main()