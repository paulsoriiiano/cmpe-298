"""
Shared book and fileset configuration for the ASR Bible pipeline.

This is the single source of truth for:
- supported books
- chapter counts
- canonical ordering
- English / Ilokano display names
- spoken chapter-heading templates used when building chapter references
- Bible.is text codes

To add another book, add one BookConfig entry here. The rest of the pipeline
should not need to be edited.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class BookConfig:
    code: str
    order: int
    chapters: int
    english_name: str
    ilokano_name: str
    english_first_heading: str
    english_other_heading: str
    ilokano_first_heading: str
    ilokano_other_heading: str
    aliases: tuple[str, ...]


BIBLEIS_TEXT_CODES: Dict[str, str] = {
    "en": "EN1ESV",
    "ilo": "ILONGN",
}

LANGUAGES: tuple[str, ...] = ("en", "ilo")


BOOKS: Dict[str, BookConfig] = {
    "MAT": BookConfig(
        code="MAT", order=1, chapters=28,
        english_name="Matthew", ilokano_name="San Mateo",
        english_first_heading="Matthew Matthew {chapter_en}",
        english_other_heading="Matthew {chapter_en}",
        ilokano_first_heading="Ti Naimbag a Damag kas Insurat ni San Mateo San Mateo Capitulo {chapter_es}",
        ilokano_other_heading="San Mateo Capitulo {chapter_es}",
        aliases=("matthew", "mat", "mateo", "san mateo"),
    ),
    "MRK": BookConfig(
        code="MRK", order=2, chapters=16,
        english_name="Mark", ilokano_name="San Marcos",
        english_first_heading="Mark Mark {chapter_en}",
        english_other_heading="Mark {chapter_en}",
        ilokano_first_heading="Ti Naimbag a Damag kas Insurat ni San Marcos San Marcos Capitulo {chapter_es}",
        ilokano_other_heading="San Marcos Capitulo {chapter_es}",
        aliases=("mark", "mrk", "marcos", "san marcos"),
    ),
    "LUK": BookConfig(
        code="LUK", order=3, chapters=24,
        english_name="Luke", ilokano_name="San Lucas",
        english_first_heading="Luke Luke {chapter_en}",
        english_other_heading="Luke {chapter_en}",
        ilokano_first_heading="Ti Naimbag a Damag kas Insurat ni San Lucas San Lucas Capitulo {chapter_es}",
        ilokano_other_heading="San Lucas Capitulo {chapter_es}",
        aliases=("luke", "luk", "lucas", "san lucas"),
    ),
    "JHN": BookConfig(
        code="JHN", order=4, chapters=21,
        english_name="John", ilokano_name="San Juan",
        english_first_heading="John John {chapter_en}",
        english_other_heading="John {chapter_en}",
        ilokano_first_heading="Ti Naimbag a Damag kas Insurat ni San Juan San Juan Capitulo {chapter_es}",
        ilokano_other_heading="San Juan Capitulo {chapter_es}",
        aliases=("john", "jhn", "juan", "san juan"),
    ),
    "ACT": BookConfig(
        code="ACT", order=5, chapters=28,
        english_name="Acts", ilokano_name="Dagiti Aramid",
        english_first_heading="Acts Acts {chapter_en}",
        english_other_heading="Acts {chapter_en}",
        ilokano_first_heading="Dagiti Aramid Dagiti Aramid Capitulo {chapter_es}",
        ilokano_other_heading="Dagiti Aramid Capitulo {chapter_es}",
        aliases=("acts", "act", "dagiti aramid", "aramid"),
    ),
    "ROM": BookConfig(
        code="ROM", order=6, chapters=16,
        english_name="Romans", ilokano_name="Taga Roma",
        english_first_heading="Romans Romans {chapter_en}",
        english_other_heading="Romans {chapter_en}",
        ilokano_first_heading="Taga Roma Taga Roma Capitulo {chapter_es}",
        ilokano_other_heading="Taga Roma Capitulo {chapter_es}",
        aliases=("romans", "rom", "roma", "taga roma"),
    ),
}

BOOK_NUMBER_TO_CODE: Dict[int, str] = {
    1: "MAT",
    2: "MRK",
    3: "LUK",
    4: "JHN",
    5: "ACT",
    6: "ROM",
}

_ALIAS_TO_CODE: Dict[str, str] = {}
for _code, _book in BOOKS.items():
    _ALIAS_TO_CODE[_code.lower()] = _code
    for _alias in _book.aliases:
        _ALIAS_TO_CODE[_alias.lower()] = _code


def normalize_book_code(book: str) -> str:
    key = str(book).strip().lower()
    if key in _ALIAS_TO_CODE:
        return _ALIAS_TO_CODE[key]

    upper = str(book).strip().upper()
    if upper in BOOKS:
        return upper

    raise ValueError(f"Unknown book: {book}")


def resolve_books(books: Iterable[str] | None) -> List[str]:
    """Return canonical book codes in configured order."""
    if not books:
        return sorted(BOOKS, key=lambda code: BOOKS[code].order)

    codes = [normalize_book_code(book) for book in books]
    seen = set()
    unique = []

    for code in codes:
        if code not in seen:
            seen.add(code)
            unique.append(code)

    return sorted(unique, key=lambda code: BOOKS[code].order)


def book_order(code: str) -> int:
    return BOOKS[normalize_book_code(code)].order


def chapter_count(code: str) -> int:
    return BOOKS[normalize_book_code(code)].chapters


def valid_book_codes() -> List[str]:
    return sorted(BOOKS, key=lambda code: BOOKS[code].order)