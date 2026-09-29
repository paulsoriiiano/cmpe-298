"""
Shared normalization utilities.

Both reference construction and prediction normalization import from this file.
"""

import re
import string
from bs4 import BeautifulSoup


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

ENGLISH_HEADER_NUMBERS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
    11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
    15: "fifteen", 16: "sixteen", 17: "seventeen",
    18: "eighteen", 19: "nineteen", 20: "twenty",
    21: "twenty one", 22: "twenty two", 23: "twenty three",
    24: "twenty four", 25: "twenty five", 26: "twenty six",
    27: "twenty seven", 28: "twenty eight",
}

SPANISH_HEADER_NUMBERS = {
    1: "uno", 2: "dos", 3: "tres", 4: "cuatro", 5: "cinco",
    6: "seis", 7: "siete", 8: "ocho", 9: "nueve", 10: "diez",
    11: "once", 12: "doce", 13: "trece", 14: "catorce",
    15: "quince", 16: "dieciseis", 17: "diecisiete",
    18: "dieciocho", 19: "diecinueve", 20: "veinte",
    21: "veintiuno", 22: "veintidos", 23: "veintitres",
    24: "veinticuatro", 25: "veinticinco", 26: "veintiseis",
    27: "veintisiete", 28: "veintiocho",
}


def english_number_to_words(n: int) -> str:
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


def ilokano_number_to_words(n: int) -> str:
    if n in ILOKANO_NUMBERS:
        return ILOKANO_NUMBERS[n]

    return " ".join(ILOKANO_NUMBERS[int(d)] for d in str(n))


def replace_digits(text: str, language: str) -> str:
    def repl(match: re.Match) -> str:
        n = int(match.group(0))

        if language == "en":
            return english_number_to_words(n)

        if language == "ilo":
            return ilokano_number_to_words(n)

        raise ValueError(f"Unsupported language for digit replacement: {language}")

    return re.sub(r"\d+", repl, text)


def clean_raw_text(text: str) -> str:
    """Clean scraped Bible.is verse text while preserving readable raw text."""
    text = BeautifulSoup(str(text), "html.parser").get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()

    text = re.sub(r"[※†*]+", " ", text)
    text = re.sub(r"\[\s*\]", " ", text)
    text = re.sub(r"\(\s*\)", " ", text)

    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_text(text: str, language: str, convert_digits: bool = True) -> str:
    """ASR/WER normalization used for both references and hypotheses."""
    text = str(text).lower()

    text = re.sub(r"[※†*]+", " ", text)

    text = (
        text.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
        .replace("—", " ")
        .replace("–", " ")
        .replace("…", " ")
    )

    if convert_digits:
        text = replace_digits(text, language)

    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_heading(text: str, language: str) -> str:
    """Normalize generated spoken headings with the same WER rules."""
    return normalize_text(text, language, convert_digits=True)