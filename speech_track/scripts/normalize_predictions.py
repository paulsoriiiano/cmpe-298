"""
Normalize Whisper Large-v3 predictions.

Input:
    transcriptions/whisper_large_v3/<language>/<chapter>.txt

Output:
    transcriptions/whisper_large_v3_normalized/<language>/<chapter>.txt
"""

import re
import string
from pathlib import Path


IN_ROOT = Path("transcriptions/whisper_large_v3")
OUT_ROOT = Path("transcriptions/whisper_large_v3_normalized")


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

    return " ".join(ILOKANO_NUMBERS[int(d)] for d in str(n))


def replace_digits(text, language):
    def repl(match):
        n = int(match.group(0))
        if language == "en":
            return english_number_to_words(n)
        return ilokano_number_to_words(n)

    return re.sub(r"\d+", repl, text)


def normalize_text(text, language):
    text = str(text).lower()

    text = text.replace("※", " ")
    text = text.replace("†", " ")
    text = text.replace("*", " ")

    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = text.replace("‘", "'")
    text = text.replace("’", "'")
    text = text.replace("—", " ")
    text = text.replace("–", " ")
    text = text.replace("…", " ")

    text = replace_digits(text, language)

    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def main():
    for language in ["en", "ilo"]:
        in_dir = IN_ROOT / language
        out_dir = OUT_ROOT / language
        out_dir.mkdir(parents=True, exist_ok=True)

        if not in_dir.exists():
            raise FileNotFoundError(f"Missing input directory: {in_dir}")

        for path in sorted(in_dir.glob("*.txt")):
            raw_prediction = path.read_text(encoding="utf-8")
            normalized_prediction = normalize_text(raw_prediction, language)

            out_path = out_dir / path.name
            out_path.write_text(normalized_prediction, encoding="utf-8")

            print(f"Wrote {out_path}")

    print("Done normalizing Whisper predictions.")


if __name__ == "__main__":
    main()