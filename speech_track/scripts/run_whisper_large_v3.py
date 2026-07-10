"""
Two-pass Whisper Large-v3 transcription for chapter-level audio.

Pass 1:
    Create a temporary clip from the beginning of the chapter audio and let
    Whisper detect the language.

Pass 2:
    Transcribe the full chapter audio from 0:00 using the detected language.

Important:
    The master CSV is treated as the text/reference dataset.
    The actual audio files are chosen with --audio-root.

Example:
    python run_whisper_large_v3.py \
        --master-csv asr_evaluation_master.csv \
        --audio-root audio_dataset_matched \
        --out-root transcriptions/whisper_large_v3_matched
"""

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

import pandas as pd


def make_clip(audio_path: Path, clip_path: Path, seconds: int) -> None:
    """
    Use ffmpeg to create a temporary clip for language detection.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(audio_path),
        "-t",
        str(seconds),
        str(clip_path),
    ]

    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def detect_language(audio_path: Path, model_name: str, seconds: int) -> str:
    """
    Run Whisper on the first N seconds and parse the detected language.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        clip_path = tmpdir / f"first_{seconds}_seconds.wav"
        detect_out = tmpdir / "detect_out"
        detect_out.mkdir(parents=True, exist_ok=True)

        print(f"  [Pass 1] Creating {seconds}-second clip...", flush=True)
        make_clip(audio_path, clip_path, seconds)

        print("  [Pass 1] Running Whisper language detection...", flush=True)

        cmd = [
            "whisper",
            str(clip_path),
            "--model",
            model_name,
            "--task",
            "transcribe",
            "--temperature",
            "0",
            "--output_format",
            "txt",
            "--output_dir",
            str(detect_out),
            "--fp16",
            "False",
        ]

        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )

        output = result.stdout + "\n" + result.stderr

    match = re.search(r"Detected language:\s*([A-Za-z]+)", output)

    if not match:
        raise RuntimeError(
            f"Could not parse detected language for {audio_path}.\n\n"
            f"Whisper output was:\n{output}"
        )

    detected_language = match.group(1)
    print(f"  [Pass 1] Detected language: {detected_language}", flush=True)

    return detected_language


def transcribe_full_audio(
    audio_path: Path,
    detected_language: str,
    out_dir: Path,
    model_name: str,
) -> None:
    """
    Transcribe the full original chapter audio using the detected language.
    """
    print("  [Pass 2] Transcribing full audio from 0:00...", flush=True)

    cmd = [
        "whisper",
        str(audio_path),
        "--model",
        model_name,
        "--task",
        "transcribe",
        "--language",
        detected_language,
        "--temperature",
        "0",
        "--output_format",
        "all",
        "--output_dir",
        str(out_dir),
        "--fp16",
        "False",
    ]

    subprocess.run(cmd, check=True)

    print("  [Pass 2] Finished full transcription.", flush=True)


def transcribe_from_master(
    master_csv: Path,
    audio_root: Path,
    out_root: Path,
    model_name: str,
    detect_seconds: int,
) -> None:
    """
    Transcribe each unique chapter listed in the master CSV.

    The CSV is used to determine:
        - language
        - chapter_id

    The actual audio path is constructed as:
        <audio_root>/<language>/<chapter_id>.mp3

    This allows the same text CSV to be reused with different audio conditions.
    """
    df = pd.read_csv(master_csv)

    required_columns = {"language", "audio_file_path"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Master CSV is missing required columns: {sorted(missing_columns)}"
        )

    chapters = (
        df[["language", "audio_file_path"]]
        .drop_duplicates()
        .sort_values(["language", "audio_file_path"])
    )

    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Total chapter audio files to check: {len(chapters)}", flush=True)
    print(f"Master CSV: {master_csv}", flush=True)
    print(f"Audio root: {audio_root}", flush=True)
    print(f"Output root: {out_root}", flush=True)

    for idx, row in chapters.reset_index(drop=True).iterrows():
        language = row["language"]
        chapter_id = Path(row["audio_file_path"]).stem
        audio_path = audio_root / language / f"{chapter_id}.mp3"

        out_dir = out_root / language
        out_dir.mkdir(parents=True, exist_ok=True)

        txt_path = out_dir / f"{chapter_id}.txt"
        json_path = out_dir / f"{chapter_id}.json"

        print("=" * 80, flush=True)
        print(f"[{idx + 1}/{len(chapters)}] {language} {chapter_id}", flush=True)
        print(f"Audio: {audio_path}", flush=True)

        if not audio_path.exists():
            raise FileNotFoundError(f"Missing audio file: {audio_path}")

        if txt_path.exists() and json_path.exists():
            print("Already completed. Skipping.", flush=True)
            continue

        detected_language = detect_language(
            audio_path=audio_path,
            model_name=model_name,
            seconds=detect_seconds,
        )

        transcribe_full_audio(
            audio_path=audio_path,
            detected_language=detected_language,
            out_dir=out_dir,
            model_name=model_name,
        )

        print(f"Wrote: {txt_path}", flush=True)
        print(f"Wrote: {json_path}", flush=True)

    print("=" * 80, flush=True)
    print("Done transcribing.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--master-csv",
        default="asr_evaluation_master.csv",
        help="Master CSV containing language and chapter metadata.",
    )

    parser.add_argument(
        "--audio-root",
        default="audio_dataset",
        help=(
            "Root directory containing chapter audio files. "
            "Audio paths are constructed as "
            "<audio-root>/<language>/<chapter_id>.mp3."
        ),
    )

    parser.add_argument(
        "--model-name",
        default="large-v3",
        help="Whisper model name.",
    )

    parser.add_argument(
        "--out-root",
        default="transcriptions/whisper_large_v3",
        help="Directory where Whisper outputs should be written.",
    )

    parser.add_argument(
        "--detect-seconds",
        type=int,
        default=30,
        help="Number of seconds used for language detection.",
    )

    args = parser.parse_args()

    transcribe_from_master(
        master_csv=Path(args.master_csv),
        audio_root=Path(args.audio_root),
        out_root=Path(args.out_root),
        model_name=args.model_name,
        detect_seconds=args.detect_seconds,
    )


if __name__ == "__main__":
    main()