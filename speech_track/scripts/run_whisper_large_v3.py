import re
import subprocess
import tempfile
import pandas as pd
from pathlib import Path

"""
Two-pass Whisper Large-v3 transcription.

Pass 1:
    Create a temporary 30-second audio clip from the beginning of the chapter.
    Run Whisper on only that 30-second clip to detect the language.

Pass 2:
    Transcribe the full original audio from 0:00 using the language Whisper detected.

This avoids losing the first 30 seconds during auto-detect mode, while still using
Whisper's own language choice rather than manually forcing a language.
"""

MASTER_CSV = "asr_evaluation_master.csv"
MODEL_NAME = "large-v3"
OUT_ROOT = Path("transcriptions/whisper_large_v3")


def make_30_second_clip(audio_path, clip_path):
    """
    Use ffmpeg to create a temporary 30-second WAV clip for language detection.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(audio_path),
        "-t", "30",
        str(clip_path),
    ]

    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def detect_language(audio_path):
    """
    Run Whisper on only the first 30 seconds and parse the detected language.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        clip_path = tmpdir / "first_30_seconds.wav"
        detect_out = tmpdir / "detect_out"
        detect_out.mkdir(parents=True, exist_ok=True)

        print("  [Pass 1] Creating 30-second clip for language detection...", flush=True)
        make_30_second_clip(audio_path, clip_path)

        print("  [Pass 1] Running Whisper language detection on first 30 seconds...", flush=True)

        cmd = [
            "whisper",
            str(clip_path),
            "--model", MODEL_NAME,
            "--task", "transcribe",
            "--temperature", "0",
            "--output_format", "txt",
            "--output_dir", str(detect_out),
            "--fp16", "False",
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


def transcribe_full_audio(audio_path, detected_language, out_dir):
    """
    Transcribe the full original audio using the language detected in Pass 1.
    """
    print("  [Pass 2] Transcribing full audio from 0:00...", flush=True)

    cmd = [
        "whisper",
        str(audio_path),
        "--model", MODEL_NAME,
        "--task", "transcribe",
        "--language", detected_language,
        "--temperature", "0",
        "--output_format", "all",
        "--output_dir", str(out_dir),
        "--fp16", "False",
    ]

    subprocess.run(cmd, check=True)

    print("  [Pass 2] Finished full transcription.", flush=True)


def main():
    df = pd.read_csv(MASTER_CSV)

    audio_files = (
        df[["language", "audio_file_path"]]
        .drop_duplicates()
        .sort_values(["language", "audio_file_path"])
    )

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Total chapter audio files to check: {len(audio_files)}", flush=True)

    for idx, row in audio_files.reset_index(drop=True).iterrows():
        language = row["language"]
        audio_path = Path(row["audio_file_path"])
        chapter_id = audio_path.stem

        out_dir = OUT_ROOT / language
        out_dir.mkdir(parents=True, exist_ok=True)

        txt_path = out_dir / f"{chapter_id}.txt"
        json_path = out_dir / f"{chapter_id}.json"

        print("=" * 80, flush=True)
        print(f"[{idx + 1}/{len(audio_files)}] {language} {chapter_id}", flush=True)
        print(f"Audio: {audio_path}", flush=True)

        if txt_path.exists() and json_path.exists():
            print("Already completed. Skipping.", flush=True)
            continue

        detected_language = detect_language(audio_path)
        transcribe_full_audio(audio_path, detected_language, out_dir)

        print(f"Wrote: {txt_path}", flush=True)
        print(f"Wrote: {json_path}", flush=True)

    print("=" * 80, flush=True)
    print("Done transcribing.", flush=True)


if __name__ == "__main__":
    main()