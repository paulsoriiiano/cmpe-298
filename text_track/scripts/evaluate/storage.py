"""Result record schema, JSONL storage, resumption index, and run manifests.

One ResultRecord per (item, model, condition, stage). Stored one JSONL file per run under
data/eval_runs/. Resumption is keyed WITHOUT run_id (see ResumeIndex) so restarting under a
fresh run_id still skips already-completed work for the same dataset/protocol version.
"""
import dataclasses
import json
import os
from dataclasses import asdict, dataclass, field

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(SCRIPT_DIR, "..", "..", "data", "eval_runs")


@dataclass
class ResultRecord:
    run_id: str
    dataset_version: str
    protocol_version: str
    item_id: str
    source: str
    model_key: str
    condition_key: str
    stage: str                      # "direct" | "translate" | "reason"
    prompt_version: str
    original_input: str
    generated_translation: str | None
    generated_rationale: str | None
    raw_response: str | None
    extracted_answer: str | None
    canonical_answer: str
    is_correct: bool | None
    format_compliant: bool | None
    failure_type: str
    input_tokens: int | None          # exact, from the provider's usage object
    output_tokens: int | None         # exact, from the provider's usage object
    system_prompt_tokens: int | None  # local tokenizer estimate — see tokenization.py
    user_input_tokens: int | None     # local tokenizer estimate
    rendered_prompt_tokens: int | None  # local tokenizer estimate; sanity-check vs input_tokens
    tokenizer_id: str | None
    tokenizer_revision: str | None
    finish_reason: str | None
    latency_ms: float
    retry_count: int
    error_message: str | None
    timestamp: str

    def resume_key(self) -> tuple:
        return make_resume_key(
            dataset_version=self.dataset_version, protocol_version=self.protocol_version,
            prompt_version=self.prompt_version, item_id=self.item_id, model_key=self.model_key,
            condition_key=self.condition_key, stage=self.stage,
        )


def make_resume_key(
    *, dataset_version: str, protocol_version: str, prompt_version: str, item_id: str,
    model_key: str, condition_key: str, stage: str,
) -> tuple:
    """Single source of truth for the resume-key shape, used both by ResultRecord.resume_key()
    and by run.py's lookups BEFORE a record exists — keeping both in sync is the whole point
    of factoring this out (a prior version of this key was built ad hoc in two places and
    drifted out of sync when prompt_version was added to one but not the other).

    Includes prompt_version so a prompt-wording change (even under the same protocol_version)
    can't accidentally resume/reuse a stale result. In practice prompt_version changes are
    expected to ship alongside a protocol_version bump (see conditions.PROTOCOL_VERSION), but
    this doesn't rely on that discipline.
    """
    return (dataset_version, protocol_version, prompt_version, item_id, model_key,
            condition_key, stage)


NON_RETRYABLE_FAILURE_TYPES = {
    "correct", "substantively_incorrect", "invalid_answer_format", "missing_answer",
    "translation_completed", "translation_format_failure", "refusal", "truncation",
    "repetition_degeneration", "parser_failure",
}


class ResumeIndex:
    """Tracks which (dataset_version, protocol_version, item_id, model_key, condition_key,
    stage) units are already completed, scanned from every existing eval_runs/*.jsonl file.
    Units whose recorded failure_type is retry-eligible (infrastructure_api_failure) are NOT
    considered complete."""

    def __init__(self):
        self._completed: dict[tuple, ResultRecord] = {}

    @classmethod
    def load_from_runs_dir(cls, runs_dir: str = RUNS_DIR) -> "ResumeIndex":
        index = cls()
        if not os.path.isdir(runs_dir):
            return index
        for name in sorted(os.listdir(runs_dir)):
            if not name.endswith(".jsonl"):
                continue
            path = os.path.join(runs_dir, name)
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    record = ResultRecord(**row)
                    index.record_if_newer(record)
        return index

    def record_if_newer(self, record: ResultRecord) -> None:
        key = record.resume_key()
        existing = self._completed.get(key)
        if existing is None:
            self._completed[key] = record
            return
        # A later completed (non-retryable) record supersedes an earlier retryable failure.
        if existing.failure_type not in NON_RETRYABLE_FAILURE_TYPES and \
                record.failure_type in NON_RETRYABLE_FAILURE_TYPES:
            self._completed[key] = record

    def is_complete(self, key: tuple) -> bool:
        record = self._completed.get(key)
        return record is not None and record.failure_type in NON_RETRYABLE_FAILURE_TYPES

    def get(self, key: tuple) -> ResultRecord | None:
        return self._completed.get(key)


class RunWriter:
    """Appends ResultRecords to data/eval_runs/{run_id}.jsonl, one line per record, flushing
    after every write so a crash never loses a completed unit of work."""

    def __init__(self, run_id: str, runs_dir: str = RUNS_DIR):
        self.run_id = run_id
        os.makedirs(runs_dir, exist_ok=True)
        self._path = os.path.join(runs_dir, f"{run_id}.jsonl")
        self._file = open(self._path, "a", encoding="utf-8")

    def append(self, record: ResultRecord) -> None:
        self._file.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def _git_commit() -> str | None:
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
            cwd=SCRIPT_DIR, check=True,
        )
        return result.stdout.strip()
    except Exception:  # noqa: BLE001 - not a git repo, git missing, etc. — manifest still writes
        return None


def write_run_manifest(
    *,
    run_id: str,
    dataset_version: str,
    protocol_version: str,
    conditions: list[str],
    models: list[str],
    total_planned_units: int,
    planned_item_ids_by_condition: dict[str, list[str]] | None = None,
    resuming_from_run_ids: list[str] | None = None,
    model_settings: dict[str, dict] | None = None,
    runs_dir: str = RUNS_DIR,
) -> str:
    """Write the run manifest BEFORE any API calls are made. Returns the manifest path.

    model_settings should map model_key -> a dict of reproducibility metadata (exact model
    ID, revision, provider, temperature, max tokens, context length, thinking mode, seed,
    precision, and — for HPC/vLLM models — vLLM version and FlashInfer sampler setting).
    Fields not knowable for a given provider (e.g. hosted APIs don't expose precision) are
    left null by the caller rather than guessed here.
    """
    import datetime

    os.makedirs(runs_dir, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "dataset_version": dataset_version,
        "protocol_version": protocol_version,
        "conditions": conditions,
        "models": models,
        "model_settings": model_settings or {},
        "total_planned_units": total_planned_units,
        "planned_item_ids_by_condition": planned_item_ids_by_condition or {},
        "resuming_from_run_ids": resuming_from_run_ids or [],
    }
    path = os.path.join(runs_dir, f"{run_id}.manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return path
