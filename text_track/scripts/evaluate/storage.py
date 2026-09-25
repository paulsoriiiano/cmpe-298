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
    config_fingerprint: str          # see models.config_fingerprint() — hashes exact model/serving config
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
    question_en_tokens: int | None     # model-native tokenizer count where available — see tokenization.py
    question_ilo_tokens: int | None
    translation_tokens: int | None      # None when this record has no translation (e.g. reason/direct stage)
    rationale_tokens: int | None        # None when this record has no rationale (e.g. translate stage)
    tokenization_tax_ratio: float | None  # question_ilo_tokens / question_en_tokens
    tokenizer_model_id: str | None
    tokenizer_revision: str | None
    word_count: int | None            # descriptive only — not a substitute for token counts
    char_count: int | None
    finish_reason: str | None
    latency_ms: float
    retry_count: int
    error_message: str | None
    timestamp: str

    def resume_key(self) -> tuple:
        return make_resume_key(
            dataset_version=self.dataset_version, protocol_version=self.protocol_version,
            prompt_version=self.prompt_version, config_fingerprint=self.config_fingerprint,
            item_id=self.item_id, model_key=self.model_key, condition_key=self.condition_key,
            stage=self.stage,
        )


def make_resume_key(
    *, dataset_version: str, protocol_version: str, prompt_version: str,
    config_fingerprint: str, item_id: str, model_key: str, condition_key: str, stage: str,
) -> tuple:
    """Single source of truth for the resume-key shape, used both by ResultRecord.resume_key()
    and by run.py's lookups BEFORE a record exists — keeping both in sync is the whole point
    of factoring this out (a prior version of this key was built ad hoc in two places and
    drifted out of sync when prompt_version was added to one but not the other).

    Includes prompt_version so a prompt-wording change (even under the same protocol_version)
    can't accidentally resume/reuse a stale result, and config_fingerprint so a changed model
    configuration (different checkpoint, precision, sampling settings, serving config) can't
    either. In practice prompt_version changes are expected to ship alongside a
    protocol_version bump (see conditions.PROTOCOL_VERSION), but this doesn't rely on that
    discipline.
    """
    return (dataset_version, protocol_version, prompt_version, config_fingerprint, item_id,
            model_key, condition_key, stage)


NON_RETRYABLE_FAILURE_TYPES = {
    "correct", "substantively_incorrect", "invalid_answer_format", "missing_answer",
    "translation_completed", "translation_format_failure", "refusal", "truncation",
    "repetition_degeneration", "parser_failure",
}


_RESULT_RECORD_FIELDS = {f.name for f in dataclasses.fields(ResultRecord)}


def _lenient_record_from_row(row: dict) -> ResultRecord | None:
    """Tolerates JSONL rows from an older schema (e.g. a protocol_v1 file read by a v2
    evaluator that added new fields like config_fingerprint/tokenization_tax_ratio) by
    filling any missing fields with None, rather than crashing the whole run. Unknown keys
    in the row (from an even newer schema) are silently dropped. Returns None if the row
    can't be turned into a ResultRecord at all (e.g. it's not a dict, or a required value
    has the wrong type) — such a row is skipped rather than guessed at; it simply won't
    match any current lookup key, so a genuinely incompatible/corrupt row can't accidentally
    be treated as completed work."""
    if not isinstance(row, dict):
        return None
    filtered = {k: v for k, v in row.items() if k in _RESULT_RECORD_FIELDS}
    for name in _RESULT_RECORD_FIELDS:
        filtered.setdefault(name, None)
    try:
        return ResultRecord(**filtered)
    except TypeError:
        return None


class ResumeIndex:
    """Tracks which (dataset_version, protocol_version, prompt_version, config_fingerprint,
    item_id, model_key, condition_key, stage) units are already completed, scanned from
    every existing eval_runs/*.jsonl file. Units whose recorded failure_type is retry-eligible
    (infrastructure_api_failure) are NOT considered complete."""

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
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # a corrupt/truncated line shouldn't crash the whole run
                    record = _lenient_record_from_row(row)
                    if record is not None:
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


# Manifest fields that must match exactly if a run_id is reused (e.g. restarting a failed
# SLURM job under the same predetermined run_id) — anything else (total_planned_units,
# resuming_from_run_ids) is allowed to differ. git_commit is included: resuming the same
# run_id from a different evaluator commit could mean different prompts/grading/tokenization
# code produced the "same" result schema, which is exactly the kind of silent drift this
# manifest exists to catch. planned_item_ids_by_condition is included: a restart that
# selects a different item set for the same run_id would silently mix two different
# experiments' data into one result file.
_MANIFEST_COMPATIBILITY_FIELDS = (
    "dataset_version", "protocol_version", "conditions", "models", "model_settings",
    "git_commit", "planned_item_ids_by_condition",
)


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

    If a manifest already exists for this run_id (e.g. restarting a failed SLURM job under
    the same predetermined run_id — see __main__.py's --run-id), its dataset/protocol
    version, conditions, models, model_settings, git_commit, and item selection must match
    exactly, or this raises rather than silently overwriting a manifest that no longer
    describes what's actually in the result JSONL for that run_id. On a compatible restart,
    the original created_at is preserved and last_resumed_at is set to now.
    """
    import datetime

    os.makedirs(runs_dir, exist_ok=True)
    path = os.path.join(runs_dir, f"{run_id}.manifest.json")

    new_values = {
        "dataset_version": dataset_version, "protocol_version": protocol_version,
        "conditions": conditions, "models": models, "model_settings": model_settings or {},
        "git_commit": _git_commit(),
        "planned_item_ids_by_condition": planned_item_ids_by_condition or {},
    }

    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    last_resumed_at = None
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            existing = json.load(f)
        mismatches = {
            field: (existing.get(field), new_values[field])
            for field in _MANIFEST_COMPATIBILITY_FIELDS
            if existing.get(field) != new_values[field]
        }
        if mismatches:
            raise ValueError(
                f"Refusing to overwrite manifest for run_id={run_id!r}: incompatible "
                f"settings would be silently applied to an existing run. Mismatches "
                f"(existing vs. new): {mismatches}. Use a new run_id for a genuinely "
                f"different configuration."
            )
        created_at = existing["created_at"]  # preserve the ORIGINAL creation time
        last_resumed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    manifest = {
        "run_id": run_id,
        "created_at": created_at,
        "last_resumed_at": last_resumed_at,
        **new_values,
        "total_planned_units": total_planned_units,
        "resuming_from_run_ids": resuming_from_run_ids or [],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return path
