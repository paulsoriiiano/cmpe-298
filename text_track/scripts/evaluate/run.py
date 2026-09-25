"""Orchestration: load dataset, build the work plan, execute conditions x models, write
resumable ResultRecords. See text_track/PROTOCOL.md for the experimental design and the
Phase 3 plan for the resumption/storage design this implements.
"""
import datetime
import json
import os
import uuid

from . import conditions as cond_mod
from . import grading
from . import tokenization
from .conditions import (
    CONDITIONS, IMPLEMENTED_CONDITIONS, STRATIFIED_SUBSET_CONDITIONS, get_condition,
    require_implemented,
)
from .models import MODEL_REGISTRY, complete_with_retry, manifest_settings_for
from .storage import RUNS_DIR, ResultRecord, ResumeIndex, RunWriter, make_resume_key, write_run_manifest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "..", "..", "data", "dataset.jsonl")
MANIFEST_PATH = os.path.join(SCRIPT_DIR, "..", "..", "data", "dataset_manifest.json")
STRATIFIED_SUBSET_PATH = os.path.join(SCRIPT_DIR, "..", "..", "data", "stratified_subset_300.json")


def load_dataset(path: str = DATASET_PATH) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def load_dataset_version(path: str = MANIFEST_PATH) -> str:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["version"]


def load_stratified_subset_ids(path: str = STRATIFIED_SUBSET_PATH) -> list[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run text_track/scripts/generate_stratified_subset.py first "
            f"to generate the fixed 300-item subset A_E0/A_I0 require."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)["ids"]


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _run_single_call_condition(
    *, item, model_key, condition, dataset_version, protocol_version, run_id, resume_index,
    writer,
):
    """A_EE / A_II: one call, question -> rationale -> canonical answer."""
    key = make_resume_key(
        dataset_version=dataset_version, protocol_version=protocol_version,
        prompt_version=condition.prompt_version, item_id=item["id"], model_key=model_key,
        condition_key=condition.key, stage="reason",
    )
    if resume_index.is_complete(key):
        return resume_index.get(key)

    system = cond_mod.PROMPT_A_EE if condition.key == "A_EE" else cond_mod.PROMPT_A_II
    user_input = item[condition.input_field]

    response, exception, retry_count, latency_ms = complete_with_retry(
        model_key, system=system, user=user_input,
    )
    failure_type, extracted, is_correct, format_compliant = grading.classify_result(
        response=response, exception=exception, canonical_answer=item["canonical_answer"],
        source=item["source"], stage="reason",
    )
    record = ResultRecord(
        run_id=run_id, dataset_version=dataset_version, protocol_version=protocol_version,
        item_id=item["id"], source=item["source"], model_key=model_key,
        condition_key=condition.key, stage="reason", prompt_version=condition.prompt_version,
        original_input=user_input, generated_translation=None,
        generated_rationale=response.text if response else None,
        raw_response=response.text if response else None,
        extracted_answer=extracted, canonical_answer=item["canonical_answer"],
        is_correct=is_correct, format_compliant=format_compliant,
        failure_type=failure_type.value,
        input_tokens=response.input_tokens if response else None,
        output_tokens=response.output_tokens if response else None,
        **tokenization.count_prompt_tokens(system, user_input),
        finish_reason=response.finish_reason if response else None,
        latency_ms=latency_ms, retry_count=retry_count,
        error_message=str(exception) if exception else None, timestamp=_now_iso(),
    )
    writer.append(record)
    resume_index.record_if_newer(record)
    return record


def _run_direct_condition(
    *, item, model_key, condition, dataset_version, protocol_version, run_id, resume_index,
    writer,
):
    """A_E0 / A_I0: one call, question -> canonical answer directly, no rationale. Runs only
    over the fixed 300-item stratified subset (see STRATIFIED_SUBSET_CONDITIONS)."""
    key = make_resume_key(
        dataset_version=dataset_version, protocol_version=protocol_version,
        prompt_version=condition.prompt_version, item_id=item["id"], model_key=model_key,
        condition_key=condition.key, stage="direct",
    )
    if resume_index.is_complete(key):
        return resume_index.get(key)

    system = cond_mod.PROMPT_A_E0 if condition.key == "A_E0" else cond_mod.PROMPT_A_I0
    user_input = item[condition.input_field]

    response, exception, retry_count, latency_ms = complete_with_retry(
        model_key, system=system, user=user_input,
    )
    failure_type, extracted, is_correct, format_compliant = grading.classify_result(
        response=response, exception=exception, canonical_answer=item["canonical_answer"],
        source=item["source"], stage="direct",
    )
    record = ResultRecord(
        run_id=run_id, dataset_version=dataset_version, protocol_version=protocol_version,
        item_id=item["id"], source=item["source"], model_key=model_key,
        condition_key=condition.key, stage="direct", prompt_version=condition.prompt_version,
        original_input=user_input, generated_translation=None, generated_rationale=None,
        raw_response=response.text if response else None,
        extracted_answer=extracted, canonical_answer=item["canonical_answer"],
        is_correct=is_correct, format_compliant=format_compliant,
        failure_type=failure_type.value,
        input_tokens=response.input_tokens if response else None,
        output_tokens=response.output_tokens if response else None,
        **tokenization.count_prompt_tokens(system, user_input),
        finish_reason=response.finish_reason if response else None,
        latency_ms=latency_ms, retry_count=retry_count,
        error_message=str(exception) if exception else None, timestamp=_now_iso(),
    )
    writer.append(record)
    resume_index.record_if_newer(record)
    return record


def _run_staged_pivot_condition(
    *, item, model_key, condition, dataset_version, protocol_version, run_id, resume_index,
    writer,
):
    """A_IE / A_EI: translate (separate call, saved) -> fresh context -> reason using only
    the saved translation. Original question and canonical answer are never included in the
    reasoning-stage prompt."""
    translate_key = make_resume_key(
        dataset_version=dataset_version, protocol_version=protocol_version,
        prompt_version=condition.prompt_version, item_id=item["id"], model_key=model_key,
        condition_key=condition.key, stage="translate",
    )
    reason_key = make_resume_key(
        dataset_version=dataset_version, protocol_version=protocol_version,
        prompt_version=condition.prompt_version, item_id=item["id"], model_key=model_key,
        condition_key=condition.key, stage="reason",
    )

    translate_record = resume_index.get(translate_key)
    if translate_record is None or not resume_index.is_complete(translate_key):
        translation_system = (
            cond_mod.TRANSLATE_ILO_TO_EN_SYSTEM if condition.translation_source_lang == "ilo"
            else cond_mod.TRANSLATE_EN_TO_ILO_SYSTEM
        )
        user_input = item[condition.input_field]

        response, exception, retry_count, latency_ms = complete_with_retry(
            model_key, system=translation_system, user=user_input,
        )
        failure_type, _, _, format_compliant = grading.classify_result(
            response=response, exception=exception, canonical_answer=None,
            source=item["source"], stage="translate",
        )
        translate_record = ResultRecord(
            run_id=run_id, dataset_version=dataset_version, protocol_version=protocol_version,
            item_id=item["id"], source=item["source"], model_key=model_key,
            condition_key=condition.key, stage="translate", prompt_version=condition.prompt_version,
            original_input=user_input, generated_translation=response.text if response else None,
            generated_rationale=None, raw_response=response.text if response else None,
            extracted_answer=None, canonical_answer=item["canonical_answer"],
            is_correct=None, format_compliant=format_compliant,
            failure_type=failure_type.value,
            input_tokens=response.input_tokens if response else None,
            output_tokens=response.output_tokens if response else None,
            **tokenization.count_prompt_tokens(translation_system, user_input),
            finish_reason=response.finish_reason if response else None,
            latency_ms=latency_ms, retry_count=retry_count,
            error_message=str(exception) if exception else None, timestamp=_now_iso(),
        )
        writer.append(translate_record)
        resume_index.record_if_newer(translate_record)

    if translate_record.failure_type != grading.FailureType.TRANSLATION_COMPLETED.value:
        # Translation stage did not produce a usable translation (including infra failure);
        # do not proceed to the reasoning stage.
        return [translate_record]

    if resume_index.is_complete(reason_key):
        return [translate_record, resume_index.get(reason_key)]

    reasoning_system = (
        cond_mod.REASON_EN_FROM_TRANSLATION_SYSTEM if condition.reasoning_lang == "en"
        else cond_mod.REASON_ILO_FROM_TRANSLATION_SYSTEM
    )
    # Fresh context: only the saved translation is sent. The original-language question and
    # the canonical answer are never included in this prompt.
    translated_text = translate_record.generated_translation or ""

    response, exception, retry_count, latency_ms = complete_with_retry(
        model_key, system=reasoning_system, user=translated_text,
    )
    failure_type, extracted, is_correct, format_compliant = grading.classify_result(
        response=response, exception=exception, canonical_answer=item["canonical_answer"],
        source=item["source"], stage="reason",
    )
    reason_record = ResultRecord(
        run_id=run_id, dataset_version=dataset_version, protocol_version=protocol_version,
        item_id=item["id"], source=item["source"], model_key=model_key,
        condition_key=condition.key, stage="reason", prompt_version=condition.prompt_version,
        original_input=translated_text, generated_translation=None,
        generated_rationale=response.text if response else None,
        raw_response=response.text if response else None,
        extracted_answer=extracted, canonical_answer=item["canonical_answer"],
        is_correct=is_correct, format_compliant=format_compliant,
        failure_type=failure_type.value,
        input_tokens=response.input_tokens if response else None,
        output_tokens=response.output_tokens if response else None,
        **tokenization.count_prompt_tokens(reasoning_system, translated_text),
        finish_reason=response.finish_reason if response else None,
        latency_ms=latency_ms, retry_count=retry_count,
        error_message=str(exception) if exception else None, timestamp=_now_iso(),
    )
    writer.append(reason_record)
    resume_index.record_if_newer(reason_record)
    return [translate_record, reason_record]


def run_evaluation(
    *, condition_keys: list[str], model_keys: list[str], limit: int | None = None,
    item_ids: list[str] | None = None,
    dataset_path: str = DATASET_PATH, manifest_path: str = MANIFEST_PATH,
    runs_dir: str = RUNS_DIR, run_id: str | None = None,
):
    """item_ids, if given, restricts every condition to exactly those item IDs (e.g. a
    hand-picked source-diverse pilot slice — one item per source — per PROTOCOL.md section
    9's pilot requirement) instead of running the full dataset / stratified subset. Applied
    before `limit`, which still works as an additional cap if both are given."""
    for key in condition_keys:
        require_implemented(key)
    for key in model_keys:
        if key not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {key!r}. Known: {list(MODEL_REGISTRY)}")

    all_items = load_dataset(dataset_path)
    if item_ids is not None:
        wanted = set(item_ids)
        all_items = [item for item in all_items if item["id"] in wanted]
        missing = wanted - {item["id"] for item in all_items}
        if missing:
            raise ValueError(f"item_ids not found in dataset: {sorted(missing)}")
    dataset_version = load_dataset_version(manifest_path)
    protocol_version = cond_mod.PROTOCOL_VERSION

    # Per-condition item pools: A_E0/A_I0 normally run only over the fixed 300-item
    # stratified subset (per PROTOCOL.md section 1); every other condition runs over the
    # full dataset. When item_ids is given (e.g. a hand-picked pilot slice), it overrides
    # the stratified-subset restriction too — the point of a pilot is to exercise every
    # requested condition on exactly the chosen items, not to silently drop A_E0/A_I0
    # because none of those IDs happened to land in the random 300-item subset.
    item_pools: dict[str, list[dict]] = {}
    if item_ids is not None:
        stratified_items = all_items
    elif any(key in STRATIFIED_SUBSET_CONDITIONS for key in condition_keys):
        subset_ids = set(load_stratified_subset_ids())
        stratified_items = [item for item in all_items if item["id"] in subset_ids]
    else:
        stratified_items = None
    for condition_key in condition_keys:
        pool = stratified_items if condition_key in STRATIFIED_SUBSET_CONDITIONS else all_items
        item_pools[condition_key] = pool[:limit] if limit else pool

    run_id = run_id or str(uuid.uuid4())
    total_planned_units = sum(len(item_pools[k]) for k in condition_keys) * len(model_keys)
    write_run_manifest(
        run_id=run_id, dataset_version=dataset_version, protocol_version=protocol_version,
        conditions=condition_keys, models=model_keys, total_planned_units=total_planned_units,
        planned_item_ids_by_condition={k: [item["id"] for item in item_pools[k]] for k in condition_keys},
        model_settings={key: manifest_settings_for(key) for key in model_keys},
        runs_dir=runs_dir,
    )

    resume_index = ResumeIndex.load_from_runs_dir(runs_dir)
    writer = RunWriter(run_id, runs_dir=runs_dir)
    try:
        for condition_key in condition_keys:
            condition = get_condition(condition_key)
            for item in item_pools[condition_key]:
                for model_key in model_keys:
                    if condition.new_context_for_reasoning:
                        handler = _run_staged_pivot_condition
                    elif condition.reasoning_lang is None:
                        handler = _run_direct_condition
                    else:
                        handler = _run_single_call_condition
                    handler(
                        item=item, model_key=model_key, condition=condition,
                        dataset_version=dataset_version, protocol_version=protocol_version,
                        run_id=run_id, resume_index=resume_index, writer=writer,
                    )
    finally:
        writer.close()

    return run_id
