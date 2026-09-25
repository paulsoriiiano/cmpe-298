"""Unit tests for storage.py's resume-key construction, manifest writer, and ResumeIndex's
tolerance of older result schemas.

Regression coverage for a real bug: ResultRecord.resume_key() and run.py's ad hoc lookup
tuples used to be built independently and drifted out of sync when prompt_version was added
to one but not the other, silently defeating resumption (every lookup missed, so completed
work was always re-run). make_resume_key() is now the single source of truth for both.
"""
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from ..storage import ResultRecord, ResumeIndex, make_resume_key, write_run_manifest


def _make_record(**overrides):
    defaults = dict(
        run_id="run-1", dataset_version="dataset_conference_v1.1",
        protocol_version="protocol_v2", item_id="gsm8k_0", source="gsm8k",
        model_key="claude_sonnet_4_6", config_fingerprint="abc123",
        condition_key="A_EE", stage="reason",
        prompt_version="v2", original_input="q", generated_translation=None,
        generated_rationale="r", raw_response="r", extracted_answer="109",
        canonical_answer="109", is_correct=True, format_compliant=True,
        failure_type="correct", input_tokens=10, output_tokens=5,
        question_en_tokens=3, question_ilo_tokens=4, translation_tokens=None,
        rationale_tokens=1, tokenization_tax_ratio=1.33,
        tokenizer_model_id="tiktoken/cl100k_base", tokenizer_revision="0.14.0",
        word_count=1, char_count=1,
        finish_reason="stop", latency_ms=1.0, retry_count=0, error_message=None,
        timestamp="2026-01-01T00:00:00+00:00",
    )
    defaults.update(overrides)
    return ResultRecord(**defaults)


class ResumeKeyTests(unittest.TestCase):
    def test_resume_key_includes_prompt_version(self):
        key_v2 = _make_record(prompt_version="v2").resume_key()
        key_v1 = _make_record(prompt_version="v1").resume_key()
        self.assertNotEqual(key_v2, key_v1)

    def test_resume_key_includes_config_fingerprint(self):
        key_a = _make_record(config_fingerprint="fp-a").resume_key()
        key_b = _make_record(config_fingerprint="fp-b").resume_key()
        self.assertNotEqual(key_a, key_b)

    def test_make_resume_key_matches_instance_method(self):
        record = _make_record()
        via_instance = record.resume_key()
        via_function = make_resume_key(
            dataset_version=record.dataset_version, protocol_version=record.protocol_version,
            prompt_version=record.prompt_version, config_fingerprint=record.config_fingerprint,
            item_id=record.item_id, model_key=record.model_key,
            condition_key=record.condition_key, stage=record.stage,
        )
        self.assertEqual(via_instance, via_function)


class ResumeIndexLenientLoadingTests(unittest.TestCase):
    """A protocol_v1 result file (written before config_fingerprint/tokenization_tax_ratio
    etc. existed) must not crash a protocol_v2 evaluator run — it should load with the
    missing fields as None and simply never match a v2 lookup key (different
    protocol_version/config_fingerprint), rather than raising."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)

    def _write_jsonl(self, name, rows):
        path = os.path.join(self.tmp_dir, name)
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    def test_old_schema_row_missing_new_fields_does_not_crash(self):
        old_row = {
            "run_id": "old-run", "dataset_version": "dataset_conference_v1.1",
            "protocol_version": "protocol_v1", "item_id": "gsm8k_0", "source": "gsm8k",
            "model_key": "qwen_3_6_27b", "condition_key": "A_EE", "stage": "reason",
            "prompt_version": "v1", "original_input": "q", "generated_translation": None,
            "generated_rationale": "r", "raw_response": "r", "extracted_answer": "109",
            "canonical_answer": "109", "is_correct": True, "format_compliant": True,
            "failure_type": "correct", "input_tokens": 10, "output_tokens": 5,
            # Missing: config_fingerprint, question_en_tokens, tokenization_tax_ratio, etc.
            "finish_reason": "stop", "latency_ms": 1.0, "retry_count": 0,
            "error_message": None, "timestamp": "2026-01-01T00:00:00+00:00",
        }
        self._write_jsonl("old.jsonl", [old_row])

        index = ResumeIndex.load_from_runs_dir(self.tmp_dir)  # must not raise

        # The old row's key includes config_fingerprint=None, so it can never match a real
        # v2 lookup (which always has an actual fingerprint string).
        v2_key = make_resume_key(
            dataset_version="dataset_conference_v1.1", protocol_version="protocol_v2",
            prompt_version="v2", config_fingerprint="real-fingerprint-hash",
            item_id="gsm8k_0", model_key="qwen_3_6_27b", condition_key="A_EE", stage="reason",
        )
        self.assertFalse(index.is_complete(v2_key))

    def test_corrupt_line_is_skipped_not_fatal(self):
        path = os.path.join(self.tmp_dir, "broken.jsonl")
        with open(path, "w") as f:
            f.write("{not valid json\n")
        index = ResumeIndex.load_from_runs_dir(self.tmp_dir)  # must not raise
        self.assertEqual(len(index._completed), 0)

    def test_well_formed_v2_row_still_loads_normally(self):
        record = _make_record()
        self._write_jsonl("new.jsonl", [record.__dict__])
        index = ResumeIndex.load_from_runs_dir(self.tmp_dir)
        self.assertTrue(index.is_complete(record.resume_key()))


class RunManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)

    def test_manifest_includes_git_commit_and_model_settings(self):
        path = write_run_manifest(
            run_id="run-1", dataset_version="dataset_conference_v1.1",
            protocol_version="protocol_v2", conditions=["A_EE"],
            models=["claude_sonnet_4_6"], total_planned_units=1,
            model_settings={"claude_sonnet_4_6": {"model_id": "claude-sonnet-4-6"}},
            runs_dir=self.tmp_dir,
        )
        with open(path) as f:
            manifest = json.load(f)
        self.assertIn("git_commit", manifest)  # may be None outside a git repo, but must be present
        self.assertEqual(
            manifest["model_settings"]["claude_sonnet_4_6"]["model_id"], "claude-sonnet-4-6",
        )

    def test_resuming_from_run_ids_recorded(self):
        path = write_run_manifest(
            run_id="run-1", dataset_version="dataset_conference_v1.1",
            protocol_version="protocol_v2", conditions=["A_EE"],
            models=["claude_sonnet_4_6"], total_planned_units=1,
            resuming_from_run_ids=["earlier-run"], runs_dir=self.tmp_dir,
        )
        with open(path) as f:
            manifest = json.load(f)
        self.assertEqual(manifest["resuming_from_run_ids"], ["earlier-run"])

    def test_same_run_id_compatible_settings_does_not_raise(self):
        kwargs = dict(
            run_id="run-1", dataset_version="dataset_conference_v1.1",
            protocol_version="protocol_v2", conditions=["A_EE"],
            models=["claude_sonnet_4_6"], total_planned_units=1,
            model_settings={"claude_sonnet_4_6": {"model_id": "claude-sonnet-4-6"}},
            runs_dir=self.tmp_dir,
        )
        write_run_manifest(**kwargs)
        write_run_manifest(**kwargs)  # simulates restarting a failed job under the same run_id

    def test_same_run_id_incompatible_settings_raises(self):
        write_run_manifest(
            run_id="run-1", dataset_version="dataset_conference_v1.1",
            protocol_version="protocol_v2", conditions=["A_EE"],
            models=["claude_sonnet_4_6"], total_planned_units=1, runs_dir=self.tmp_dir,
        )
        with self.assertRaises(ValueError):
            write_run_manifest(
                run_id="run-1", dataset_version="dataset_conference_v1.1",
                protocol_version="protocol_v2", conditions=["A_EE", "A_II"],  # different!
                models=["claude_sonnet_4_6"], total_planned_units=2, runs_dir=self.tmp_dir,
            )

    def test_same_run_id_different_item_selection_raises(self):
        write_run_manifest(
            run_id="run-1", dataset_version="dataset_conference_v1.1",
            protocol_version="protocol_v2", conditions=["A_EE"],
            models=["claude_sonnet_4_6"], total_planned_units=1,
            planned_item_ids_by_condition={"A_EE": ["gsm8k_0"]}, runs_dir=self.tmp_dir,
        )
        with self.assertRaises(ValueError):
            write_run_manifest(
                run_id="run-1", dataset_version="dataset_conference_v1.1",
                protocol_version="protocol_v2", conditions=["A_EE"],
                models=["claude_sonnet_4_6"], total_planned_units=1,
                planned_item_ids_by_condition={"A_EE": ["gsm8k_1"]},  # different item set!
                runs_dir=self.tmp_dir,
            )

    def test_same_run_id_different_git_commit_raises(self):
        from .. import storage
        with mock.patch.object(storage, "_git_commit", return_value="commit-a"):
            write_run_manifest(
                run_id="run-1", dataset_version="dataset_conference_v1.1",
                protocol_version="protocol_v2", conditions=["A_EE"],
                models=["claude_sonnet_4_6"], total_planned_units=1, runs_dir=self.tmp_dir,
            )
        with mock.patch.object(storage, "_git_commit", return_value="commit-b"):
            with self.assertRaises(ValueError):
                write_run_manifest(
                    run_id="run-1", dataset_version="dataset_conference_v1.1",
                    protocol_version="protocol_v2", conditions=["A_EE"],
                    models=["claude_sonnet_4_6"], total_planned_units=1, runs_dir=self.tmp_dir,
                )

    def test_created_at_preserved_and_last_resumed_at_set_on_restart(self):
        from .. import storage
        with mock.patch.object(storage, "_git_commit", return_value="fixed-commit"):
            kwargs = dict(
                run_id="run-1", dataset_version="dataset_conference_v1.1",
                protocol_version="protocol_v2", conditions=["A_EE"],
                models=["claude_sonnet_4_6"], total_planned_units=1, runs_dir=self.tmp_dir,
            )
            path = write_run_manifest(**kwargs)
            with open(path) as f:
                first = json.load(f)
            self.assertIsNone(first["last_resumed_at"])

            path = write_run_manifest(**kwargs)
            with open(path) as f:
                second = json.load(f)
            self.assertEqual(second["created_at"], first["created_at"])
            self.assertIsNotNone(second["last_resumed_at"])


if __name__ == "__main__":
    unittest.main()
