"""Unit tests for storage.py's resume-key construction and manifest writer.

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

from ..storage import ResultRecord, make_resume_key, write_run_manifest


def _make_record(**overrides):
    defaults = dict(
        run_id="run-1", dataset_version="dataset_conference_v1.1",
        protocol_version="protocol_v2", item_id="gsm8k_0", source="gsm8k",
        model_key="claude_sonnet_4_6", condition_key="A_EE", stage="reason",
        prompt_version="v2", original_input="q", generated_translation=None,
        generated_rationale="r", raw_response="r", extracted_answer="109",
        canonical_answer="109", is_correct=True, format_compliant=True,
        failure_type="correct", input_tokens=10, output_tokens=5,
        system_prompt_tokens=1, user_input_tokens=1, rendered_prompt_tokens=2,
        tokenizer_id="tiktoken/cl100k_base", tokenizer_revision="0.14.0",
        finish_reason="stop", latency_ms=1.0, retry_count=0, error_message=None,
        timestamp="2026-01-01T00:00:00+00:00",
    )
    defaults.update(overrides)
    return ResultRecord(**defaults)


class ResumeKeyTests(unittest.TestCase):
    def test_resume_key_includes_prompt_version(self):
        record = _make_record(prompt_version="v2")
        key_v2 = record.resume_key()
        record_v1 = _make_record(prompt_version="v1")
        key_v1 = record_v1.resume_key()
        self.assertNotEqual(key_v2, key_v1)

    def test_make_resume_key_matches_instance_method(self):
        record = _make_record()
        via_instance = record.resume_key()
        via_function = make_resume_key(
            dataset_version=record.dataset_version, protocol_version=record.protocol_version,
            prompt_version=record.prompt_version, item_id=record.item_id,
            model_key=record.model_key, condition_key=record.condition_key, stage=record.stage,
        )
        self.assertEqual(via_instance, via_function)


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


if __name__ == "__main__":
    unittest.main()
