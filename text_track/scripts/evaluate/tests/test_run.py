"""Small-slice dry run: exercises run_evaluation() end-to-end across every implemented
condition with FakeModelClient substituted for the model registry, so zero real API calls
are made. Confirms every produced ResultRecord matches the schema with no unexpected Nones.
"""
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from .. import models
from ..conditions import IMPLEMENTED_CONDITIONS
from ..run import run_evaluation
from .fakes import FakeModelClient, canned_response


def translate_or_reason_responder(system, user):
    if "translator" in system.lower():
        return canned_response(f"[translated] {user}")
    return canned_response("Reasoning steps...\n<answer>42</answer>")


class RunEvaluationDryRunTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)

        self.fake_client = FakeModelClient(translate_or_reason_responder)
        patcher = mock.patch.dict(
            models._CLIENT_CACHE, {"claude_sonnet_4_6": self.fake_client}, clear=True,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_produces_well_formed_records_for_every_implemented_condition(self):
        run_id = run_evaluation(
            condition_keys=IMPLEMENTED_CONDITIONS,
            model_keys=["claude_sonnet_4_6"],
            limit=3,
            runs_dir=self.tmp_dir,
        )

        results_path = os.path.join(self.tmp_dir, f"{run_id}.jsonl")
        manifest_path = os.path.join(self.tmp_dir, f"{run_id}.manifest.json")
        self.assertTrue(os.path.exists(results_path))
        self.assertTrue(os.path.exists(manifest_path))

        with open(manifest_path) as f:
            manifest = json.load(f)
        self.assertEqual(set(manifest["conditions"]), set(IMPLEMENTED_CONDITIONS))
        self.assertGreater(manifest["total_planned_units"], 0)

        with open(results_path) as f:
            records = [json.loads(line) for line in f if line.strip()]
        self.assertGreater(len(records), 0)

        seen_conditions = set()
        for record in records:
            seen_conditions.add(record["condition_key"])
            self.assertIn(record["stage"], {"direct", "translate", "reason"})
            self.assertIsInstance(record["item_id"], str)
            self.assertIsInstance(record["source"], str)
            self.assertEqual(record["model_key"], "claude_sonnet_4_6")
            self.assertIn(record["failure_type"], {
                "correct", "substantively_incorrect", "invalid_answer_format",
                "missing_answer", "translation_completed", "translation_format_failure",
                "refusal", "truncation", "repetition_degeneration", "parser_failure",
                "infrastructure_api_failure",
            })
            if record["stage"] == "translate":
                self.assertIsNotNone(record["generated_translation"])
                self.assertIsNone(record["is_correct"])
            if record["stage"] in ("direct", "reason"):
                self.assertIsNotNone(record["canonical_answer"])
            if record["stage"] == "direct":
                # A_E0/A_I0 have no rationale by design — rationale_tokens must be None
                # even though the fake responder's text ("Reasoning steps...") would yield
                # a nonzero count if it were (wrongly) tokenized as a rationale.
                self.assertIsNone(record["rationale_tokens"])

        self.assertEqual(seen_conditions, set(IMPLEMENTED_CONDITIONS))

        # Staged pivots (A_IE/A_EI) must produce both a translate and a reason row per item;
        # the reasoning-stage record's original_input must be the saved translation, never
        # the original-language question or the canonical answer.
        for record in records:
            if record["condition_key"] in ("A_IE", "A_EI") and record["stage"] == "reason":
                self.assertTrue(record["original_input"].startswith("[translated]"))


if __name__ == "__main__":
    unittest.main()
