import argparse

from dotenv import load_dotenv

from .conditions import IMPLEMENTED_CONDITIONS
from .models import MODEL_REGISTRY
from .run import run_evaluation

# HPC-backed models (provider="hpc") require HPC_VLLM_BASE_URL to be set (raises a clear
# RuntimeError otherwise — see models.get_client) — excluded from the default roster so a
# plain `python -m evaluate` doesn't fail on an unconfigured endpoint by default. They're
# still selectable explicitly via --models once the HPC vLLM server is up.
DEFAULT_MODELS = [key for key, cfg in MODEL_REGISTRY.items() if cfg.provider != "hpc"]


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Staged cross-lingual LLM evaluation.")
    parser.add_argument("--limit", type=int, help="Limit the number of dataset rows evaluated.")
    parser.add_argument(
        "--run-id", default=None,
        help="Stable, predetermined run ID (e.g. one assigned per SLURM job). Restarting a "
             "failed job under the SAME --run-id appends to the same result JSONL instead "
             "of starting a new one; write_run_manifest() refuses to proceed if the new "
             "invocation's settings don't match what's already recorded for that run_id. "
             "Default: a fresh random UUID.",
    )
    parser.add_argument(
        "--item-ids", nargs="+", default=None,
        help="Restrict to exactly these dataset item IDs (e.g. a hand-picked source-diverse "
             "pilot slice), overriding both the full-dataset default and the A_E0/A_I0 "
             "stratified-subset restriction. Default: no restriction.",
    )
    parser.add_argument(
        "--conditions", nargs="+", default=IMPLEMENTED_CONDITIONS,
        choices=IMPLEMENTED_CONDITIONS,
        help="Which conditions to run (default: all implemented conditions).",
    )
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS,
        choices=list(MODEL_REGISTRY),
        help="Which models to run (default: all non-HPC registered models).",
    )
    args = parser.parse_args()

    run_id = run_evaluation(
        condition_keys=args.conditions, model_keys=args.models, limit=args.limit,
        item_ids=args.item_ids, run_id=args.run_id,
    )
    print(f"Run complete. run_id={run_id}")
    print(f"Results: text_track/data/eval_runs/{run_id}.jsonl")
    print(f"Manifest: text_track/data/eval_runs/{run_id}.manifest.json")


if __name__ == "__main__":
    main()
