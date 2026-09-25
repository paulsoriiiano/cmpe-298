import argparse

from dotenv import load_dotenv

from .conditions import IMPLEMENTED_CONDITIONS
from .models import MODEL_REGISTRY
from .run import run_evaluation

# HPC-backed models (provider="hpc") have no working client yet (see models.HPCClient) —
# excluded from the default roster so a plain `python -m evaluate` doesn't immediately
# raise NotImplementedError. They're still selectable explicitly via --models.
DEFAULT_MODELS = [key for key, cfg in MODEL_REGISTRY.items() if cfg.provider != "hpc"]


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Staged cross-lingual LLM evaluation.")
    parser.add_argument("--limit", type=int, help="Limit the number of dataset rows evaluated.")
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
    )
    print(f"Run complete. run_id={run_id}")
    print(f"Results: text_track/data/eval_runs/{run_id}.jsonl")
    print(f"Manifest: text_track/data/eval_runs/{run_id}.manifest.json")


if __name__ == "__main__":
    main()
