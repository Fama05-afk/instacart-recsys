"""
Model evaluation.
"""

import sys, json, argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.als_model  import ALSRecommender
from src.models.bpr_model  import BPRRecommender
from src.models.ease_model import EASERecommender
from src.evaluation.evaluator import Evaluator

PARAMS_FILES = {
    "als":  "configs/best_params_als.json",
    "bpr":  "configs/best_params_bpr.json",
    "ease": "configs/best_params_ease.json",
}

MODEL_CLASSES = {
    "als":  ALSRecommender,
    "bpr":  BPRRecommender,
    "ease": EASERecommender,
}


def evaluate_model(name: str, n_users: int):
    print(f"\n{'='*50}")
    print(f"  Evaluation: {name.upper()}")
    print(f"{'='*50}")

    params_path = Path(PARAMS_FILES[name])
    if not params_path.exists():
        raise FileNotFoundError(
            f"Params not found for {name.upper()}: {params_path}\n"
            f"Run first: python src/tuning/tune_{name}.py"
        )

    with open(params_path) as f:
        params = json.load(f)

    wrapper = MODEL_CLASSES[name](**params)
    wrapper.load_data()
    wrapper.train(use_mlflow=False)

    evaluator = Evaluator(k=10)

    if name == "ease":
        evaluator.top_idx = wrapper.top_idx
        evaluator.load(
            model=wrapper.B,
            matrix=wrapper.matrix,
            mappings=wrapper.mappings,
            model_type="ease"
        )
    else:
        evaluator.load(
            model=wrapper.model,
            matrix=wrapper.matrix,
            mappings=wrapper.mappings,
            model_type="implicit"
        )

    return evaluator.evaluate(n_users=n_users, model_name=name.upper())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate instacart-recsys models")
    parser.add_argument(
        "--model",
        choices=["als", "bpr", "ease", "all"],
        default="all",
        help="Model to evaluate (default: all)"
    )
    parser.add_argument(
        "--n-users",
        type=int,
        default=1000,
        help="Number of users to evaluate (default: 1000)"
    )
    args = parser.parse_args()

    models = ["als", "bpr", "ease"] if args.model == "all" else [args.model]
    results = []
    for name in models:
        r = evaluate_model(name, args.n_users)
        results.append(r)

    if len(results) > 1:
        print(f"\n{'='*50}")
        print("  Model Comparison")
        print(f"{'='*50}")
        print(f"  {'Model':<8} {'Hit Rate @10':>12} {'NDCG @10':>10}")
        print(f"  {'-'*32}")
        for r in results:
            print(f"  {r['model']:<8} {r['hit_rate@K']:>12.4f} {r['ndcg@K']:>10.4f}")