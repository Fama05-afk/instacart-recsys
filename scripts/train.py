import sys, json, argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.matrix_builder import MatrixBuilder
from src.models.als_model import ALSRecommender
from src.models.bpr_model import BPRRecommender
from src.models.ease_model import EASERecommender

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


def train_model(name: str):
    print(f"\n{'='*50}")
    print(f"  Training: {name.upper()}")
    print(f"{'='*50}")

    params_path = Path(PARAMS_FILES[name])
    if not params_path.exists():
        raise FileNotFoundError(
            f"Params not found for {name.upper()}: {params_path}\n"
            f"Run first: python src/tuning/tune_{name}.py"
        )

    with open(params_path) as f:
        params = json.load(f)
    print(f"Params: {params}")

    model = MODEL_CLASSES[name](**params)
    model.run()

    if hasattr(model, "run_id") and model.run_id:
        print(f"[OK] MLflow run registered — ID: {model.run_id}")
    else:
        print(f"[OK] Model trained (MLflow disabled)")

    print(f"\nTest recommendations — user_id 1:")
    recs = model.recommend(user_id=1, n=10)
    for r in recs:
        print(f"  {r['product']:<45} score: {r['score']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train instacart-recsys models")
    parser.add_argument(
        "--model",
        choices=["als", "bpr", "ease", "all"],
        default="all",
        help="Model to train (default: all)"
    )
    parser.add_argument(
        "--rebuild-matrix",
        action="store_true",
        help="Force sparse matrix reconstruction"
    )
    args = parser.parse_args()

    if args.rebuild_matrix or not Path("data/processed/matrix.pkl").exists():
        print("Building sparse matrix...")
        MatrixBuilder().run()

    models = ["als", "bpr", "ease"] if args.model == "all" else [args.model]
    for name in models:
        train_model(name)