import sys, json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import optuna
import mlflow
from src.models.ease_model import EASERecommender


def get_test_data():
    orders = pd.read_csv("data/raw/orders.csv", usecols=["order_id", "user_id", "eval_set"])
    order_products_train = pd.read_csv("data/raw/order_products__train.csv", usecols=["order_id", "product_id"])
    train_orders = orders[orders["eval_set"] == "train"][["order_id", "user_id"]]
    return train_orders.merge(order_products_train, on="order_id")


def hit_rate(model, test_data, n_users=300, k=10):
    mappings = model.mappings
    users = [u for u in test_data["user_id"].unique() if u in mappings["user2idx"]][:n_users]
    hits = 0
    for user_id in users:
        recs = {r["product_idx"] for r in model.recommend(user_id, n=k)}
        true_items = set(
            mappings["item2idx"][p]
            for p in test_data[test_data["user_id"] == user_id]["product_id"]
            if p in mappings["item2idx"]
        )
        if recs & true_items:
            hits += 1
    return hits / len(users) if users else 0.0


def objective(trial, test_data):
    lambda_ = trial.suggest_float("lambda_", 50, 2000, log=True)

    model = EASERecommender(lambda_=lambda_)
    model.load_data()
    model.train(use_mlflow=False)  # MLflow handled here via nested

    hr = hit_rate(model, test_data, n_users=300, k=10)

    with mlflow.start_run(run_name=f"EASE-trial-{trial.number}", nested=True):
        mlflow.log_param("lambda_", lambda_)
        mlflow.log_metric("hit_rate", hr)

    print(f"  Trial {trial.number} → HR={hr:.4f} | lambda={lambda_:.1f}")
    return hr


if __name__ == "__main__":
    test_data = get_test_data()
    print(f"  {test_data['user_id'].nunique():,} users in test set")

    mlflow.set_experiment("instacart-recsys")
    with mlflow.start_run(run_name="Optuna-EASE-tuning"):
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(lambda trial: objective(trial, test_data), n_trials=10)

    print(f"\nBest trial → Hit Rate @10: {study.best_value:.4f} ({study.best_value*100:.1f}%)")
    print(f"  Parameters: {study.best_params}")

    with open("configs/best_params_ease.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    print("Saved → configs/best_params_ease.json")