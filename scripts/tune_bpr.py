import sys
from pathlib import Path
import pickle
import numpy as np
import optuna
import mlflow
import implicit
sys.path.append(str(Path(__file__).resolve().parent.parent))


def load_data():
    with open("data/processed/matrix.pkl", "rb") as f:
        matrix = pickle.load(f)
    with open("data/processed/mappings.pkl", "rb") as f:
        mappings = pickle.load(f)
    return matrix, mappings


def get_test_data():
    import pandas as pd
    orders = pd.read_csv("data/raw/orders.csv", usecols=["order_id", "user_id", "eval_set"])
    order_products_train = pd.read_csv("data/raw/order_products__train.csv", usecols=["order_id", "product_id"])
    train_orders = orders[orders["eval_set"] == "train"][["order_id", "user_id"]]
    return train_orders.merge(order_products_train, on="order_id")


def hit_rate(model, matrix, mappings, test_data, n_users=300, k=10):
    users = [u for u in test_data["user_id"].unique() if u in mappings["user2idx"]][:n_users]
    hits = 0
    for user_id in users:
        user_idx = mappings["user2idx"][user_id]
        item_ids, _ = model.recommend(user_idx, matrix[user_idx], N=k, filter_already_liked_items=True)
        recs = set(item_ids)
        true_items = set(
            mappings["item2idx"][p]
            for p in test_data[test_data["user_id"] == user_id]["product_id"].tolist()
            if p in mappings["item2idx"]
        )
        if len(recs & true_items) > 0:
            hits += 1
    return hits / len(users) if users else 0.0


def objective(trial, matrix, mappings, test_data):
    factors        = trial.suggest_int("factors", 50, 100)
    iterations     = trial.suggest_int("iterations", 50, 200)
    learning_rate  = trial.suggest_float("learning_rate", 0.001, 0.1, log=True)
    regularization = trial.suggest_float("regularization", 0.001, 0.1, log=True)

    model = implicit.bpr.BayesianPersonalizedRanking(
        factors=factors,
        iterations=iterations,
        learning_rate=learning_rate,
        regularization=regularization,
        use_gpu=False,
    )
    model.fit(matrix)
    hr = hit_rate(model, matrix, mappings, test_data)

    with mlflow.start_run(run_name=f"BPR-trial-{trial.number}", nested=True):
        mlflow.log_params({"factors": factors, "iterations": iterations,
                           "learning_rate": learning_rate, "regularization": regularization})
        mlflow.log_metric("hit_rate", hr)

    print(f"  Trial {trial.number} → HR={hr:.4f} | factors={factors}, iter={iterations}, lr={learning_rate:.4f}, reg={regularization:.4f}")
    return hr


if __name__ == "__main__":
    matrix, mappings = load_data()
    test_data = get_test_data()
    print(f"  {test_data['user_id'].nunique():,} users in test set")

    mlflow.set_experiment("instacart-recsys")
    with mlflow.start_run(run_name="Optuna-BPR-tuning"):
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(lambda trial: objective(trial, matrix, mappings, test_data), n_trials=10)

    print(f"\nBest trial → HR={study.best_value*100:.1f}%")
    print(f"  Hit Rate @10: {study.best_value:.4f} ({study.best_value*100:.1f}%)")
    print(f"  Parameters  : {study.best_params}")

    import json
    with open("configs/best_params_bpr.json", "w") as f:
        json.dump(study.best_params, f, indent=2)