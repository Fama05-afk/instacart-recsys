"""
ALS tuning with Optuna, search for best hyperparameters
over 15 trials and save the result in configs/best_params.json.
"""
import pickle
import numpy as np
import optuna
import mlflow
import implicit
import json
from scipy.sparse import csr_matrix
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))


def load_data():
    with open("data/processed/matrix.pkl", "rb") as f:
        matrix = pickle.load(f)
    with open("data/processed/mappings.pkl", "rb") as f:
        mappings = pickle.load(f)
    return matrix, mappings


def get_test_data(mappings):
    import pandas as pd

    orders = pd.read_csv(
        "data/raw/orders.csv",
        usecols=["order_id", "user_id", "eval_set"]
    )
    order_products_train = pd.read_csv(
        "data/raw/order_products__train.csv",
        usecols=["order_id", "product_id"]
    )

    # train orders = ground truth for evaluating recommendations
    train_orders = orders[orders["eval_set"] == "train"][["order_id", "user_id"]]
    test_data = train_orders.merge(order_products_train, on="order_id")

    return test_data


def hit_rate(model, matrix, mappings, test_data, n_users=500, k=10):
    users = [
        u for u in test_data["user_id"].unique()
        if u in mappings["user2idx"]
    ][:n_users]
    
    hits = 0
    for user_id in users:
        user_idx = mappings["user2idx"][user_id]
        item_ids, _ = model.recommend(
            user_idx,
            matrix[user_idx],
            N=k,
            filter_already_liked_items=True,
        )
        recs = set(item_ids)
        true_products = test_data[
            test_data["user_id"] == user_id
        ]["product_id"].tolist()
        true_items = set(
            mappings["item2idx"][p]
            for p in true_products
            if p in mappings["item2idx"]
        )
        if len(recs & true_items) > 0:
            hits += 1

    return hits / len(users) if users else 0.0


def objective(trial, matrix, mappings, test_data):
    factors        = trial.suggest_int("factors", 50, 100)
    iterations     = trial.suggest_int("iterations", 10, 50)
    regularization = trial.suggest_float("regularization", 0.001, 1.0, log=True)
    alpha          = trial.suggest_float("alpha", 10, 80)

    model = implicit.als.AlternatingLeastSquares(
        factors=factors,
        iterations=iterations,
        regularization=regularization,
        use_gpu=False,
    )
    model.fit((matrix * alpha).astype(np.float32))

    hr = hit_rate(model, matrix, mappings, test_data, n_users=500, k=10)

    with mlflow.start_run(run_name=f"ALS-trial-{trial.number}", nested=True): # Log to MLflow
        mlflow.log_params({
            "factors":        factors,
            "iterations":     iterations,
            "regularization": regularization,
            "alpha":          alpha,
        })
        mlflow.log_metric("hit_rate", hr)

    print(f"  Trial {trial.number} → HR={hr:.4f} | factors={factors}, iter={iterations}, reg={regularization:.4f}, alpha={alpha:.1f}")

    return hr


if __name__ == "__main__":
    print("Loading data...")
    matrix, mappings = load_data()
    test_data = get_test_data(mappings)

    mlflow.set_experiment("instacart-recsys")

    with mlflow.start_run(run_name="Optuna-ALS-tuning"):
        # TPE = Bayesian, each trial builds on previous ones
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        study.optimize(
            lambda trial: objective(trial, matrix, mappings, test_data),
            n_trials=15,
        )

    print(f"\nBest trial → Hit Rate @10: {study.best_value:.4f} ({study.best_value*100:.1f}%)")
    print(f"  Parameters: {study.best_params}")

    with open("configs/best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"  Saved → configs/best_params.json")