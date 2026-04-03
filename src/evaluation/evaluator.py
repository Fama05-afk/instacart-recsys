import pandas as pd
import numpy as np
import mlflow
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))


class Evaluator:
    """
    Evaluate ALS, BPR or EASE on the Instacart test set.
    prior = training, train = last order of each user = ground truth.
    """

    def __init__(self, k: int = 10):
        self.k               = k
        self.model           = None  # raw implicit model or B matrix for EASE
        self.matrix          = None
        self.mappings        = None
        self.test_data       = None
        self.model_type      = None  # "implicit" or "ease"
        self.user_true_items = {}

    def load(self, model, matrix, mappings, data_dir: str = "data", model_type: str = "implicit"):
        self.model      = model
        self.matrix     = matrix
        self.mappings   = mappings
        self.model_type = model_type

        orders = pd.read_csv(
            f"{data_dir}/raw/orders.csv",
            usecols=["order_id", "user_id", "eval_set"]
        )
        order_products_train = pd.read_csv(
            f"{data_dir}/raw/order_products__train.csv",
            usecols=["order_id", "product_id"]
        )
        train_orders = orders[orders["eval_set"] == "train"][["order_id", "user_id"]]
        self.test_data = train_orders.merge(order_products_train, on="order_id")

        # precompute ground truth per user
        self.user_true_items = {
            user_id: set(
                mappings["item2idx"][p]
                for p in group["product_id"]
                if p in mappings["item2idx"]
            )
            for user_id, group in self.test_data.groupby("user_id")
        }

        print(f"  {self.test_data['user_id'].nunique():,} users in test set")
        return self

    def _get_recommendations(self, user_id: int) -> set:
        if user_id not in self.mappings["user2idx"]:
            return set()
        user_idx = self.mappings["user2idx"][user_id]

        if self.model_type == "ease":
            # self.model = B, self.matrix contient top_idx dans model.top_idx
            user_vec = self.matrix[user_idx, :][:, self.top_idx].toarray().flatten().astype(np.float32)
            scores = user_vec @ self.model
            scores[user_vec > 0] = -np.inf
            top_items = np.argsort(scores)[::-1][:self.k]
            return {int(self.top_idx[i]) for i in top_items}
        else:
            item_ids, _ = self.model.recommend(
                user_idx, self.matrix[user_idx], N=self.k, filter_already_liked_items=True
            )
            return set(item_ids)

    def _get_true_items(self, user_id: int) -> set:
        return self.user_true_items.get(user_id, set())

    def hit_rate(self, n_users: int = 1000) -> float:
        users = self.test_data["user_id"].unique()[:n_users]
        hits  = sum(
            1 for user_id in users
            if self._get_recommendations(user_id) & self._get_true_items(user_id)
        )
        return hits / len(users) if len(users) > 0 else 0.0

    def ndcg(self, n_users: int = 1000) -> float:
        users  = self.test_data["user_id"].unique()[:n_users]
        scores = []

        for user_id in users:
            if user_id not in self.mappings["user2idx"]:
                continue
            user_idx = self.mappings["user2idx"][user_id]

            if self.model_type == "ease":
                user_vec = self.matrix[user_idx, :][:, self.top_idx].toarray().flatten().astype(np.float32)
                s = user_vec @ self.model
                s[user_vec > 0] = -np.inf
                item_ids = [int(self.top_idx[i]) for i in np.argsort(s)[::-1][:self.k]]
            else:
                item_ids, _ = self.model.recommend(
                    user_idx, self.matrix[user_idx], N=self.k, filter_already_liked_items=True
                )

            true_items = self._get_true_items(user_id)
            dcg = sum(
                1 / np.log2(rank + 1)
                for rank, item_idx in enumerate(item_ids, start=1)
                if item_idx in true_items
            )
            n_relevant = min(len(true_items), self.k)
            idcg = sum(1 / np.log2(r + 1) for r in range(1, n_relevant + 1))
            scores.append(dcg / idcg if idcg > 0 else 0.0)

        return float(np.mean(scores)) if scores else 0.0

    def evaluate(self, n_users: int = 1000, model_name: str = "", log_mlflow: bool = True) -> dict:
        print(f"Evaluating {model_name} on {n_users} users (K={self.k})...")

        hr         = self.hit_rate(n_users)
        ndcg_score = self.ndcg(n_users)

        print(f"  Hit Rate @{self.k}: {hr:.4f} ({hr*100:.1f}%)")
        print(f"  NDCG @{self.k}    : {ndcg_score:.4f}")

        if log_mlflow:
            mlflow.set_experiment("instacart-recsys")
            with mlflow.start_run(run_name=f"{model_name}-evaluation"):
                mlflow.log_metrics({"hit_rate": hr, "ndcg": ndcg_score})
                mlflow.log_params({"K": self.k, "n_users_eval": n_users, "model": model_name})
            print("  Logged to MLflow.")

        return {
            "model":        model_name,
            "hit_rate@K":   round(hr, 4),
            "ndcg@K":       round(ndcg_score, 4),
            "K":            self.k,
            "n_users_eval": n_users,
        }