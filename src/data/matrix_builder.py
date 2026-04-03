# src/data/matrix_builder.py

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from pathlib import Path
import pickle


class MatrixBuilder:
    """
    Builds the sparse users × products matrix
    from the merged Instacart prior dataset.
    """

    def __init__(self, data_path: str = "data/processed/prior_merged.csv"):
        self.data_path = Path(data_path)
        self.df = None

        # Mappings index <-> id (necessary to interpret ALS results)
        self.user2idx = {}
        self.idx2user = {}
        self.item2idx = {}
        self.idx2item = {}
        self.idx2name = {}  # product index -> readable name

        self.matrix = None  # the final sparse matrix

    # def load(self):
    #     print("Loading data...")
    #     self.df = pd.read_csv(self.data_path)
    #     print(f"  {len(self.df):,} rows loaded")
    #     return self

    def load(self):
        print("Loading data...")
        self.df = pd.read_csv(
            self.data_path,
            usecols=["user_id", "product_id", "product_name", "order_number"]
        )
        print(f"  {len(self.df):,} rows loaded")
        return self

    def build_mappings(self):
        """
        Creates continuous integer indices for users and products.
        ALS needs indices 0, 1, 2... not arbitrary IDs.
        """
        users    = self.df["user_id"].unique()
        products = self.df["product_id"].unique()

        # dict comprehension: {user_id: integer_index}
        self.user2idx = {u: i for i, u in enumerate(users)}
        self.idx2user = {i: u for u, i in self.user2idx.items()}

        self.item2idx = {p: i for i, p in enumerate(products)}
        self.idx2item = {i: p for p, i in self.item2idx.items()}

        # product index -> name (to display readable recommendations)
        product_names = self.df[["product_id", "product_name"]].drop_duplicates()
        self.idx2name = {
            self.item2idx[row.product_id]: row.product_name
            for row in product_names.itertuples()
        }

        print(f"  {len(users):,} users | {len(products):,} products")
        return self

    def build_matrix(self):
        """
        Builds the sparse matrix (users × products).
        Each cell value = number of times the user purchased
        this product (purchase frequency).
        """
        print("Building sparse matrix...")

        # Count number of purchases per (user, product)
        counts = (
            self.df.groupby(["user_id", "product_id"])
            .size()
            .reset_index(name="frequency")
        )

        # Convert IDs to integer indices
        row = counts["user_id"].map(self.user2idx).values
        col = counts["product_id"].map(self.item2idx).values
        data = counts["frequency"].values

        n_users = len(self.user2idx)
        n_items = len(self.item2idx)

        # csr_matrix((data, (row, col)), shape)
        # CSR = Compressed Sparse Row — optimal format for ALS
        self.matrix = csr_matrix(
            (data, (row, col)),
            shape=(n_users, n_items),
            dtype=np.float32
        )

        sparsity = 1 - self.matrix.nnz / (n_users * n_items)
        print(f"  Shape: {self.matrix.shape}")
        print(f"  Non-zero values: {self.matrix.nnz:,}")
        print(f"  Sparsity: {sparsity:.4%}")
        return self

    # def build_matrix(self):
    #     print("Building sparse matrix with temporal weighting...")

    #     # Convert directly to numpy arrays — no pandas
    #     user_idx = self.df["user_id"].map(self.user2idx).values.astype(np.int32)
    #     item_idx = self.df["product_id"].map(self.item2idx).values.astype(np.int32)
    #     order_num = self.df["order_number"].values.astype(np.int32)

    #     # Max order per user with numpy
    #     n_users = len(self.user2idx)
    #     max_order = np.zeros(n_users, dtype=np.int32)
    #     np.maximum.at(max_order, user_idx, order_num)

    #     # Recency and weighting
    #     recency = max_order[user_idx] - order_num
    #     time_weight = np.where(recency <= 3, 1.0,
    #                 np.where(recency <= 8, 0.5, 0.1)).astype(np.float32)

    #     # Direct sparse construction — scipy automatically sums duplicates
    #     from scipy.sparse import coo_matrix
    #     self.matrix = coo_matrix(
    #         (time_weight, (user_idx, item_idx)),
    #         shape=(n_users, len(self.item2idx)),
    #         dtype=np.float32
    #     ).tocsr()

    #     sparsity = 1 - self.matrix.nnz / (n_users * len(self.item2idx))
    #     print(f"  Shape: {self.matrix.shape}")
    #     print(f"  Non-zero values: {self.matrix.nnz:,}")
    #     print(f"  Sparsity: {sparsity:.4%}")
        return self

    def save(self, out_dir: str = "data/processed"):
        """Save the matrix and mappings to pickle files."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        with open(out / "matrix.pkl", "wb") as f:
            pickle.dump(self.matrix, f)

        mappings = {
            "user2idx": self.user2idx,
            "idx2user": self.idx2user,
            "item2idx": self.item2idx,
            "idx2item": self.idx2item,
            "idx2name": self.idx2name,
        }
        with open(out / "mappings.pkl", "wb") as f:
            pickle.dump(mappings, f)

        print(f"  Matrix saved -> {out}/matrix.pkl")
        print(f"  Mappings saved -> {out}/mappings.pkl")
        return self

    def run(self):
        """Complete pipeline in one line."""
        return self.load().build_mappings().build_matrix().save()


if __name__ == "__main__":
    MatrixBuilder().run()