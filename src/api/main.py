"""
FastAPI API: product recommendations via ALS, EASE or BPR.
"""
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from src.models.ease_model import EASERecommender
import json
import os, sys, pickle
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# load ALS and BPR models
with open("models/als_model.pkl", "rb") as f:
    als_model = pickle.load(f)
with open("models/bpr_model.pkl", "rb") as f:
    bpr_model = pickle.load(f)
with open("data/processed/mappings.pkl", "rb") as f:
    mappings = pickle.load(f)
with open("data/processed/matrix.pkl", "rb") as f:
    matrix = pickle.load(f)

# load EASE from pkl, no retraining at startup
with open("configs/best_params_ease.json") as f:
    ease_params = json.load(f)
ease = EASERecommender(lambda_=ease_params["lambda_"])
ease.load_data()
with open("models/ease_model.pkl", "rb") as f:
    data = pickle.load(f)
    ease.B       = data["B"]
    ease.top_idx = data["top_idx"]


app = FastAPI(
    title="Instacart Recommender API",
    description="Product recommendations — ALS, EASE, BPR",
    version="2.0.0",
)


class RecommendedProduct(BaseModel):
    product: str
    score: float

class RecommendationResponse(BaseModel):
    user_id: int
    model:   str
    recommendations: List[RecommendedProduct]


@app.get("/health")
def health():
    return {"status": "ok", "models": ["als", "bpr", "ease"]}


@app.get("/models")
def list_models():
    """Metrics for models evaluated on 10,000 users."""
    return {
        "als":  {"hit_rate@10": 0.1616, "ndcg@10": 0.0247},
        "bpr":  {"hit_rate@10": 0.1457, "ndcg@10": 0.0224},
        "ease": {"hit_rate@10": 0.1456, "ndcg@10": 0.0229},
    }


@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
def recommend(user_id: int, n: int = 10, model_name: str = "als"):
    model_name = model_name.lower()

    if model_name == "ease":
        if user_id not in ease.mappings["user2idx"]:
            raise HTTPException(status_code=404, detail=f"user_id {user_id} not found.")
        recs = ease.recommend(user_id, n=n)
        recommendations = [
            RecommendedProduct(product=r["product"], score=round(r["score"], 4))
            for r in recs
        ]

    elif model_name in ("als", "bpr"):
        if user_id not in mappings["user2idx"]:
            raise HTTPException(status_code=404, detail=f"user_id {user_id} not found.")

        user_idx = mappings["user2idx"][user_id]
        m        = als_model if model_name == "als" else bpr_model
        item_ids, scores = m.recommend(
            user_idx,
            matrix[user_idx],
            N=n,
            filter_already_liked_items=True,
        )
        recommendations = [
            RecommendedProduct(
                product=mappings["idx2name"].get(int(i), f"product_{i}"),
                score=round(float(s), 4),
            )
            for i, s in zip(item_ids, scores)
        ]

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {model_name}. Choose from als, bpr, ease."
        )

    return RecommendationResponse(user_id=user_id, model=model_name, recommendations=recommendations)