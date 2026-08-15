from __future__ import annotations

import os
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "olist_matplotlib"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*")

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TRACK_B_POSITIVE_THRESHOLD = 0.53
TRACK_B_RISK_THRESHOLD = 1 - TRACK_B_POSITIVE_THRESHOLD
TRACK_B_CAUTION_RISK_THRESHOLD = 0.35
DEFAULT_TRACK_C_QUANTILE = 0.90
TRACK_C_QUANTILE_OPTIONS = (0.80, 0.85, 0.90, 0.95)
RANDOM_STATE = 42


PRE_ORDER_COLS = [
    "order_item_id",
    "price",
    "freight_value",
    "freight_ratio",
    "total_price",
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm",
    "order_purchase_dayofweek",
    "order_purchase_month",
    "expected_delivery_days",
    "main_category",
    "sub_category",
    "distance_km",
    "distance_cat",
    "cross_state",
    "is_sp_customer",
    "is_sp_seller",
    "sp_route_type",
    "sp_route_type_customer",
    "sp_route_type_seller",
]


@dataclass
class TrainedModels:
    track_b: Pipeline
    track_c: Pipeline
    feature_cols: list[str]
    track_b_positive_threshold: float = TRACK_B_POSITIVE_THRESHOLD
    track_c_quantile: float = DEFAULT_TRACK_C_QUANTILE


@dataclass
class ConsoleArtifacts:
    models: TrainedModels
    test_orders: pd.DataFrame
    high_risk_orders: pd.DataFrame


def load_ml_data(data_path: str | Path = "data/processed/ml_data.csv") -> pd.DataFrame:
    return pd.read_csv(data_path)


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def add_pre_order_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["freight_ratio"] = safe_divide(out["freight_value"], out["price"])
    out["total_price"] = out["price"] + out["freight_value"]

    out["is_sp_customer"] = (out["customer_state"] == "SP").astype(int)
    out["is_sp_seller"] = (out["seller_state"] == "SP").astype(int)

    out["sp_route_type"] = 0
    partial_route = (out["seller_state"] == "SP") | (out["customer_state"] == "SP")
    internal_route = (out["seller_state"] == "SP") & (out["customer_state"] == "SP")
    out.loc[partial_route, "sp_route_type"] = 1
    out.loc[internal_route, "sp_route_type"] = 2

    out["sp_route_type_customer"] = 0
    out.loc[out["customer_state"] == "SP", "sp_route_type_customer"] = 1
    out.loc[internal_route, "sp_route_type_customer"] = 2

    out["sp_route_type_seller"] = 0
    out.loc[out["seller_state"] == "SP", "sp_route_type_seller"] = 1
    out.loc[internal_route, "sp_route_type_seller"] = 2

    return out.replace([np.inf, -np.inf], np.nan)


def review_risk_target(review_score: pd.Series) -> pd.Series:
    return (review_score >= 4).astype(int)


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
                    ]
                ),
                cat_cols,
            ),
        ]
    )


def split_by_order(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    return (
        X.iloc[train_idx],
        X.iloc[test_idx],
        y.iloc[train_idx],
        y.iloc[test_idx],
        groups.iloc[train_idx],
        groups.iloc[test_idx],
    )


def build_track_b_pipeline(X: pd.DataFrame) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(X)),
            (
                "classifier",
                LGBMClassifier(
                    n_estimators=800,
                    learning_rate=0.011153121252070788,
                    num_leaves=29,
                    max_depth=5,
                    colsample_bytree=0.5028265220878869,
                    min_child_samples=47,
                    class_weight="balanced",
                    n_jobs=1,
                    random_state=RANDOM_STATE,
                    verbose=-1,
                ),
            ),
        ]
    )


def build_track_c_pipeline(X: pd.DataFrame, quantile: float = DEFAULT_TRACK_C_QUANTILE) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(X)),
            (
                "regressor",
                LGBMRegressor(
                    objective="quantile",
                    alpha=quantile,
                    n_estimators=700,
                    learning_rate=0.03,
                    num_leaves=31,
                    max_depth=6,
                    min_child_samples=40,
                    n_jobs=1,
                    random_state=RANDOM_STATE,
                    verbose=-1,
                ),
            ),
        ]
    )


def prepare_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    prepared = add_pre_order_features(df)
    required_cols = ["order_id", "review_score", "delivery_days", *PRE_ORDER_COLS]
    return prepared[required_cols].dropna(subset=["order_id", "review_score", "delivery_days"])


def train_models(
    df: pd.DataFrame,
    quantile: float = DEFAULT_TRACK_C_QUANTILE,
    feature_cols: Iterable[str] = PRE_ORDER_COLS,
) -> TrainedModels:
    model_df = prepare_model_frame(df)
    feature_cols = list(feature_cols)
    X = model_df[feature_cols]
    y_track_b = review_risk_target(model_df["review_score"])
    y_track_c = model_df["delivery_days"]

    track_b = build_track_b_pipeline(X)
    track_b.fit(X, y_track_b)

    track_c = build_track_c_pipeline(X, quantile=quantile)
    track_c.fit(X, y_track_c)

    return TrainedModels(
        track_b=track_b,
        track_c=track_c,
        feature_cols=feature_cols,
        track_c_quantile=quantile,
    )


def risk_level_from_probability(
    review_risk_probability: float,
    risk_threshold: float = TRACK_B_RISK_THRESHOLD,
    caution_threshold: float = TRACK_B_CAUTION_RISK_THRESHOLD,
) -> str:
    if review_risk_probability > risk_threshold:
        return "고위험"
    if review_risk_probability > caution_threshold:
        return "주의"
    return "일반"


def predict_order(
    models: TrainedModels,
    order_features: pd.DataFrame,
    risk_threshold: float = TRACK_B_RISK_THRESHOLD,
    caution_threshold: float = TRACK_B_CAUTION_RISK_THRESHOLD,
) -> dict[str, float | str]:
    X = order_features[models.feature_cols]
    positive_probability = float(models.track_b.predict_proba(X)[:, 1][0])
    review_risk_probability = 1.0 - positive_probability
    predicted_delivery_days = float(models.track_c.predict(X)[0])
    current_expected_days = float(X["expected_delivery_days"].iloc[0])
    is_track_c_target = review_risk_probability > risk_threshold
    recommended_expected_days = float(max(current_expected_days, np.ceil(predicted_delivery_days)))
    adjustment_days = recommended_expected_days - current_expected_days
    risk_level = risk_level_from_probability(
        review_risk_probability,
        risk_threshold=risk_threshold,
        caution_threshold=caution_threshold,
    )

    return {
        "positive_probability": positive_probability,
        "review_risk_probability": review_risk_probability,
        "predicted_delivery_days": predicted_delivery_days,
        "current_expected_days": current_expected_days,
        "recommended_expected_days": recommended_expected_days,
        "adjustment_days": adjustment_days,
        "risk_level": risk_level,
        "is_track_c_target": is_track_c_target,
    }


def train_console_artifacts(
    df: pd.DataFrame,
    quantile: float = DEFAULT_TRACK_C_QUANTILE,
    risk_threshold: float = TRACK_B_RISK_THRESHOLD,
    caution_threshold: float = TRACK_B_CAUTION_RISK_THRESHOLD,
) -> ConsoleArtifacts:
    model_df = prepare_model_frame(df)
    X = model_df[PRE_ORDER_COLS]
    y_track_b = review_risk_target(model_df["review_score"])
    y_track_c = model_df["delivery_days"]
    groups = model_df["order_id"]
    X_train, X_test, y_train_b, _, _, _ = split_by_order(X, y_track_b, groups)
    y_train_c = y_track_c.loc[X_train.index]

    track_b = build_track_b_pipeline(X_train)
    track_b.fit(X_train, y_train_b)

    track_c = build_track_c_pipeline(X_train, quantile=quantile)
    track_c.fit(X_train, y_train_c)

    models = TrainedModels(
        track_b=track_b,
        track_c=track_c,
        feature_cols=PRE_ORDER_COLS,
        track_c_quantile=quantile,
    )

    positive_proba = track_b.predict_proba(X_test)[:, 1]
    risk_proba = 1 - positive_proba
    pred_delivery = track_c.predict(X_test)
    recommended = np.maximum(X_test["expected_delivery_days"].to_numpy(), np.ceil(pred_delivery))
    adjustment = recommended - X_test["expected_delivery_days"].to_numpy()

    scored_test = df.loc[X_test.index].copy()
    scored_test["review_risk_probability"] = risk_proba
    scored_test["predicted_delivery_days_quantile"] = pred_delivery
    scored_test["recommended_expected_days"] = recommended
    scored_test["adjustment_days"] = adjustment
    scored_test["risk_level"] = [
        risk_level_from_probability(
            probability,
            risk_threshold=risk_threshold,
            caution_threshold=caution_threshold,
        )
        for probability in risk_proba
    ]
    high_risk_orders = (
        scored_test[scored_test["risk_level"] == "고위험"]
        .sort_values("review_risk_probability", ascending=False)
        .drop_duplicates("order_id")
        .reset_index(drop=True)
    )
    test_orders = scored_test.drop_duplicates("order_id").reset_index(drop=True)

    return ConsoleArtifacts(
        models=models,
        test_orders=test_orders,
        high_risk_orders=high_risk_orders,
    )


def evaluate_track_b(df: pd.DataFrame) -> dict[str, float]:
    model_df = prepare_model_frame(df)
    X = model_df[PRE_ORDER_COLS]
    y = review_risk_target(model_df["review_score"])
    groups = model_df["order_id"]
    X_train, X_test, y_train, y_test, _, _ = split_by_order(X, y, groups)

    model = build_track_b_pipeline(X_train)
    model.fit(X_train, y_train)
    positive_proba = model.predict_proba(X_test)[:, 1]
    pred = (positive_proba >= TRACK_B_POSITIVE_THRESHOLD).astype(int)

    return {
        "positive_threshold": TRACK_B_POSITIVE_THRESHOLD,
        "accuracy": float((pred == y_test).mean()),
        "balanced_acc": float(balanced_accuracy_score(y_test, pred)),
        "neg_recall": float(recall_score(y_test, pred, pos_label=0)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
    }


def evaluate_track_c_quantiles(
    df: pd.DataFrame,
    quantiles: Iterable[float] = (0.80, 0.90, 0.95),
) -> pd.DataFrame:
    model_df = prepare_model_frame(df)
    X = model_df[PRE_ORDER_COLS]
    y = model_df["delivery_days"]
    groups = model_df["order_id"]
    X_train, X_test, y_train, y_test, _, _ = split_by_order(X, y, groups)

    rows = []
    current_over_3_rate = float(((y_test - X_test["expected_delivery_days"]) > 3).mean())
    current_any_delay_rate = float(((y_test - X_test["expected_delivery_days"]) > 0).mean())

    for q in quantiles:
        model = build_track_c_pipeline(X_train, quantile=q)
        model.fit(X_train, y_train)
        pred_delivery = model.predict(X_test)
        recommended = np.maximum(X_test["expected_delivery_days"].to_numpy(), np.ceil(pred_delivery))
        over_3_rate = float(((y_test.to_numpy() - recommended) > 3).mean())
        any_delay_rate = float(((y_test.to_numpy() - recommended) > 0).mean())
        adjustment = recommended - X_test["expected_delivery_days"].to_numpy()

        rows.append(
            {
                "quantile": q,
                "mae_against_actual_delivery_days": float(mean_absolute_error(y_test, pred_delivery)),
                "current_any_delay_rate": current_any_delay_rate,
                "recommended_any_delay_rate": any_delay_rate,
                "current_over_3_delay_rate": current_over_3_rate,
                "recommended_over_3_delay_rate": over_3_rate,
                "avg_adjustment_days": float(np.mean(adjustment)),
                "median_adjustment_days": float(np.median(adjustment)),
                "share_orders_adjusted": float((adjustment > 0).mean()),
            }
        )

    return pd.DataFrame(rows)


def make_recommendation_examples(
    df: pd.DataFrame,
    n_examples: int = 20,
    quantile: float = DEFAULT_TRACK_C_QUANTILE,
    high_risk_only: bool = True,
    risk_threshold: float = TRACK_B_RISK_THRESHOLD,
    caution_threshold: float = TRACK_B_CAUTION_RISK_THRESHOLD,
) -> pd.DataFrame:
    model_df = prepare_model_frame(df)
    X = model_df[PRE_ORDER_COLS]
    y = review_risk_target(model_df["review_score"])
    groups = model_df["order_id"]
    X_train, X_test, y_train, _, _, _ = split_by_order(X, y, groups)

    track_b = build_track_b_pipeline(X_train)
    track_b.fit(X_train, y_train)
    track_c = build_track_c_pipeline(X_train, quantile=quantile)
    track_c.fit(X_train, model_df.loc[X_train.index, "delivery_days"])

    positive_proba = track_b.predict_proba(X_test)[:, 1]
    risk_proba = 1 - positive_proba
    pred_delivery = track_c.predict(X_test)
    recommended = np.maximum(X_test["expected_delivery_days"].to_numpy(), np.ceil(pred_delivery))
    adjustment = recommended - X_test["expected_delivery_days"].to_numpy()

    examples = X_test.copy()
    examples["review_risk_probability"] = risk_proba
    examples["predicted_delivery_days_p90"] = pred_delivery
    examples["recommended_expected_days"] = recommended
    examples["adjustment_days"] = adjustment
    examples["risk_level"] = [
        risk_level_from_probability(
            probability,
            risk_threshold=risk_threshold,
            caution_threshold=caution_threshold,
        )
        for probability in risk_proba
    ]
    if high_risk_only:
        examples = examples[examples["risk_level"] == "고위험"]
    return (
        examples.sort_values("review_risk_probability", ascending=False)
        .head(n_examples)
        .reset_index(drop=True)
    )
