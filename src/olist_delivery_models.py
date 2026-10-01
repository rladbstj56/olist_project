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
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .olist_imputation import RAW_INPUT_COLS, TrainOnlyImputer, attach_raw_imputation_columns


TRACK_B_POSITIVE_THRESHOLD = 0.54
TRACK_B_RISK_THRESHOLD = 1 - TRACK_B_POSITIVE_THRESHOLD
TRACK_B_CAUTION_RISK_THRESHOLD = 0.35
DEFAULT_TRACK_C_QUANTILE = 0.90
TRACK_C_QUANTILE_OPTIONS = (0.80, 0.85, 0.90, 0.95)
RANDOM_STATE = 42
DEFAULT_LABEL_POLICY = "1_2_negative_4_5_positive_drop_3"

LABEL_POLICY_DESCRIPTIONS = {
    DEFAULT_LABEL_POLICY: "1-2점은 낮은 만족도, 4-5점은 긍정, 3점은 중립으로 제외",
    "1_2_negative_3_5_non_negative": "1-2점은 낮은 만족도, 3-5점은 비위험",
    "1_3_risk_4_5_positive": "1-3점은 CS 리스크, 4-5점은 긍정",
}


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

TRACK_A_COLS = [
    *PRE_ORDER_COLS,
    "approved_days",
    "dispatch_days",
    "delivery_days",
    "delay_days",
    "is_delayed",
    "delay_days_cat",
    "delivery_speed",
    "day_per_km",
    "delivery_ratio",
    "delivery_distance",
    "delivery_price",
]


@dataclass
class TrainedModels:
    track_b: Pipeline
    track_c: Pipeline
    feature_cols: list[str]
    label_policy: str = DEFAULT_LABEL_POLICY
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


def add_post_delivery_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["delivery_speed"] = safe_divide(out["distance_km"], out["delivery_days"])
    out["day_per_km"] = safe_divide(out["delivery_days"], out["distance_km"])
    out["delivery_ratio"] = safe_divide(out["delivery_days"], out["expected_delivery_days"])
    out["delivery_distance"] = out["delivery_days"] * out["distance_km"]
    out["delivery_price"] = out["delivery_days"] * out["price"]
    return out.replace([np.inf, -np.inf], np.nan)


def review_risk_target(
    review_score: pd.Series,
    label_policy: str = DEFAULT_LABEL_POLICY,
) -> pd.Series:
    score = pd.to_numeric(review_score, errors="coerce")
    target = pd.Series(pd.NA, index=review_score.index, dtype="Int64")

    if label_policy == DEFAULT_LABEL_POLICY:
        target.loc[score.isin([1, 2])] = 0
        target.loc[score.isin([4, 5])] = 1
    elif label_policy == "1_2_negative_3_5_non_negative":
        target.loc[score.isin([1, 2])] = 0
        target.loc[score.isin([3, 4, 5])] = 1
    elif label_policy == "1_3_risk_4_5_positive":
        target.loc[score.isin([1, 2, 3])] = 0
        target.loc[score.isin([4, 5])] = 1
    else:
        valid = ", ".join(LABEL_POLICY_DESCRIPTIONS)
        raise ValueError(f"Unknown label_policy: {label_policy}. Valid policies: {valid}")

    return target


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


def split_train_valid_test_by_order(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    test_size: float = 0.2,
    valid_size: float = 0.2,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    splitter_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    train_valid_idx, test_idx = next(splitter_test.split(X, y, groups=groups))

    X_train_valid = X.iloc[train_valid_idx]
    y_train_valid = y.iloc[train_valid_idx]
    groups_train_valid = groups.iloc[train_valid_idx]
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]
    groups_test = groups.iloc[test_idx]

    splitter_valid = GroupShuffleSplit(n_splits=1, test_size=valid_size, random_state=RANDOM_STATE)
    train_idx, valid_idx = next(
        splitter_valid.split(X_train_valid, y_train_valid, groups=groups_train_valid)
    )

    return (
        X_train_valid.iloc[train_idx],
        X_train_valid.iloc[valid_idx],
        X_test,
        y_train_valid.iloc[train_idx],
        y_train_valid.iloc[valid_idx],
        y_test,
        groups_train_valid.iloc[train_idx],
        groups_train_valid.iloc[valid_idx],
        groups_test,
    )


def select_positive_threshold(
    y_true: pd.Series,
    positive_proba: np.ndarray,
    thresholds: Iterable[float] = np.arange(0.10, 0.91, 0.01),
) -> float:
    rows = [
        risk_classification_metrics(y_true, positive_proba, positive_threshold=float(threshold))
        for threshold in thresholds
    ]
    threshold_df = pd.DataFrame(rows)
    threshold_df["gap"] = (threshold_df["risk_recall"] - threshold_df["balanced_acc"]).abs()
    return float(threshold_df.loc[threshold_df["gap"].idxmin(), "positive_threshold"])


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


def build_track_a_pipeline(X: pd.DataFrame) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(X)),
            (
                "classifier",
                LGBMClassifier(
                    n_estimators=800,
                    learning_rate=0.014883605700319194,
                    num_leaves=47,
                    max_depth=7,
                    colsample_bytree=0.6218455076693483,
                    min_child_samples=47,
                    class_weight="balanced",
                    n_jobs=1,
                    random_state=RANDOM_STATE,
                    verbose=-1,
                ),
            ),
        ]
    )


def prepare_corrected_model_frame(
    df: pd.DataFrame,
    label_policy: str = DEFAULT_LABEL_POLICY,
    raw_data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach unimputed source values without learning any replacement statistics."""
    if raw_data is None:
        raw_data = pd.read_csv(Path(__file__).resolve().parents[1] / "data/processed/merged_data.csv")
    return attach_raw_imputation_columns(prepare_labeled_model_frame(df, label_policy), raw_data)


def build_corrected_track_pipeline(X: pd.DataFrame, track: str) -> Pipeline:
    """A/B augmented features -> fold-local imputation followed by the existing estimator."""
    if track == "A":
        columns, builder = TRACK_A_COLS, build_track_a_pipeline
    elif track == "B":
        columns, builder = PRE_ORDER_COLS, build_track_b_pipeline
    else:
        raise ValueError("track must be A or B")
    legacy = builder(X[columns])
    return Pipeline([("train_only_imputation", TrainOnlyImputer(tuple(columns))), *legacy.steps])


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
    prepared = add_post_delivery_features(prepared)
    required_cols = list(dict.fromkeys(["order_id", "review_score", "delivery_days", *TRACK_A_COLS]))
    return prepared[required_cols].dropna(subset=["order_id", "review_score", "delivery_days"])


def prepare_labeled_model_frame(
    df: pd.DataFrame,
    label_policy: str = DEFAULT_LABEL_POLICY,
) -> pd.DataFrame:
    model_df = prepare_model_frame(df)
    model_df["review_label"] = review_risk_target(model_df["review_score"], label_policy=label_policy)
    model_df = model_df.dropna(subset=["review_label"]).copy()
    model_df["review_label"] = model_df["review_label"].astype(int)
    return model_df


def train_models(
    df: pd.DataFrame,
    quantile: float = DEFAULT_TRACK_C_QUANTILE,
    feature_cols: Iterable[str] = PRE_ORDER_COLS,
    label_policy: str = DEFAULT_LABEL_POLICY,
) -> TrainedModels:
    model_df = prepare_labeled_model_frame(df, label_policy=label_policy)
    feature_cols = list(feature_cols)
    X = model_df[feature_cols]
    y_track_b = model_df["review_label"]
    y_track_c = model_df["delivery_days"]

    track_b = build_track_b_pipeline(X)
    track_b.fit(X, y_track_b)

    track_c = build_track_c_pipeline(X, quantile=quantile)
    track_c.fit(X, y_track_c)

    return TrainedModels(
        track_b=track_b,
        track_c=track_c,
        feature_cols=feature_cols,
        label_policy=label_policy,
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
    label_policy: str = DEFAULT_LABEL_POLICY,
) -> ConsoleArtifacts:
    model_df = prepare_labeled_model_frame(df, label_policy=label_policy)
    X = model_df[PRE_ORDER_COLS]
    y_track_b = model_df["review_label"]
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
        label_policy=label_policy,
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


def risk_classification_metrics(
    y_true: pd.Series,
    positive_proba: np.ndarray,
    positive_threshold: float = TRACK_B_POSITIVE_THRESHOLD,
) -> dict[str, float]:
    pred = (positive_proba >= positive_threshold).astype(int)
    risk_true = (y_true.to_numpy() == 0).astype(int)
    risk_pred = (pred == 0).astype(int)
    risk_proba = 1 - positive_proba

    return {
        "positive_threshold": positive_threshold,
        "risk_threshold": 1 - positive_threshold,
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, pred)),
        "risk_precision": float(precision_score(risk_true, risk_pred, zero_division=0)),
        "risk_recall": float(recall_score(risk_true, risk_pred, zero_division=0)),
        "risk_pr_auc": float(average_precision_score(risk_true, risk_proba)),
        "flagged_rate": float(risk_pred.mean()),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
    }


def label_distribution(
    model_df: pd.DataFrame,
    source_rows: int,
    label_policy: str,
) -> dict[str, float | int | str]:
    review_counts = model_df["review_score"].value_counts().sort_index()
    return {
        "label_policy": label_policy,
        "description": LABEL_POLICY_DESCRIPTIONS[label_policy],
        "source_rows": int(source_rows),
        "modeled_rows": int(len(model_df)),
        "dropped_rows": int(source_rows - len(model_df)),
        "dropped_rate": float((source_rows - len(model_df)) / source_rows),
        "risk_rows": int((model_df["review_label"] == 0).sum()),
        "positive_rows": int((model_df["review_label"] == 1).sum()),
        "risk_rate": float((model_df["review_label"] == 0).mean()),
        "positive_rate": float((model_df["review_label"] == 1).mean()),
        "review_1_rows": int(review_counts.get(1, 0)),
        "review_2_rows": int(review_counts.get(2, 0)),
        "review_3_rows": int(review_counts.get(3, 0)),
        "review_4_rows": int(review_counts.get(4, 0)),
        "review_5_rows": int(review_counts.get(5, 0)),
    }


def evaluate_track_b(
    df: pd.DataFrame,
    label_policy: str = DEFAULT_LABEL_POLICY,
    raw_data: pd.DataFrame | None = None,
) -> dict[str, float | int | str]:
    source_frame = prepare_model_frame(df)
    model_df = prepare_corrected_model_frame(df, label_policy=label_policy, raw_data=raw_data)
    X = model_df[PRE_ORDER_COLS + RAW_INPUT_COLS]
    y = model_df["review_label"]
    groups = model_df["order_id"]
    X_train, X_valid, X_test, y_train, y_valid, y_test, _, _, groups_test = (
        split_train_valid_test_by_order(X, y, groups)
    )

    model = build_corrected_track_pipeline(X_train, "B")
    model.fit(X_train, y_train)
    valid_proba = model.predict_proba(X_valid)[:, 1]
    positive_threshold = select_positive_threshold(y_valid, valid_proba)
    positive_proba = model.predict_proba(X_test)[:, 1]

    metrics = risk_classification_metrics(
        y_test,
        positive_proba,
        positive_threshold=positive_threshold,
    )
    metrics.update(label_distribution(model_df, len(source_frame), label_policy))
    metrics["track"] = "Track B (LightGBM, 사전 예측)"
    metrics["n_features"] = len(PRE_ORDER_COLS)
    metrics["test_rows"] = int(len(y_test))
    metrics["test_orders"] = int(groups_test.nunique())
    return metrics


def evaluate_track_a_vs_b(
    df: pd.DataFrame,
    label_policy: str = DEFAULT_LABEL_POLICY,
    raw_data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    model_df = prepare_corrected_model_frame(df, label_policy=label_policy, raw_data=raw_data)
    y = model_df["review_label"]
    groups = model_df["order_id"]
    rows = []

    for track, feature_cols, track_id in [
        ("Track A (LightGBM, 사후 원인분석)", TRACK_A_COLS, "A"),
        ("Track B (LightGBM, 사전 예측)", PRE_ORDER_COLS, "B"),
    ]:
        X = model_df[feature_cols + RAW_INPUT_COLS]
        X_train, X_valid, X_test, y_train, y_valid, y_test, _, _, groups_test = (
            split_train_valid_test_by_order(X, y, groups)
        )
        model = build_corrected_track_pipeline(X_train, track_id)
        model.fit(X_train, y_train)
        valid_proba = model.predict_proba(X_valid)[:, 1]
        positive_threshold = select_positive_threshold(y_valid, valid_proba)
        positive_proba = model.predict_proba(X_test)[:, 1]
        row = risk_classification_metrics(
            y_test,
            positive_proba,
            positive_threshold=positive_threshold,
        )
        row.update(
            {
                "label_policy": label_policy,
                "track": track,
                "n_features": len(feature_cols),
                "test_rows": int(len(y_test)),
                "test_orders": int(groups_test.nunique()),
            }
        )
        rows.append(row)

    X_baseline = model_df[PRE_ORDER_COLS]
    _, _, X_test_b, y_train_b, _, y_test_b, _, _, groups_test_b = split_train_valid_test_by_order(
        X_baseline,
        y,
        groups,
    )
    majority_label = int(y_train_b.mode().iloc[0])
    baseline_positive_proba = np.full(len(y_test_b), float(majority_label))
    row = risk_classification_metrics(y_test_b, baseline_positive_proba)
    row.update(
        {
            "label_policy": label_policy,
            "track": "Baseline (다수 클래스)",
            "n_features": 0,
            "test_rows": int(len(y_test_b)),
            "test_orders": int(groups_test_b.nunique()),
        }
    )
    rows.append(row)

    return pd.DataFrame(rows)


def evaluate_label_policies(df: pd.DataFrame, raw_data: pd.DataFrame | None = None) -> pd.DataFrame:
    source_frame = prepare_model_frame(df)
    rows = []

    for policy in LABEL_POLICY_DESCRIPTIONS:
        model_df = prepare_labeled_model_frame(df, label_policy=policy)
        track_b_metrics = evaluate_track_b(df, label_policy=policy, raw_data=raw_data)
        baseline_row = evaluate_track_a_vs_b(df, label_policy=policy, raw_data=raw_data)
        baseline_metrics = baseline_row[baseline_row["track"] == "Baseline (다수 클래스)"].iloc[0]
        row = label_distribution(model_df, len(source_frame), policy)
        row.update(
            {
                "baseline_accuracy": float(baseline_metrics["accuracy"]),
                "baseline_balanced_acc": float(baseline_metrics["balanced_acc"]),
                "baseline_risk_recall": float(baseline_metrics["risk_recall"]),
                "track_b_accuracy": float(track_b_metrics["accuracy"]),
                "track_b_balanced_acc": float(track_b_metrics["balanced_acc"]),
                "track_b_risk_precision": float(track_b_metrics["risk_precision"]),
                "track_b_risk_recall": float(track_b_metrics["risk_recall"]),
                "track_b_risk_pr_auc": float(track_b_metrics["risk_pr_auc"]),
                "track_b_flagged_rate": float(track_b_metrics["flagged_rate"]),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


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
    label_policy: str = DEFAULT_LABEL_POLICY,
) -> pd.DataFrame:
    model_df = prepare_labeled_model_frame(df, label_policy=label_policy)
    X = model_df[PRE_ORDER_COLS]
    y = model_df["review_label"]
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
