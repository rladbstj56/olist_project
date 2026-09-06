from pathlib import Path
import json
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.olist_delivery_models import (
    DEFAULT_LABEL_POLICY,
    DEFAULT_TRACK_C_QUANTILE,
    LABEL_POLICY_DESCRIPTIONS,
    evaluate_label_policies,
    evaluate_track_a_vs_b,
    evaluate_track_b,
    evaluate_track_c_quantiles,
    load_ml_data,
    make_recommendation_examples,
)


DATA_PATH = PROJECT_ROOT / "data" / "processed" / "ml_data.csv"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
METADATA_DIR = PROJECT_ROOT / "outputs" / "metadata"


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    df = load_ml_data(DATA_PATH)

    track_b_result = evaluate_track_b(df, label_policy=DEFAULT_LABEL_POLICY)
    track_a_vs_b = evaluate_track_a_vs_b(df, label_policy=DEFAULT_LABEL_POLICY)
    label_policy_result = evaluate_label_policies(df)
    track_c_result = evaluate_track_c_quantiles(df, quantiles=(0.80, 0.90, 0.95))
    examples = make_recommendation_examples(df, quantile=DEFAULT_TRACK_C_QUANTILE)

    track_a_vs_b.to_csv(TABLE_DIR / "track_a_vs_b_comparison.csv", index=False)
    label_policy_result.to_csv(TABLE_DIR / "label_policy_comparison.csv", index=False)
    track_c_result.to_csv(TABLE_DIR / "track_c_quantile_results.csv", index=False)
    examples.to_csv(TABLE_DIR / "track_c_recommendation_examples.csv", index=False)

    summary = {
        "selected_label_policy": DEFAULT_LABEL_POLICY,
        "selected_label_policy_description": LABEL_POLICY_DESCRIPTIONS[DEFAULT_LABEL_POLICY],
        "selected_track_a_model": "LightGBM",
        "selected_track_a_threshold": float(track_a_vs_b.iloc[0]["positive_threshold"]),
        "selected_track_b_model": "LightGBM",
        "track_b_best_params": {
            "classifier__colsample_bytree": 0.5028265220878869,
            "classifier__learning_rate": 0.011153121252070788,
            "classifier__max_depth": 5,
            "classifier__min_child_samples": 47,
            "classifier__n_estimators": 800,
            "classifier__num_leaves": 29,
        },
        "track_b_positive_threshold": float(track_b_result["positive_threshold"]),
        "track_b_risk_threshold": float(track_b_result["risk_threshold"]),
        "track_b_risk_precision": float(track_b_result["risk_precision"]),
        "track_b_risk_recall": float(track_b_result["risk_recall"]),
        "track_b_risk_pr_auc": float(track_b_result["risk_pr_auc"]),
        "track_b_flagged_rate": float(track_b_result["flagged_rate"]),
        "modeled_rows": int(track_b_result["modeled_rows"]),
        "dropped_rows": int(track_b_result["dropped_rows"]),
        "risk_rate": float(track_b_result["risk_rate"]),
        "positive_rate": float(track_b_result["positive_rate"]),
    }
    (METADATA_DIR / "refreshed_search_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Track B validation-style result")
    for key, value in track_b_result.items():
        print(f"- {key}: {value:.4f}" if isinstance(value, float) else f"- {key}: {value}")

    print("\nTrack A vs Track B comparison")
    print(track_a_vs_b.to_string(index=False))

    print("\nLabel policy comparison")
    print(label_policy_result.to_string(index=False))

    print("\nTrack C quantile comparison")
    print(track_c_result.to_string(index=False))

    print("\nSaved files")
    print(f"- {TABLE_DIR / 'track_a_vs_b_comparison.csv'}")
    print(f"- {TABLE_DIR / 'label_policy_comparison.csv'}")
    print(f"- {TABLE_DIR / 'track_c_quantile_results.csv'}")
    print(f"- {TABLE_DIR / 'track_c_recommendation_examples.csv'}")
    print(f"- {METADATA_DIR / 'refreshed_search_summary.json'}")


if __name__ == "__main__":
    main()
