from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.olist_delivery_models import (
    DEFAULT_TRACK_C_QUANTILE,
    evaluate_track_b,
    evaluate_track_c_quantiles,
    load_ml_data,
    make_recommendation_examples,
)


DATA_PATH = PROJECT_ROOT / "data" / "processed" / "ml_data.csv"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df = load_ml_data(DATA_PATH)

    track_b_result = evaluate_track_b(df)
    track_c_result = evaluate_track_c_quantiles(df, quantiles=(0.80, 0.90, 0.95))
    examples = make_recommendation_examples(df, quantile=DEFAULT_TRACK_C_QUANTILE)

    track_c_result.to_csv(TABLE_DIR / "track_c_quantile_results.csv", index=False)
    examples.to_csv(TABLE_DIR / "track_c_recommendation_examples.csv", index=False)

    print("Track B validation-style result")
    for key, value in track_b_result.items():
        print(f"- {key}: {value:.4f}" if isinstance(value, float) else f"- {key}: {value}")

    print("\nTrack C quantile comparison")
    print(track_c_result.to_string(index=False))

    print("\nSaved files")
    print(f"- {TABLE_DIR / 'track_c_quantile_results.csv'}")
    print(f"- {TABLE_DIR / 'track_c_recommendation_examples.csv'}")


if __name__ == "__main__":
    main()
