from pathlib import Path

import pandas as pd
import streamlit as st

from src.olist_delivery_models import (
    DEFAULT_TRACK_C_QUANTILE,
    PRE_ORDER_COLS,
    add_pre_order_features,
    load_ml_data,
    predict_order,
    train_models,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "ml_data.csv"


st.set_page_config(
    page_title="Olist CS Risk Console",
    page_icon="",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_source_data() -> pd.DataFrame:
    return load_ml_data(DATA_PATH)


@st.cache_resource(show_spinner="모델 학습 중입니다.")
def train_cached_models(df: pd.DataFrame):
    return train_models(df, quantile=DEFAULT_TRACK_C_QUANTILE)


def option_list(df: pd.DataFrame, column: str) -> list:
    return sorted(df[column].dropna().unique().tolist())


def build_input_row(values: dict) -> pd.DataFrame:
    row = pd.DataFrame([values])
    prepared = add_pre_order_features(row)
    return prepared[PRE_ORDER_COLS]


def risk_action_text(risk_level: str, adjustment_days: float) -> list[str]:
    if risk_level == "고위험":
        actions = [
            "판매자 출고 상태 우선 확인",
            "예상 배송일 임박 전 고객 선제 안내",
            "CS 모니터링 대상 등록",
        ]
        if adjustment_days > 0:
            actions.insert(1, "추천 예상 배송일 기준으로 보수적 배송 약속 검토")
        return actions
    if risk_level == "주의":
        return [
            "배송 상태 모니터링 강화",
            "예상 배송일 초과 조짐 발생 시 고객 안내",
        ]
    return ["일반 처리"]


def render_result(result: dict[str, float | str]) -> None:
    risk_level = str(result["risk_level"])
    risk_probability = float(result["review_risk_probability"])
    current_expected_days = float(result["current_expected_days"])
    recommended_expected_days = float(result["recommended_expected_days"])
    adjustment_days = float(result["adjustment_days"])
    predicted_delivery_days = float(result["predicted_delivery_days"])

    status_color = {
        "고위험": "#b42318",
        "주의": "#b54708",
        "일반": "#067647",
    }[risk_level]

    st.markdown(
        f"""
        <div style="border:1px solid #d0d5dd; padding:18px; border-radius:8px;">
            <div style="font-size:13px; color:#667085;">Track B 판정</div>
            <div style="font-size:28px; font-weight:700; color:{status_color};">{risk_level}</div>
            <div style="font-size:15px; margin-top:4px;">부정 리뷰 위험 확률 {risk_probability:.1%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(3)
    metric_cols[0].metric("현재 예상 배송일", f"{current_expected_days:.0f}일")
    metric_cols[1].metric(
        "Track C 추천 예상 배송일",
        f"{recommended_expected_days:.0f}일",
        f"{adjustment_days:+.0f}일",
    )
    metric_cols[2].metric(
        "90% 분위수 배송 소요일",
        f"{predicted_delivery_days:.1f}일",
    )

    st.subheader("운영 액션")
    for action in risk_action_text(risk_level, adjustment_days):
        st.write(f"- {action}")


def main() -> None:
    df = load_source_data()
    models = train_cached_models(df)

    st.title("Olist CS Risk Console")

    sample_df = df.drop_duplicates("order_id").reset_index(drop=True)
    selected_order_id = st.selectbox(
        "샘플 주문",
        sample_df["order_id"].head(500).tolist(),
        index=0,
    )
    sample = sample_df.loc[sample_df["order_id"] == selected_order_id].iloc[0]

    with st.sidebar:
        st.header("주문 입력")
        price = st.number_input("상품 가격", min_value=0.0, value=float(sample["price"]), step=10.0)
        freight_value = st.number_input(
            "배송비", min_value=0.0, value=float(sample["freight_value"]), step=1.0
        )
        expected_delivery_days = st.number_input(
            "현재 예상 배송일",
            min_value=1,
            max_value=80,
            value=int(sample["expected_delivery_days"]),
            step=1,
        )
        distance_km = st.number_input(
            "고객-셀러 거리(km)",
            min_value=0.0,
            value=float(sample["distance_km"]),
            step=10.0,
        )
        order_item_id = st.number_input(
            "주문 아이템 번호",
            min_value=1,
            max_value=30,
            value=int(sample["order_item_id"]),
            step=1,
        )
        order_purchase_month = st.slider(
            "주문 월",
            min_value=1,
            max_value=12,
            value=int(sample["order_purchase_month"]),
        )
        order_purchase_dayofweek = st.selectbox(
            "주문 요일",
            option_list(df, "order_purchase_dayofweek"),
            index=option_list(df, "order_purchase_dayofweek").index(sample["order_purchase_dayofweek"]),
        )

        st.header("상품")
        product_weight_g = st.number_input(
            "무게(g)",
            min_value=0.0,
            value=float(sample["product_weight_g"]),
            step=100.0,
        )
        product_length_cm = st.number_input(
            "길이(cm)",
            min_value=0.0,
            value=float(sample["product_length_cm"]),
            step=1.0,
        )
        product_height_cm = st.number_input(
            "높이(cm)",
            min_value=0.0,
            value=float(sample["product_height_cm"]),
            step=1.0,
        )
        product_width_cm = st.number_input(
            "너비(cm)",
            min_value=0.0,
            value=float(sample["product_width_cm"]),
            step=1.0,
        )
        main_category = st.selectbox(
            "대분류",
            option_list(df, "main_category"),
            index=option_list(df, "main_category").index(sample["main_category"]),
        )
        sub_category = st.selectbox(
            "중분류",
            option_list(df, "sub_category"),
            index=option_list(df, "sub_category").index(sample["sub_category"]),
        )

        st.header("지역")
        customer_state = st.selectbox(
            "고객 주",
            option_list(df, "customer_state"),
            index=option_list(df, "customer_state").index(sample["customer_state"]),
        )
        seller_state = st.selectbox(
            "셀러 주",
            option_list(df, "seller_state"),
            index=option_list(df, "seller_state").index(sample["seller_state"]),
        )
        distance_cat = st.selectbox(
            "거리 구간",
            option_list(df, "distance_cat"),
            index=option_list(df, "distance_cat").index(sample["distance_cat"]),
        )

    values = {
        "order_item_id": order_item_id,
        "price": price,
        "freight_value": freight_value,
        "product_weight_g": product_weight_g,
        "product_length_cm": product_length_cm,
        "product_height_cm": product_height_cm,
        "product_width_cm": product_width_cm,
        "order_purchase_dayofweek": order_purchase_dayofweek,
        "order_purchase_month": order_purchase_month,
        "expected_delivery_days": expected_delivery_days,
        "main_category": main_category,
        "sub_category": sub_category,
        "distance_km": distance_km,
        "distance_cat": distance_cat,
        "customer_state": customer_state,
        "seller_state": seller_state,
        "cross_state": int(customer_state != seller_state),
    }

    input_row = build_input_row(values)
    result = predict_order(models, input_row)

    left, right = st.columns([1.15, 0.85])
    with left:
        render_result(result)
    with right:
        st.subheader("입력 주문")
        st.dataframe(input_row, use_container_width=True, hide_index=True)

    st.caption(
        "Track C는 90% 분위수를 기본 운영 기준으로 사용한다. 실제 운영 기준은 지연 감소 효과와 예상 배송일 증가에 따른 전환 손실을 함께 비교해 조정해야 한다."
    )


if __name__ == "__main__":
    main()
