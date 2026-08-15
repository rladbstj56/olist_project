from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.olist_delivery_models import (
    DEFAULT_TRACK_C_QUANTILE,
    PRE_ORDER_COLS,
    TRACK_B_CAUTION_RISK_THRESHOLD,
    TRACK_B_RISK_THRESHOLD,
    TRACK_C_QUANTILE_OPTIONS,
    add_pre_order_features,
    load_ml_data,
    predict_order,
    train_console_artifacts,
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
def train_cached_console(
    df: pd.DataFrame,
    quantile: float,
    risk_threshold: float,
    caution_threshold: float,
):
    return train_console_artifacts(
        df,
        quantile=quantile,
        risk_threshold=risk_threshold,
        caution_threshold=caution_threshold,
    )


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


def render_result(result: dict[str, float | str], selected_quantile: float) -> None:
    risk_level = str(result["risk_level"])
    risk_probability = float(result["review_risk_probability"])
    current_expected_days = float(result["current_expected_days"])
    recommended_expected_days = float(result["recommended_expected_days"])
    adjustment_days = float(result["adjustment_days"])
    predicted_delivery_days = float(result["predicted_delivery_days"])
    is_track_c_target = bool(result["is_track_c_target"])

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
    if is_track_c_target:
        metric_cols[1].metric(
            "Track C 추천 예상 배송일",
            f"{recommended_expected_days:.0f}일",
            f"{adjustment_days:+.0f}일",
        )
    else:
        metric_cols[1].metric("Track C 적용", "비대상")
    metric_cols[2].metric(
        f"{selected_quantile:.0%} 분위수 배송 소요일",
        f"{predicted_delivery_days:.1f}일",
    )

    if not is_track_c_target:
        st.info("Track C 추천 예상 배송일은 Track B가 고위험으로 판정한 주문에만 운영 액션으로 적용합니다.")

    st.subheader("운영 액션")
    for action in risk_action_text(risk_level, adjustment_days):
        st.write(f"- {action}")


def main() -> None:
    df = load_source_data()

    st.title("Olist CS Risk Console")

    with st.sidebar:
        st.header("운영 기준")
        risk_threshold_percent = st.slider(
            "Track B 고위험 기준",
            min_value=20,
            max_value=80,
            value=int(round(TRACK_B_RISK_THRESHOLD * 100)),
            step=1,
            help="부정 리뷰 위험 확률이 이 값보다 높은 주문만 Track C 추천 대상이 됩니다.",
        )
        caution_threshold_percent = st.slider(
            "Track B 주의 기준",
            min_value=10,
            max_value=max(10, risk_threshold_percent - 1),
            value=min(int(round(TRACK_B_CAUTION_RISK_THRESHOLD * 100)), risk_threshold_percent - 1),
            step=1,
            help="고위험은 아니지만 배송 모니터링이 필요한 중간 위험군 기준입니다.",
        )
        selected_quantile_percent = st.select_slider(
            "Track C 분위수 기준",
            options=[int(q * 100) for q in TRACK_C_QUANTILE_OPTIONS],
            value=int(DEFAULT_TRACK_C_QUANTILE * 100),
            help="값이 높을수록 추천 예상 배송일이 보수적으로 길어집니다.",
        )

    risk_threshold = risk_threshold_percent / 100
    caution_threshold = caution_threshold_percent / 100
    selected_quantile = selected_quantile_percent / 100

    console = train_cached_console(df, selected_quantile, risk_threshold, caution_threshold)
    models = console.models

    with st.expander("Track B 판정 기준", expanded=False):
        st.write(
            f"- 고위험: 부정 리뷰 위험 확률이 {risk_threshold:.0%} 초과인 주문"
        )
        st.write(
            f"- 주의: 부정 리뷰 위험 확률이 {caution_threshold:.0%} 초과부터 {risk_threshold:.0%} 이하인 주문"
        )
        st.write("- 일반: 부정 리뷰 위험 확률이 주의 기준 이하인 주문")
        st.write(f"- Track C 분위수 기준: {selected_quantile:.0%}")
        st.write("Track C 추천 예상 배송일은 고위험 주문에만 적용합니다.")
        st.write("기본값은 검증셋 threshold와 90% 분위수 운영 가정에 기반하며, 실제 운영에서는 CS 처리 가능량과 구매 전환 손실을 고려해 조정할 수 있습니다.")

    sample_df = console.test_orders
    selected_order_id = st.selectbox(
        "테스트 주문 예시",
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
    result = predict_order(
        models,
        input_row,
        risk_threshold=risk_threshold,
        caution_threshold=caution_threshold,
    )

    left, right = st.columns([1.15, 0.85])
    with left:
        render_result(result, selected_quantile)
    with right:
        st.subheader("입력 주문")
        st.dataframe(input_row, use_container_width=True, hide_index=True)

    st.subheader("고위험 주문 관리 목록")
    st.caption(
        f"현재 기준에서 고위험으로 분류된 테스트 주문 {len(console.high_risk_orders):,}건입니다. 이 목록이 Track C 추천 예상 배송일 검토 대상입니다."
    )
    high_risk_view = console.high_risk_orders[
        [
            "order_id",
            "review_risk_probability",
            "expected_delivery_days",
            "predicted_delivery_days_quantile",
            "recommended_expected_days",
            "adjustment_days",
            "price",
            "freight_value",
            "main_category",
            "sub_category",
            "distance_km",
            "customer_state",
            "seller_state",
        ]
    ].head(50).copy()
    high_risk_view["review_risk_probability"] = high_risk_view["review_risk_probability"] * 100
    st.dataframe(
        high_risk_view,
        use_container_width=True,
        hide_index=True,
        column_config={
            "order_id": "주문 ID",
            "review_risk_probability": st.column_config.NumberColumn("부정 리뷰 위험 확률", format="%.1f%%"),
            "expected_delivery_days": "현재 예상 배송일",
            "predicted_delivery_days_quantile": st.column_config.NumberColumn(
                f"{selected_quantile:.0%} 분위수 배송 소요일",
                format="%.1f",
            ),
            "recommended_expected_days": "추천 예상 배송일",
            "adjustment_days": "조정일수",
            "price": "상품 가격",
            "freight_value": "배송비",
            "main_category": "대분류",
            "sub_category": "중분류",
            "distance_km": st.column_config.NumberColumn("거리(km)", format="%.1f"),
            "customer_state": "고객 주",
            "seller_state": "셀러 주",
        },
    )

    st.caption(
        "기본값은 Track B 고위험 기준 47%, Track C 분위수 90%입니다. 실제 운영 기준은 지연 감소 효과와 예상 배송일 증가에 따른 전환 손실을 함께 비교해 조정해야 합니다."
    )


if __name__ == "__main__":
    main()
