# 모델링 의사결정 및 수정 기록

작성일: 2026-07-27
대상 파일: `preprocessed_final.ipynb`, `eda_final.ipynb`, `ml_classifier.ipynb`

## 1. 문서 목적

이 문서는 지금까지 EDA, 전처리, 분류 모델링 과정에서 결정한 내용과 수정된 사항을 기록하기 위한 문서다. 단순히 어떤 코드를 바꿨는지가 아니라, 왜 그렇게 수정했는지의 판단 기준을 함께 남긴다.

분석의 핵심 목적은 Olist 주문 데이터를 기반으로 리뷰 점수가 낮아질 위험을 탐지하는 것이다. 다만 모델을 언제 사용할 것인지에 따라 사용할 수 있는 컬럼이 달라지므로, 모델을 Track A와 Track B로 분리했다.

## 2. 전체 분석 흐름

1. `preprocessed_final.ipynb`

   - 원천 CSV들을 병합하고 분석용 기본 데이터셋을 생성한다.
   - 최종 산출물은 `data/merged_final_data.csv`다.
2. `eda_final.ipynb`

   - 전처리된 데이터를 기반으로 탐색적 분석을 수행한다.
   - 모델링에 사용할 학습 기반 데이터셋 `data/merged_train_data.csv`를 생성한다.
3. `ml_classifier.ipynb`

   - 리뷰 점수를 긍정/부정으로 이진 분류한다.
   - Track A와 Track B를 분리해 각각 모델링한다.
   - 모델 비교 결과와 최종 성능 요약 CSV를 생성한다.

## 3. 타깃 정의

리뷰 점수 `review_score`를 다음과 같이 이진화했다.

| 원본 리뷰 점수 | 모델 타깃 | 의미                 |
| -------------- | --------: | -------------------- |
| 1점, 2점, 3점  |         0 | 부정 리뷰, CS 리스크 |
| 4점, 5점       |         1 | 긍정 리뷰            |

기존에 “NPS 기준”처럼 해석될 수 있는 표현은 제거하고, “CS 리스크 기준”으로 수정했다.

판단 기준:

- Olist의 `review_score`는 엄밀한 NPS 문항이 아니다.
- NPS처럼 추천 의향을 측정한 값이 아니라 구매 후 만족도에 가까운 별점이다.
- 따라서 1점부터 3점까지를 CS 리스크로 보는 것이 더 정확하다.

## 4. 현재 인스턴스 기준과 데이터 병합 구조

현재 `merged_final_data.csv`의 한 행은 **주문 1건(order-level)** 이 아니라 **주문 안의 상품 아이템 1건(order-item-level)** 을 의미한다.

정확한 인스턴스 기준:

| 기준                                      |      값 |
| ----------------------------------------- | ------: |
| 최종 행 수                                | 109,294 |
| 고유`order_id` 수                       |  95,784 |
| 고유`(order_id, order_item_id)` 조합 수 | 109,294 |
| 여러 행으로 확장된 주문 수                |   9,507 |
| 여러 아이템 주문에 속한 행 수             |  23,017 |
| 한 주문의 최대 아이템 행 수               |      21 |

즉, 현재 데이터는 “주문별 1행”이 아니라 “주문-아이템별 1행”이다. 하나의 주문에 상품이 여러 개 있으면 `order_id`는 반복되고, `order_item_id`, 상품, 셀러, 가격, 배송비 정보가 아이템 단위로 달라진다.

이 기준을 선택한 이유:

- 상품 카테고리, 상품 무게/크기, 셀러, 가격, 배송비는 아이템 단위 정보다.
- 주문 단위로 강제 집계하면 어떤 상품/셀러/카테고리가 리뷰 리스크와 연결되는지 잃게 된다.
- 이 프로젝트는 배송/상품/셀러 특성이 리뷰 점수에 미치는 리스크를 보려는 목적이 있으므로 아이템 단위가 더 풍부하다.

주의점:

- 리뷰 점수는 `order_id` 단위로 붙는다.
- 따라서 한 주문에 여러 아이템이 있으면 같은 `review_score`가 여러 아이템 행에 복제된다.
- 이 때문에 train/test를 행 단위로 나누면 같은 주문의 일부 아이템이 train에, 다른 아이템이 test에 들어가 data leakage가 발생할 수 있다.
- 그래서 모델링 단계에서 `order_id` 기준 group split을 적용했다.

### 4.1 병합 과정

최종 병합은 `orders`를 출발점으로 하고, 필요한 테이블을 순서대로 붙이는 구조다.

```python
df_final = pd.merge(df_orders, df_customers_geo, on="customer_id", how="left")
df_final = pd.merge(df_final, df_items, on="order_id", how="left")
df_final = pd.merge(df_final, df_reviews, on="order_id", how="left")
df_final = pd.merge(df_final, df_products, on="product_id", how="left")
df_final = pd.merge(df_final, df_sellers_geo, on="seller_id", how="left")
```

병합 구조:

| 순서 | 병합 대상                      | 병합 키         | 병합 후 의미                    |
| ---- | ------------------------------ | --------------- | ------------------------------- |
| 1    | `orders` + `customers_geo` | `customer_id` | 주문에 고객 지역/좌표 정보 추가 |
| 2    | +`order_items`               | `order_id`    | 주문이 아이템 단위 행으로 확장  |
| 3    | +`reviews`                   | `order_id`    | 주문 단위 리뷰 점수 추가        |
| 4    | +`products`                  | `product_id`  | 상품 카테고리, 무게, 크기 추가  |
| 5    | +`sellers_geo`               | `seller_id`   | 셀러 지역/좌표 정보 추가        |

핵심은 2단계다. `orders`는 주문 단위 테이블이고 `order_items`는 아이템 단위 테이블이다. `order_id`로 붙이는 순간 데이터의 인스턴스 기준은 주문 단위에서 주문-아이템 단위로 바뀐다.

### 4.2 병합 전 데이터 증식 방지 처리

병합 전에 데이터가 의도치 않게 늘어나는 것을 막기 위해 몇 가지 기준을 적용했다.

| 처리                   | 기준                                                     | 이유                                                                                                          |
| ---------------------- | -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| geolocation 중복 축약  | `geolocation_zip_code_prefix`별 위도/경도 평균         | 같은 zip prefix에 여러 좌표가 있어 그대로 병합하면 고객/셀러 행이 증식될 수 있음                              |
| reviews 중복 축약      | `order_id`별 최신 `review_answer_timestamp` 1건 유지 | 한 주문에 리뷰가 여러 개 있으면 아이템 병합 후 중복이 더 커질 수 있음                                         |
| payments 제외          | 최종 병합에 사용하지 않음                                | 결제는 한 주문에 여러 결제 수단/회차가 있을 수 있어 그대로 붙이면 주문-아이템 행이 결제 행 수만큼 추가 증식됨 |
| products 카테고리 번역 | `product_category_name` 기준 번역 테이블 left join     | 카테고리 해석 가능성을 높이기 위한 참조 병합                                                                  |

특히 `payments`는 읽고 품질 확인은 했지만 최종 모델링 데이터에는 병합하지 않았다. 결제 테이블은 `order_id` 기준으로 여러 행을 가질 수 있어, 별도 집계 없이 붙이면 현재의 주문-아이템 단위보다 더 세밀한 “주문-아이템-결제” 단위로 데이터가 변해버린다. 이번 모델의 분석 단위와 맞지 않아 제외한 것이 타당하다.

### 4.3 리뷰 중복 축약 기준과 한계

리뷰 테이블은 `order_id` 기준으로 병합했다. 다만 같은 `order_id`에 리뷰가 여러 개 존재하는 경우가 있어, 병합 전에 다음 기준으로 1건만 남겼다.

```python
df_reviews = df_reviews.sort_values(by=['order_id', 'review_answer_timestamp'])
df_reviews = df_reviews.drop_duplicates(subset=['order_id'], keep='last')
```

처리 기준:
- `review_answer_timestamp` 기준 최신 리뷰 1건을 주문의 대표 리뷰로 사용했다.
- 목적은 리뷰 중복으로 인한 row explosion을 막는 것이다.

이 처리가 필요한 이유:
- 현재 최종 데이터는 주문-아이템 단위다.
- 리뷰를 중복 그대로 두면 `orders × items × reviews` 형태가 되어 같은 주문이 리뷰 수만큼 추가 반복된다.
- 그러면 특정 주문이 모델 학습에서 과도한 가중치를 갖고, 리뷰가 많은 주문이 전체 패턴을 왜곡할 수 있다.

프로젝트 한계:
- 최신 리뷰가 반드시 주문의 대표 리뷰라고 보장할 수 없다.
- 다중 상품 주문에서는 리뷰가 특정 상품에 대한 평가인지, 전체 주문/배송 경험에 대한 평가인지 구분할 수 없다.
- 따라서 상품 단위 feature importance를 해석할 때 “해당 상품이 직접 부정 리뷰의 원인”이라고 단정하면 안 된다.
- 이 한계는 처리 실수라기보다 Olist 데이터셋의 리뷰 granularity와 상품 granularity가 맞지 않는 구조적 제약이다.

포트폴리오 표현:
- “리뷰 중복은 모델 편의를 위한 임의 삭제가 아니라, 서로 다른 데이터 단위가 병합될 때 발생하는 row explosion을 막기 위한 대표값 선택입니다.”
- “다만 리뷰가 주문 단위로만 제공되어 상품 단위 원인 해석에는 한계가 있음을 명시했습니다.”

### 4.4 시간 순서 이상치 처리 기준

주문 시간 컬럼은 다음 순서가 일반적인 업무 흐름이다.

```text
order_purchase_timestamp
→ order_approved_at
→ order_delivered_carrier_date
→ order_delivered_customer_date
```

전처리 단계에서 이 순서를 검증했다.

| 검증 조건 | 의미 | 처리 |
|---|---|---|
| `purchase > approved` | 구매 시각이 승인 시각보다 늦음 | 발견 0건 |
| `approved > carrier` | 승인 시각이 택배사 인도 시각보다 늦음 | 제거하지 않음 |
| `carrier > delivered` | 택배사 인도 시각이 고객 도착 시각보다 늦음 | 23건 제거 |
| `delivered < purchase` | 고객 도착 시각이 구매 시각보다 빠름 | 발견 0건 |

삭제한 케이스:
- `order_delivered_carrier_date > order_delivered_customer_date`인 23건은 제거했다.
- 물건이 택배사에 전달된 시각이 고객에게 도착한 시각보다 늦다는 뜻이므로 물류 흐름상 성립하기 어렵다.
- 이 값은 `delivery_days`, `dispatch_days`, `carrier_days`류 파생 변수의 해석을 직접 깨뜨릴 수 있어 제거가 타당하다.

삭제하지 않은 케이스:
- `order_approved_at > order_delivered_carrier_date` 케이스는 제거하지 않았다.
- 업무 흐름상 이례적이지만, Olist 데이터에서 결제 승인 시각이 실제 승인 시점보다 늦게 기록된 시스템 지연 가능성이 있다.
- 비중이 작고, `approved_days`와 `dispatch_days`는 둘 다 `order_purchase_timestamp` 기준으로 계산되어 승인-출고 선후관계가 직접 계산식을 깨뜨리지는 않는다.
- 따라서 완전한 오류로 단정해 삭제하기보다 데이터셋 특이 패턴으로 남겼다.

추가 제거:
- 병합 및 파생 변수 생성 후 `dispatch_days < 0`인 2행이 확인되어 최종 데이터에서 제거했다.
- `dispatch_days`는 구매 후 택배사 인도까지의 소요일이므로 음수가 되면 사후 물류 피처로 사용할 수 없다.
- 제거 전 109,296행, 제거 후 109,294행이 되었다.

프로젝트 한계:
- 시간 컬럼은 운영 시스템 기록값이므로 실제 물류 이벤트 시각과 100% 일치한다고 보장할 수 없다.
- 일부 승인/출고 순서 역전은 시스템 기록 지연 또는 이벤트 입력 시점 차이일 수 있다.
- 따라서 시간 기반 피처는 강한 예측 신호로 사용할 수 있지만, 개별 주문 단위 원인 해석에서는 주의가 필요하다.

### 4.5 최종 데이터 생성 과정

중간 산출물 흐름:

| 파일                           | 생성 위치              | 의미                                                                         |
| ------------------------------ | ---------------------- | ---------------------------------------------------------------------------- |
| `data/merged_data.csv`       | 1차 병합 직후          | orders, customers, items, reviews, products, sellers를 붙인 원본 통합 데이터 |
| `data/merged_final_data.csv` | 결측/파생/컬럼 정리 후 | EDA와 ML이 공통으로 사용하는 최종 분석 데이터                                |
| `data/merged_train_data.csv` | EDA 노트북             | EDA 결과를 모델링으로 넘기기 위한 학습 후보 데이터                           |
| `data/ml_data.csv`           | ML 노트북              | 모델링 직전 기준 데이터 스냅샷                                               |

`merged_final_data.csv` 최종 컬럼은 33개다.

주요 컬럼 그룹:

- 식별자: `order_id`, `customer_id`, `customer_unique_id`, `order_item_id`, `seller_id`
- 고객/셀러 지역: `customer_city`, `customer_state`, `seller_city`, `seller_state`, `cross_state`
- 주문/가격: `shipping_limit_date`, `price`, `freight_value`
- 리뷰: `review_score`
- 상품: `product_weight_g`, `product_length_cm`, `product_height_cm`, `product_width_cm`, `category`, `main_category`, `sub_category`
- 시간/배송: `order_purchase_dayofweek`, `order_purchase_month`, `approved_days`, `dispatch_days`, `delivery_days`, `expected_delivery_days`, `delay_days`, `delay_days_int`, `is_delayed`, `delay_days_cat`
- 거리: `distance_km`, `distance_cat`

### 4.6 현재 인스턴스 기준의 장단점

장점:

- 상품별 무게/크기, 카테고리, 가격, 배송비, 셀러 정보를 보존할 수 있다.
- 여러 상품을 한 번에 주문한 케이스에서도 어떤 아이템 특성이 리뷰 리스크와 연결되는지 볼 수 있다.
- Track B 사전 예측에서 상품 스펙을 활용할 수 있다.

단점:

- 리뷰는 주문 단위이므로 같은 주문의 리뷰 점수가 여러 아이템 행에 반복된다.
- 주문 단위 성과를 해석할 때 아이템 수가 많은 주문이 더 큰 가중치를 갖는다.
- 모델 평가에서 `order_id` group split을 하지 않으면 data leakage가 생긴다.

현재 대응:

- 모델 입력에서는 `order_id`를 제거했다.
- split과 CV에서는 `order_id`를 group 기준으로 사용했다.
- 따라서 “아이템 단위 정보는 살리되, 같은 주문이 train/test에 동시에 들어가는 문제”를 방지했다.

향후 선택 가능한 대안:

- 주문 단위 예측이 목적이면 `order_id`별로 아이템 정보를 집계해 order-level dataset을 별도로 만들 수 있다.
- 예: 총 상품 수, 총 상품 가격, 총 배송비, 최대 무게, 총 부피, 대표 카테고리, 셀러 수, cross-state 여부 등을 주문 1행으로 집계한다.
- 다만 이 경우 상품/셀러 단위의 세밀한 해석력은 줄어든다.

## 5. Track A와 Track B의 차이

| 구분                | Track A                                               | Track B                                     |
| ------------------- | ----------------------------------------------------- | ------------------------------------------- |
| 목적                | 사후 원인분석 모델                                    | 사전 예측 모델                              |
| 사용 시점           | 배송 완료 후                                          | 주문 확정 또는 배송 시작 전                 |
| 주요 활용           | 부정 리뷰 발생 원인 진단, CS 우선순위, 보상/환불 판단 | 고위험 주문 선별, 선제 안내, 배송 우선 처리 |
| 배송 결과 컬럼 사용 | 사용 가능                                             | 사용 불가                                   |
| 성능 기대치         | 더 높아야 자연스러움                                  | Track A보다 낮아도 실무 가치 있음           |
| 해석 방식           | “왜 부정 리뷰가 발생했는가”                         | “배송 전에 어떤 주문이 위험한가”          |

핵심 판단 기준:

- 배송 완료 후에만 알 수 있는 컬럼을 주문 시점 예측 모델에 넣으면 미래 정보를 미리 보는 leakage가 된다.
- 따라서 Track B는 실제 운영 시점에 이미 알 수 있는 정보만 사용해야 한다.
- Track A는 사후 분석 목적이므로 배송 결과 컬럼을 사용하는 것이 문제되지 않는다.

## 6. Track A 사용 컬럼

Track A는 배송 완료 후 사후 원인분석 모델이다. 따라서 배송 결과 관련 컬럼까지 포함한다.

Track A에서 사용하는 주요 컬럼:

| 컬럼 그룹          | 컬럼                                                                                                                                    |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------- |
| 주문/가격          | `order_item_id`, `price`, `freight_value`, `freight_ratio`, `total_price`                                                     |
| 상품 스펙          | `product_weight_g`, `product_length_cm`, `product_height_cm`, `product_width_cm`                                                |
| 주문 시점          | `order_purchase_dayofweek`, `order_purchase_month`                                                                                  |
| 배송 프로세스      | `approved_days`, `dispatch_days`, `delivery_days`, `expected_delivery_days`, `delay_days`, `is_delayed`, `delay_days_cat` |
| 배송 파생          | `delivery_speed`, `day_per_km`, `delivery_ratio`, `delivery_distance`, `delivery_price`                                       |
| 카테고리           | `main_category`, `sub_category`                                                                                                     |
| 거리/지역          | `distance_km`, `distance_cat`, `cross_state`                                                                                      |
| 상파울루 물류/지역 | `is_sp_customer`, `is_sp_seller`, `sp_route_type`, `sp_route_type_customer`, `sp_route_type_seller`                           |

Track A에서 제외한 컬럼:

| 제외 컬럼                             | 제외 이유                                                     |
| ------------------------------------- | ------------------------------------------------------------- |
| `review_score`                      | 타깃 변수이므로 입력 피처에서 제외                            |
| `order_id`                          | 모델 입력에서는 제외하되 split group 기준으로만 사용          |
| `customer_unique_id`                | 고객 식별자라 모델이 특정 고객을 외우는 방향으로 학습할 위험  |
| `customer_city`, `customer_state` | 원본 지역 문자열은 고카디널리티 및 파생 지역 변수와 중복 가능 |
| `seller_city`, `seller_state`     | 원본 지역 문자열은 파생 변수와 중복 가능                      |
| `shipping_limit_date`               | 날짜 원본 문자열이며 모델 입력용으로 직접 사용하기 부적합     |

상품 무게/크기 컬럼에 대한 판단:

- Track A에서는 이미 전체 피처 세트 안에 포함되어 사용 중이다.
- 무게와 크기는 배송 난이도, 배송비, 취급 난이도, 파손 가능성 등을 설명할 수 있다.
- 사후 원인분석에서는 배송 결과와 함께 상품 스펙을 같이 보는 것이 자연스럽다.

## 7. Track B 사용 컬럼

Track B는 주문 시점에 부정 리뷰 위험을 사전 탐지하는 모델이다. 따라서 배송이 진행되거나 완료된 뒤에야 확정되는 컬럼은 제외한다.

Track B 최종 사용 컬럼 22개:

| 컬럼 그룹          | 컬럼                                                                                                          |
| ------------------ | ------------------------------------------------------------------------------------------------------------- |
| 주문/가격          | `order_item_id`, `price`, `freight_value`, `freight_ratio`, `total_price`                           |
| 상품 스펙          | `product_weight_g`, `product_length_cm`, `product_height_cm`, `product_width_cm`                      |
| 주문 시점          | `order_purchase_dayofweek`, `order_purchase_month`                                                        |
| 예상 배송          | `expected_delivery_days`                                                                                    |
| 카테고리           | `main_category`, `sub_category`                                                                           |
| 거리/지역          | `distance_km`, `distance_cat`, `cross_state`                                                            |
| 상파울루 물류/지역 | `is_sp_customer`, `is_sp_seller`, `sp_route_type`, `sp_route_type_customer`, `sp_route_type_seller` |

Track B에서 제외한 컬럼:

| 제외 컬럼             | 제외 이유                                            |
| --------------------- | ---------------------------------------------------- |
| `approved_days`     | 주문 승인 완료 후 확정되는 운영 결과                 |
| `dispatch_days`     | 발송 처리 후 확정되는 운영 결과                      |
| `delivery_days`     | 배송 완료 후 확정되는 결과                           |
| `delay_days`        | 실제 배송 완료일과 예상 배송일 비교 후 확정되는 결과 |
| `is_delayed`        | 실제 지연 여부이므로 배송 완료 후 정보               |
| `delay_days_cat`    | `delay_days` 기반 파생 컬럼                        |
| `delivery_speed`    | `delivery_days` 기반 파생 컬럼                     |
| `day_per_km`        | `delivery_days` 기반 파생 컬럼                     |
| `delivery_ratio`    | `delivery_days` 기반 파생 컬럼                     |
| `delivery_distance` | `delivery_days` 기반 파생 컬럼                     |
| `delivery_price`    | `delivery_days` 기반 파생 컬럼                     |

## 8. 기존에 제외했던 상품 무게/크기 컬럼을 Track B에 다시 포함한 이유

Track B에서 새로 포함한 상품 스펙 컬럼:

- `product_weight_g`
- `product_length_cm`
- `product_height_cm`
- `product_width_cm`

기존에는 사전 예측 모델의 피처를 보수적으로 줄이는 과정에서 상품 무게/크기 컬럼도 제외했었다. 하지만 재검토 후 포함하는 것이 더 적합하다고 판단했다.

판단 기준:

- 이 컬럼들은 상품 카탈로그에 있는 정보라 주문 시점에 이미 알 수 있다.
- 배송 완료 후 생성되는 결과 정보가 아니므로 data leakage가 아니다.
- 무겁거나 큰 상품은 배송비, 배송 난이도, 파손 가능성, 배송 지연 가능성과 연결될 수 있다.
- Track B의 목적은 완벽한 원인 설명이 아니라 사전 리스크 신호를 최대한 확보하는 것이다.
- 따라서 사용할 수 있는 사전 정보라면 포함한 뒤 모델 성능과 중요도를 확인하는 방식이 더 실무적이다.

포트폴리오에서 설명할 포인트:

- “사전 예측 모델에서는 미래 정보는 제외했지만, 주문 시점에 이미 알 수 있는 상품 스펙은 포함했습니다.”
- “피처 제외 기준은 컬럼의 종류가 아니라 운영 시점에서 해당 정보를 알 수 있는지 여부로 판단했습니다.”

## 9. Data Leakage 방지를 위한 split 기준

기존에는 행 단위로 train, valid, test를 나누면 같은 `order_id`를 공유하는 여러 상품 행이 서로 다른 split에 들어갈 수 있었다.

문제:

- Olist 데이터는 하나의 주문에 여러 `order_item_id`가 연결될 수 있다.
- 같은 주문의 일부 행이 train에 있고 다른 행이 test에 있으면, 모델이 사실상 동일 주문 정보를 학습한 뒤 평가받게 된다.
- 이 경우 test 성능이 실제보다 과대평가될 수 있다.

수정 기준:

- `order_id`를 모델 입력 피처로 쓰지는 않는다.
- 대신 split과 CV의 group 기준으로 사용한다.

적용 방식:

| 단계                   | 적용 방법                                    | 목적                                                                                   |
| ---------------------- | -------------------------------------------- | -------------------------------------------------------------------------------------- |
| train/valid/test 분리  | `GroupShuffleSplit`                        | 같은`order_id`가 서로 다른 split에 나뉘지 않도록 방지                                |
| 하이퍼파라미터 탐색 CV | `StratifiedGroupKFold`                     | 클래스 비율을 최대한 유지하면서 같은`order_id`가 서로 다른 fold에 나뉘지 않도록 방지 |
| 검증                   | split별 row 수, order 수, positive rate 확인 | 그룹 분할 후 클래스 비율이 크게 깨지지 않았는지 확인                                   |

최종 split 결과:

| split |   rows | orders | positive_rate |
| ----- | -----: | -----: | ------------: |
| train | 69,914 | 61,301 |         0.769 |
| valid | 17,568 | 15,326 |         0.767 |
| test  | 21,812 | 19,157 |         0.767 |

판단 기준:

- 그룹 누수를 막는 것이 stratify보다 우선이다.
- 다만 클래스 비율이 크게 흔들리면 모델 비교가 불안정해질 수 있으므로 positive rate를 별도로 확인했다.
- 세 split의 positive rate가 거의 동일하므로 현재 split은 적합하다고 판단했다.

## 10. 전처리 및 피처 엔지니어링 수정 사항

### 10.1 `safe_divide` 추가

추가 함수:

```python
def safe_divide(numerator, denominator):
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator
```

기능:

- 분자 `numerator`와 분모 `denominator`를 받아 나눗셈 결과를 반환한다.
- 분모가 0인 경우 무한대 값이 생기지 않도록 `NaN`으로 바꾼 뒤 계산한다.

왜 필요한가:

- `distance_km`, `delivery_days`, `expected_delivery_days`, `price`가 분모로 쓰이는 파생 변수에서 0 나눗셈 위험이 있다.
- 무한대 값은 모델 학습과 스케일링 과정에서 오류 또는 왜곡을 만들 수 있다.
- 이후 `SimpleImputer(strategy='median')`가 train 기준 중앙값으로 결측을 처리하므로 파이프라인 안에서 일관되게 처리된다.

### 10.2 파생 컬럼 재정의

Track A에서만 사용하는 사후 물류 파생 컬럼:

- `delivery_speed = distance_km / delivery_days`
- `day_per_km = delivery_days / distance_km`
- `delivery_ratio = delivery_days / expected_delivery_days`
- `delivery_distance = delivery_days * distance_km`
- `delivery_price = delivery_days * price`

Track A와 Track B 모두 사용할 수 있는 주문 시점 파생 컬럼:

- `freight_ratio = freight_value / price`
- `total_price = price + freight_value`

판단 기준:

- 파생 컬럼 자체가 문제가 아니라, 그 컬럼을 만드는 데 사용한 원천 정보가 언제 확정되는지가 중요하다.
- `delivery_days` 기반 파생 컬럼은 배송 완료 후 정보이므로 Track B에서는 제외한다.
- `price`, `freight_value` 기반 파생 컬럼은 주문 시점에 알 수 있으므로 Track B에서도 사용한다.

### 10.3 OneHotEncoder sparse 유지

수정 내용:

- `OneHotEncoder(handle_unknown='ignore', sparse_output=True)` 사용

판단 기준:

- 범주형 컬럼을 원핫 인코딩하면 feature 수가 늘어난다.
- dense matrix로 변환하면 메모리와 실행 시간이 크게 증가한다.
- scikit-learn, LightGBM, RandomForest, XGBoost 파이프라인에서 sparse 입력을 처리할 수 있으므로 sparse를 유지하는 것이 더 효율적이다.

## 11. 모델 탐색 및 선택 기준

### 11.1 탐색 대상

Track A에서 비교한 모델:

- LightGBM
- RandomForest
- XGBoost

Track B에서 사용한 모델:

- LightGBM

Track B에서 LightGBM만 사용한 이유:

- Track A에서 모델 계열 비교를 통해 LightGBM이 검증셋 기준 핵심 지표에서 가장 적합하다고 판단했다.
- Track B는 컬럼 사용 시점 제한에 따른 성능 변화를 보는 것이 핵심이므로, 모델 계열을 다시 넓게 비교하기보다 LightGBM의 하이퍼파라미터를 재탐색했다.

### 11.2 평가 지표

| 지표              | 사용 이유                                                        |
| ----------------- | ---------------------------------------------------------------- |
| accuracy          | 전체 정답률 확인용. 단, 클래스 불균형에서는 단독 기준으로 부적합 |
| balanced_accuracy | 긍정/부정 클래스 성능을 균형 있게 반영                           |
| negative recall   | 부정 리뷰를 얼마나 놓치지 않는지 확인                            |
| macro_f1          | 클래스 불균형 상황에서 양쪽 클래스 성능을 함께 확인              |

최종 판단에서는 accuracy보다 `negative recall`, `balanced_accuracy`, `macro_f1`을 더 중요하게 봤다.

판단 기준:

- 데이터는 긍정 리뷰 비율이 약 77%로 불균형하다.
- 모든 주문을 긍정으로만 예측해도 accuracy는 높게 나온다.
- 하지만 이 경우 부정 리뷰 탐지 능력은 0이므로 실무적으로 무가치하다.

### 11.3 threshold 조정

최종 threshold:

| 모델             | threshold |
| ---------------- | --------: |
| Track A LightGBM |      0.57 |
| Track B LightGBM |      0.53 |

판단 기준:

- 기본 threshold 0.5를 고정하면 프로젝트 목표인 부정 리뷰 탐지에 최적이 아닐 수 있다.
- valid set에서 threshold를 조정해 negative recall과 balanced accuracy의 균형을 확인했다.
- test set은 최종 확인용으로만 사용했다.

## 12. 전체 재탐색 결과

Track A 모델 비교 결과:

| model        | cv_best_macro_f1 | threshold | neg_recall | balanced_acc | macro_f1 | accuracy |
| ------------ | ---------------: | --------: | ---------: | -----------: | -------: | -------: |
| LightGBM     |            0.646 |      0.57 |      0.625 |        0.636 |    0.592 |    0.642 |
| RandomForest |            0.648 |      0.57 |      0.623 |        0.633 |    0.588 |    0.638 |
| XGBoost      |            0.616 |      0.82 |      0.614 |        0.620 |    0.575 |    0.623 |

최종 선택:

- Track A 최종 모델은 LightGBM으로 유지했다.

판단 기준:

- RandomForest의 CV macro F1이 아주 근소하게 높았다.
- 하지만 validation 기준 `negative recall`, `balanced_accuracy`, `macro_f1`은 LightGBM이 더 좋았다.
- 이 프로젝트는 부정 리뷰 탐지가 핵심이므로 validation 성능과 목적 지표를 더 우선했다.

## 13. Track A vs Track B 최종 test 성능

| track                           | n_features | accuracy | balanced_acc | neg_recall | macro_f1 |
| ------------------------------- | ---------: | -------: | -----------: | ---------: | -------: |
| Track A LightGBM, 사후 원인분석 |         33 |    0.652 |        0.651 |      0.648 |    0.603 |
| Track B LightGBM, 사전 예측     |         22 |    0.600 |        0.599 |      0.597 |    0.553 |
| Baseline, 다수 클래스           |          0 |    0.767 |        0.500 |      0.000 |    0.434 |

해석:

- Baseline은 accuracy가 가장 높지만 부정 리뷰를 하나도 잡지 못한다.
- Track B는 accuracy는 baseline보다 낮지만 `negative recall`, `balanced_accuracy`, `macro_f1`이 모두 baseline보다 높다.
- Track B는 배송 전 정보만으로도 부정 리뷰 위험의 약 60%를 탐지한다.
- Track A가 Track B보다 좋은 것은 자연스럽다. Track A는 배송 결과라는 강한 정보를 사용할 수 있기 때문이다.
- 두 모델의 성능 차이는 “사전 예측은 사후 설명보다 어렵다”는 점을 정량적으로 보여준다.

## 14. CSV 파일 생성 이유와 각 파일의 의미

### 14.1 `data/product_category_name_translation.csv`

의미:

- Olist 상품 카테고리의 포르투갈어명을 영어명으로 매핑하는 참조 테이블이다.

생성/확보 이유:

- 카테고리명을 사람이 이해하기 쉬운 값으로 바꾸기 위해 필요하다.
- `preprocessed_final.ipynb` 실행에 필요한 외부 참조 데이터였으므로 프로젝트 안의 `data` 폴더에 확보했다.

### 14.2 `data/merged_final_data.csv`

의미:

- 원천 Olist CSV들을 병합하고 기본 전처리를 마친 최종 통합 데이터다.

생성 이유:

- 매번 여러 원천 파일을 다시 조인하면 재현성과 실행 시간이 떨어진다.
- EDA와 모델링이 동일한 출발 데이터를 사용하도록 고정된 중간 산출물이 필요하다.

### 14.3 `data/merged_train_data.csv`

의미:

- EDA 이후 모델링에 사용할 수 있도록 정리된 학습 후보 데이터다.

생성 이유:

- EDA에서 확인한 분석 기준과 파생 컬럼을 모델링 단계로 전달하기 위해 필요하다.
- `ml_classifier.ipynb`가 EDA 결과를 안정적으로 이어받을 수 있게 한다.

### 14.4 `data/ml_data.csv`

의미:

- `ml_classifier.ipynb`에서 모델링 직전 기준으로 저장한 분류 모델 입력용 데이터다.

생성 이유:

- 모델링 단계의 원본 입력 상태를 스냅샷으로 남겨 재실행과 디버깅을 쉽게 하기 위해 생성했다.
- 이후 feature engineering과 Track A/B 분기를 적용하기 전 기준 데이터를 확인할 수 있다.

### 14.5 `outputs/model_comparison_results.csv`

의미:

- Track A에서 LightGBM, RandomForest, XGBoost를 같은 기준으로 재탐색하고 비교한 결과다.

포함 내용:

- 모델명
- best parameters
- CV scoring 기준
- CV best macro F1
- 선택 threshold
- validation 성능 지표
- group-aware CV 적용 여부

생성 이유:

- 노트북 출력만으로는 모델 선택 근거가 휘발된다.
- 모델 선택 기준을 CSV로 남기면 발표자료, 보고서, 재검증에 바로 사용할 수 있다.

### 14.6 `outputs/track_a_vs_b_comparison.csv`

의미:

- Track A, Track B, baseline의 최종 test 성능을 비교한 결과다.

생성 이유:

- 두 모델의 목적 차이와 성능 차이를 정량적으로 보여주기 위해 필요하다.
- “사후 원인분석 모델과 사전 예측 모델은 비교 목적이 다르다”는 점을 숫자로 설명할 수 있다.

### 14.7 `outputs/refreshed_search_summary.json`

의미:

- 전체 재탐색 후 최종 선택된 모델과 threshold, Track B best parameter를 요약한 파일이다.

생성 이유:

- 노트북 재실행 없이도 최종 모델 설정을 빠르게 확인하기 위해 남겼다.
- 모델 배포나 Streamlit 연결 단계에서 참조하기 쉽다.

## 15. 노트북별 주요 수정 사항

### 15.1 `preprocessed_final.ipynb`

수정 및 확인 내용:

- 필요한 원천 데이터 로드 흐름 확인
- `product_category_name_translation.csv` 확보 후 전처리 실행 가능하도록 정리
- `data/merged_final_data.csv` 생성 확인

판단 기준:

- 뒤 단계 노트북이 전처리 산출물을 전제로 실행되므로, 중간 CSV 생성 여부가 전체 파이프라인 재현성의 핵심이다.

### 15.2 `eda_final.ipynb`

수정 내용:

- `merged_final_data.csv`가 없을 때 `preprocessed_final.ipynb`를 먼저 실행해야 한다는 안내 추가
- train/test 분할에서 `stratify=y` 적용
- `delay_days` 부호와 해석 설명 수정
- 인과처럼 보일 수 있는 표현을 완화
- 리뷰 점수 차이에 대한 과도한 배수 표현 제거
- `data/merged_train_data.csv` 생성 확인

판단 기준:

- EDA는 모델링보다 앞 단계이므로 데이터 정의와 해석 문구가 정확해야 한다.
- 상관관계나 그룹 차이를 인과로 표현하면 분석 신뢰도가 떨어진다.
- 클래스 불균형이 있는 타깃은 stratify로 분포를 유지해야 평가가 안정적이다.

### 15.3 `ml_classifier.ipynb`

수정 내용:

- `lightgbm` 의존성 설치 및 `requirements.txt` 추가
- `nbclient`, `nbformat`을 `requirements.txt`에 추가해 노트북 실행 검증 가능하도록 정리
- `safe_divide` 추가
- `data/ml_data.csv` 저장 시 `index=False` 적용
- 타깃 설명을 “CS 리스크 기준”으로 수정
- 행 단위 split을 `order_id` 기준 group split으로 변경
- 하이퍼파라미터 탐색 CV를 `StratifiedGroupKFold`로 변경
- Track A 3개 모델 전체 재탐색
- Track B LightGBM 하이퍼파라미터 재탐색
- Track B에 상품 무게/크기 컬럼 추가
- `OneHotEncoder`를 sparse 출력으로 유지
- 최신 모델 결과 CSV와 해석 표 갱신
- 전체 노트북 실행 검증 완료

판단 기준:

- 모델 성능보다 먼저 data leakage 방지가 우선이다.
- 모델 선택은 accuracy가 아니라 부정 리뷰 탐지 목적에 맞는 지표를 중심으로 해야 한다.
- 사전 예측 모델은 “컬럼을 언제 알 수 있는가”를 기준으로 피처를 선택해야 한다.
- 산출 CSV는 결과 재현성과 보고서 작성 효율을 위해 필요하다.

## 16. 현재 최종 결론

Track A는 배송 완료 후의 사후 원인분석 모델로 유지한다. 배송 지연, 실제 배송일, 배송 속도 등 강한 사후 신호를 사용하므로 성능이 가장 높고, CS 우선순위 판단이나 불만 원인 진단에 적합하다.

Track B는 주문 시점의 사전 예측 모델로 유지한다. 배송 완료 후 정보는 제외하되, 상품 무게/크기처럼 주문 시점에 이미 알 수 있는 정보는 포함한다. 성능은 Track A보다 낮지만 baseline보다 실무적으로 의미 있는 개선을 보인다.

최종 운영 관점에서는 두 모델을 함께 쓰는 구조가 가장 설득력 있다.

1. Track B로 주문 확정 시점에 고위험 주문을 선별한다.
2. 배송이 완료된 뒤 Track A로 실제 불만 가능성과 원인을 진단한다.
3. 두 결과를 함께 사용해 선제 안내, 배송 우선순위, CS 보상 정책을 설계한다.

면접 및 포트폴리오에서 강조할 핵심은 다음이다.

- 단순히 모델 성능을 올린 것이 아니라, 실제 운영 시점에 맞게 피처 사용 가능성을 재정의했다.
- `order_id` 기준 group split으로 같은 주문이 train과 test에 동시에 들어가는 leakage를 막았다.
- 불균형 데이터에서 accuracy의 한계를 인식하고 `negative recall`, `balanced_accuracy`, `macro_f1`을 중심으로 모델을 선택했다.
- Track A와 Track B를 분리해 “사후 설명”과 “사전 예측”이라는 서로 다른 비즈니스 문제를 명확히 구분했다.
