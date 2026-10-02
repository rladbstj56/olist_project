# 모델링 의사결정 및 수정 기록

작성일: 2026-07-27
대상 파일: `preprocessed_final.ipynb`, `eda_final.ipynb`, `ml_classifier.ipynb`

## 1. 문서 목적

이 문서는 지금까지 EDA, 전처리, 분류 모델링 과정에서 결정한 내용과 수정된 사항을 기록하기 위한 문서다. 단순히 어떤 코드를 바꿨는지가 아니라, 왜 그렇게 수정했는지의 판단 기준을 함께 남긴다.

분석의 핵심 목적은 Olist 주문 데이터를 기반으로 리뷰 점수가 낮아질 위험을 탐지하는 것이다. 다만 모델을 언제 사용할 것인지에 따라 사용할 수 있는 컬럼이 달라지므로, 모델을 Track A와 Track B로 분리했다.

## 2. 전체 분석 흐름

1. `preprocessed_final.ipynb`

   - 원천 CSV들을 병합하고 분석용 기본 데이터셋을 생성한다.
   - 최종 산출물은 `data/processed/merged_final_data.csv`다.
2. `eda_final.ipynb`

   - 전처리된 데이터를 기반으로 탐색적 분석을 수행한다.
   - 모델링에 사용할 학습 기반 데이터셋 `data/processed/merged_train_data.csv`를 생성한다.
3. `ml_classifier.ipynb`

   - 리뷰 점수를 긍정/부정으로 이진 분류한다.
   - Track A와 Track B를 분리해 각각 모델링한다.
   - 모델 비교 결과와 최종 성능 요약 CSV를 생성한다.

## 3. 타깃 정의

2026년 9월 지티씨솔루션 면접 복기 이후, 리뷰 점수 `review_score`의 이진화 기준을 재정의했다. 기존의 “1점부터 3점까지를 모두 부정 리뷰로 본다”는 설명은 주 모델 기준에서 내리고, 3점 리뷰는 중립으로 제외한 뒤 민감도 분석에서 비교하는 방식으로 보완했다.

주 모델 기준:

| 원본 리뷰 점수 | 모델 타깃 | 의미 |
| -------------- | --------: | ---- |
| 1점, 2점 | 0 | 낮은 만족도 리뷰, CS 우선 확인 대상 |
| 3점 | 제외 | 중립 또는 판단 유보 구간 |
| 4점, 5점 | 1 | 긍정 리뷰 |

판단 기준:

- Olist의 `review_score`는 엄밀한 NPS 문항이 아니다.
- NPS처럼 추천 의향을 측정한 값이 아니라 구매 후 만족도에 가까운 별점이다.
- 따라서 NPS 기준을 그대로 적용해 1점부터 5점까지의 별점을 promoter, passive, detractor처럼 해석하는 것은 적절하지 않다.
- 3점은 의미가 모호하므로 명확한 낮은 만족도와 긍정을 비교하려는 분석 목적에 따라 제외합니다. 모든 3점이 실제로 중립이라는 사실이나 이 정책의 성능 우월성을 입증한 것은 아닙니다.
- 1점·2점은 낮은 만족도가 비교적 명확하므로 CS 우선 확인 대상 라벨로 사용한다.

라벨 정책별 비교 (주문·아이템 행 기준):

정책마다 평가 대상과 클래스 비율이 달라 지표만으로 우열을 판정하지 않습니다. 0.597은 1점부터 3점까지를 리스크로 묶은 민감도 결과이며 현재 주 모델 재현율 0.619와 구분합니다.

| 라벨 정책 | 모델링 행 수 | 제외 행 수 | 낮은 만족도 비율 | Track B Balanced Accuracy | Track B Risk Precision | Track B Risk Recall | Track B PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1점·2점 낮은 만족도, 4점·5점 긍정, 3점 제외 | 100,107 | 9,187 | 16.1% | 0.622 | 0.235 | 0.619 | 0.289 |
| 1점·2점 낮은 만족도, 3점부터 5점까지 비위험 | 109,294 | 0 | 14.8% | 0.623 | 0.223 | 0.626 | 0.280 |
| 1점부터 3점까지 CS 리스크, 4점·5점 긍정 | 109,294 | 0 | 23.2% | 0.599 | 0.312 | 0.597 | 0.356 |

포트폴리오 표현:

- “처음에는 1점부터 3점까지를 CS 리스크로 묶었지만, 3점의 의미가 애매하다는 한계를 인정하고 주 모델에서는 3점을 중립으로 제외했습니다.”
- “대신 3점을 제외하거나, 비위험으로 포함하거나, 리스크로 포함하는 세 가지 라벨 정책을 비교해 기준 선택이 성능과 운영 대상 규모에 어떤 영향을 주는지 확인했습니다.”
- “이 프로젝트의 목적은 별점 자체를 정확히 맞히는 것이 아니라 낮은 만족도 리뷰를 조기에 탐지해 운영자가 먼저 확인할 주문을 좁히는 것입니다.”

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
| `data/processed/merged_data.csv`       | 1차 병합 직후          | orders, customers, items, reviews, products, sellers를 붙인 원본 통합 데이터 |
| `data/processed/merged_final_data.csv` | 결측/파생/컬럼 정리 후 | EDA와 ML이 공통으로 사용하는 최종 분석 데이터                                |
| `data/processed/merged_train_data.csv` | EDA 노트북             | EDA 결과를 모델링으로 넘기기 위한 학습 후보 데이터                           |
| `data/processed/ml_data.csv`           | ML 노트북              | 모델링 직전 기준 데이터 스냅샷                                               |

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
| 주요 활용           | 낮은 만족도 리뷰 발생 원인 진단, CS 우선순위, 보상/환불 판단 | 고위험 주문 선별, 선제 안내, 배송 우선 처리 |
| 배송 결과 컬럼 사용 | 사용 가능                                             | 사용 불가                                   |
| 성능 기대치         | 더 높아야 자연스러움                                  | Track A보다 낮아도 실무 가치 있음           |
| 해석 방식           | “왜 낮은 만족도 리뷰가 발생했는가”                         | “배송 전에 어떤 주문이 위험한가”          |

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

Track B는 주문 시점에 낮은 만족도 리뷰 위험을 사전 탐지하는 모델이다. 따라서 배송이 진행되거나 완료된 뒤에야 확정되는 컬럼은 제외한다.

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

| 지표 | 사용 이유 |
| ---- | --------- |
| accuracy | 전체 정답률 확인용. 단, 클래스 불균형에서는 단독 기준으로 부적합 |
| balanced_accuracy | 긍정/낮은 만족도 클래스 성능을 균형 있게 반영 |
| risk precision | 위험으로 올린 주문 중 실제 낮은 만족도 리뷰가 얼마나 되는지 확인 |
| risk recall | 실제 낮은 만족도 리뷰를 얼마나 놓치지 않는지 확인 |
| risk PR-AUC | 낮은 만족도 클래스가 소수일 때 ranking 품질을 확인 |
| flagged rate | 운영자가 실제 확인해야 할 주문 비율을 확인 |
| macro_f1 | 클래스 불균형 상황에서 양쪽 클래스 성능을 함께 확인 |

최종 판단에서는 accuracy보다 `risk_recall`, `risk_precision`, `risk_pr_auc`, `flagged_rate`, `balanced_accuracy`를 더 중요하게 봤다.

판단 기준:

- 주 라벨 정책 기준 데이터는 긍정 리뷰 비율이 약 84%, 낮은 만족도 리뷰 비율이 약 16%로 불균형하다.
- 모든 주문을 긍정으로만 예측해도 accuracy는 높게 나온다.
- 하지만 이 경우 낮은 만족도 리뷰 탐지 능력은 0이므로 실무적으로 무가치하다.

### 11.3 threshold 조정

최종 threshold:

| 모델             | threshold |
| ---------------- | --------: |
| Track A LightGBM |      0.57 |
| Track B LightGBM |      0.54 |

판단 기준:

- 기본 threshold 0.5를 고정하면 프로젝트 목표인 낮은 만족도 리뷰 탐지에 최적이 아닐 수 있다.
- valid set에서 threshold를 조정해 risk recall과 balanced accuracy의 균형을 확인했다.
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
- 하지만 validation 기준 `risk_recall`, `balanced_accuracy`, `macro_f1`은 LightGBM이 더 좋았다.
- 이 프로젝트는 낮은 만족도 리뷰 탐지가 핵심이므로 validation 성능과 목적 지표를 더 우선했다.

## 13. Track A vs Track B 최종 test 성능

| track | n_features | accuracy | balanced_acc | risk_precision | risk_recall | risk_pr_auc | flagged_rate | macro_f1 |
| ----- | ---------: | -------: | -----------: | -------------: | ----------: | ----------: | -----------: | -------: |
| Track A LightGBM, 사후 원인분석 | 33 | 0.686 | 0.683 | 0.287 | 0.678 | 0.505 | 0.370 | 0.595 |
| Track B LightGBM, 사전 예측 | 22 | 0.624 | 0.622 | 0.235 | 0.619 | 0.289 | 0.413 | 0.539 |
| Baseline, 다수 클래스 | 0 | 0.843 | 0.500 | 0.000 | 0.000 | 0.157 | 0.000 | 0.458 |

해석:

- Baseline은 accuracy가 가장 높지만 낮은 만족도 리뷰를 하나도 잡지 못한다.
- Track B는 accuracy는 baseline보다 낮지만 `risk_recall`, `balanced_accuracy`, `risk_pr_auc`가 baseline보다 높다.
- Track B는 배송 전 정보만으로도 낮은 만족도 리뷰 위험의 약 62%를 탐지한다.
- Track B의 `flagged_rate`는 0.413이므로 테스트 주문 중 약 41%를 우선 확인 대상으로 올리는 운영 기준이다.
- Track B의 `risk_precision`은 0.235로 높지 않으므로, 이 모델은 자동 차단이나 자동 보상 모델이 아니라 CS·물류 담당자가 먼저 볼 후보를 좁히는 우선순위 모델로 해석한다.
- Track A가 Track B보다 좋은 것은 자연스럽다. Track A는 배송 결과라는 강한 정보를 사용할 수 있기 때문이다.
- 두 모델의 성능 차이는 “사전 예측은 사후 설명보다 어렵다”는 점을 정량적으로 보여준다.

## 14. CSV 파일 생성 이유와 각 파일의 의미

### 14.1 `data/raw/product_category_name_translation.csv`

의미:

- Olist 상품 카테고리의 포르투갈어명을 영어명으로 매핑하는 참조 테이블이다.

생성/확보 이유:

- 카테고리명을 사람이 이해하기 쉬운 값으로 바꾸기 위해 필요하다.
- `preprocessed_final.ipynb` 실행에 필요한 외부 참조 데이터였으므로 프로젝트 안의 `data` 폴더에 확보했다.

### 14.2 `data/processed/merged_final_data.csv`

의미:

- 원천 Olist CSV들을 병합하고 기본 전처리를 마친 최종 통합 데이터다.

생성 이유:

- 매번 여러 원천 파일을 다시 조인하면 재현성과 실행 시간이 떨어진다.
- EDA와 모델링이 동일한 출발 데이터를 사용하도록 고정된 중간 산출물이 필요하다.

### 14.3 `data/processed/merged_train_data.csv`

의미:

- EDA 이후 모델링에 사용할 수 있도록 정리된 학습 후보 데이터다.

생성 이유:

- EDA에서 확인한 분석 기준과 파생 컬럼을 모델링 단계로 전달하기 위해 필요하다.
- `ml_classifier.ipynb`가 EDA 결과를 안정적으로 이어받을 수 있게 한다.

### 14.4 `data/processed/ml_data.csv`

의미:

- `ml_classifier.ipynb`에서 모델링 직전 기준으로 저장한 분류 모델 입력용 데이터다.

생성 이유:

- 모델링 단계의 원본 입력 상태를 스냅샷으로 남겨 재실행과 디버깅을 쉽게 하기 위해 생성했다.
- 이후 feature engineering과 Track A/B 분기를 적용하기 전 기준 데이터를 확인할 수 있다.

### 14.5 `outputs/tables/model_comparison_results.csv`

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

### 14.6 `outputs/tables/track_a_vs_b_comparison.csv`

의미:

- Track A, Track B, baseline의 최종 test 성능을 비교한 결과다.

생성 이유:

- 두 모델의 목적 차이와 성능 차이를 정량적으로 보여주기 위해 필요하다.
- “사후 원인분석 모델과 사전 예측 모델은 비교 목적이 다르다”는 점을 숫자로 설명할 수 있다.

### 14.7 `outputs/tables/label_policy_comparison.csv`

의미:

- 리뷰 점수 3점을 어떻게 처리하는지에 따른 라벨 분포와 Track B 성능 비교 결과다.

생성 이유:

- 3점 리뷰를 부정, 비위험, 중립 제외 중 어떤 기준으로 둘 것인지가 모델의 문제 정의를 바꾼다.
- 면접에서 라벨 기준을 질문받았을 때 단일 기준을 우기는 대신, 기준별 성능과 운영 대상 규모를 비교했다는 근거로 설명할 수 있다.

### 14.8 `outputs/metadata/refreshed_search_summary.json`

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
- `data/processed/merged_final_data.csv` 생성 확인

판단 기준:

- 뒤 단계 노트북이 전처리 산출물을 전제로 실행되므로, 중간 CSV 생성 여부가 전체 파이프라인 재현성의 핵심이다.

### 15.2 `eda_final.ipynb`

수정 내용:

- `merged_final_data.csv`가 없을 때 `preprocessed_final.ipynb`를 먼저 실행해야 한다는 안내 추가
- train/test 분할에서 `stratify=y` 적용
- `delay_days` 부호와 해석 설명 수정
- 인과처럼 보일 수 있는 표현을 완화
- 리뷰 점수 차이에 대한 과도한 배수 표현 제거
- 배송 지연 구간이 1-3일에서 4일 이상으로 넘어갈 때 리뷰 점수가 낮아지는지 Mann-Whitney U 검정과 Rank-biserial correlation으로 추가 검증
- `data/processed/merged_train_data.csv` 생성 확인

판단 기준:

- EDA는 모델링보다 앞 단계이므로 데이터 정의와 해석 문구가 정확해야 한다.
- 상관관계나 그룹 차이를 인과로 표현하면 분석 신뢰도가 떨어진다.
- 클래스 불균형이 있는 타깃은 stratify로 분포를 유지해야 평가가 안정적이다.

#### 통계 검정과 포트폴리오 차트의 집계 단위 판단

이 프로젝트의 최종 분석 데이터는 주문-아이템 단위다. 따라서 같은 `order_id` 안에 여러 상품이 있으면 리뷰 점수 `review_score`가 여러 행에 반복된다. 이 구조에서는 목적에 따라 서로 다른 기준으로 계산될 수 있다.

| 계산 기준 | 의미 | 적합한 사용처 | 주의점 |
| --- | --- | --- | --- |
| 주문-아이템 단위 | 상품·셀러·배송비 등 아이템별 특성을 보존한 행 기준 | 모델링, 상품·셀러 단서 탐색 | 리뷰 점수가 주문 단위라서 아이템 수가 많은 주문의 리뷰가 여러 번 반영됨 |
| 주문 단위 | `order_id` 기준으로 주문 1건당 1행만 남긴 기준 | 리뷰 점수 평균 비교, 통계 검정, 포트폴리오 핵심 수치·차트 | 상품별 세부 정보는 축약되므로 상품·셀러 단서 해석에는 부적합 |
| 고객 단위 | `customer_unique_id` 기준으로 고객별 주문을 집계한 기준 | 고객 반복 구매·장기 만족도 분석 | 이번 프로젝트의 CS 리스크 모델 목적과는 거리가 있음 |
| 판매자·지역·상품군 단위 | 특정 운영 단위별 평균·지연율을 집계한 기준 | 운영 액션 후보 탐색, 모니터링 대상 선정 | 주문 수가 적은 그룹은 평균이 불안정할 수 있음 |

이번 포트폴리오에 제시하는 리뷰 점수 차이와 검정 결과는 주문 단위 기준으로 통일했다. 이는 리뷰가 아이템별로 배정되기 때문이 아니라, 리뷰 점수 `review_score`가 `order_id` 기준 주문 단위로 부여되기 때문이다. 판단 근거는 다음과 같다.

- 결과변수인 `review_score`는 `order_id` 기준 주문 단위로 작성된다.
- 주문-아이템 단위 평균을 사용하면 아이템 수가 많은 주문의 리뷰가 더 큰 가중치를 갖는다.
- "배송 지연 주문은 리뷰 점수가 낮은가", "3일 초과 지연 주문은 별도 리스크 구간인가"라는 주장은 주문 1건을 관측 단위로 보는 것이 더 자연스럽다.
- 포트폴리오 차트와 문구는 같은 기준이어야 하므로, 핵심 결과 차트도 `order_id` 기준 주문 단위로 재생성했다.
- 다만 모델링 데이터는 상품·셀러·배송비 등 아이템별 정보를 활용해야 하므로 주문-아이템 단위를 유지하고, 평가 분할에서 `order_id` 기준 그룹 분리로 누수를 통제했다.

3일 초과 지연 기준 검정 결과:

- 검정 위치: `notebooks/02_eda.ipynb`
- 산출 파일: `outputs/tables/delay_threshold_test_summary.csv`
- 검정 단위: `order_id` 기준 주문 단위. 리뷰 점수는 주문 단위인데 분석 데이터는 주문-아이템 단위이므로, 같은 주문 리뷰가 여러 아이템 행에 반복되는 영향을 줄이기 위해 주문당 1행만 남겼다.
- 비교 그룹: 지연 주문 중 `1-3일 지연` vs `4일 이상 지연`
- 평균 리뷰 점수: `1-3일 지연` 3.75점, `4일 이상 지연` 1.93점
- 평균 차이: 1.83점
- 검정: Mann-Whitney U 검정, p < .001
- 효과크기: Rank-biserial correlation 0.594, 큰 효과

해석:

- 4일 이상 지연 주문은 1-3일 지연 주문보다 리뷰 점수가 통계적으로 유의하게 낮고, 효과크기도 큰 수준이다.
- 따라서 3일 초과 지연을 CS 리스크가 커지는 운영 관리 기준으로 제시할 수 있다.
- 다만 이는 관찰 데이터 기반의 그룹 차이이므로, 배송 지연이 리뷰 하락의 유일한 원인이라고 단정하지 않는다.

### 15.3 `ml_classifier.ipynb`

수정 내용:

- `lightgbm` 의존성 설치 및 `requirements.txt` 추가
- `nbclient`, `nbformat`을 `requirements.txt`에 추가해 노트북 실행 검증 가능하도록 정리
- `safe_divide` 추가
- `data/processed/ml_data.csv` 저장 시 `index=False` 적용
- 타깃 설명을 `1점·2점 = 낮은 만족도`, `4점·5점 = 긍정`, `3점 = 중립 제외` 기준으로 수정
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
- 모델 선택은 accuracy가 아니라 낮은 만족도 리뷰 탐지 목적에 맞는 지표를 중심으로 해야 한다.
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
- 불균형 데이터에서 accuracy의 한계를 인식하고 `risk_recall`, `risk_precision`, `risk_pr_auc`, `flagged_rate`, `balanced_accuracy`를 중심으로 모델을 해석했다.
- Track A와 Track B를 분리해 “사후 설명”과 “사전 예측”이라는 서로 다른 비즈니스 문제를 명확히 구분했다.

## 17. 회귀 모델의 최종 제외 판단

초기에는 `review_score`를 1점부터 5점까지의 연속적인 점수처럼 예측하는 회귀 모델도 검토했다. 그러나 최종 모델에서는 회귀가 아니라 분류 모델을 유지했다.

판단 기준은 다음과 같다.

- `review_score`는 엄밀한 연속형 수치가 아니라 1점부터 5점까지의 순서형 별점이다.
- 비즈니스 목적은 정확히 3.7점, 4.2점처럼 점수를 맞히는 것이 아니라, CS 개입이 필요한 낮은 만족도 리뷰 위험 주문을 조기에 탐지하는 것이다.
- 회귀 모델의 test 성능은 `RMSE 1.1886`, `MAE 0.9170`, `R2 0.2213` 수준이었다. 설명력이 낮아 최종 운영 모델로 제시하기 어렵다.
- 반면 분류 모델은 낮은 만족도 리뷰 recall, balanced accuracy, macro F1처럼 불균형 타겟에 맞는 지표로 성능을 평가할 수 있다.

따라서 최종 실행 모델은 `notebooks/03_ml_classifier.ipynb`의 분류 모델로 유지한다. 회귀 노트북은 실험 기록과 비교 근거로만 보존하며, 현재 위치는 `archive/reference_models/04_ml_regressor.ipynb`다.

## 18. Track B 운영 활용과 Track C 예상 배송일 추천 모델 제안

작성일: 2026-08-15

### 18.1 Track B의 운영상 의미

Track B는 주문 시점에 이미 알 수 있는 정보만 사용해 낮은 만족도 리뷰 위험을 예측하는 모델이다. 현재 `notebooks/03_ml_classifier.ipynb`의 `pre_order_cols` 기준으로 `expected_delivery_days`도 Track B feature에 포함되어 있다.

따라서 Track B는 다음처럼 해석한다.

```text
판매자 또는 플랫폼이 설정한 기존 예상 배송일을 포함해,
해당 주문이 낮은 만족도 리뷰로 이어질 위험이 높은지 주문 직후 판단하는 모델
```

Track B의 목적은 모든 주문의 리뷰를 정확히 맞히는 것이 아니라, 운영자가 먼저 확인해야 할 고위험 주문을 좁히는 것이다.

운영 액션 후보:

- 고위험 주문 우선 모니터링
- 판매자 출고 상태 확인
- 예상 배송일 임박 전 고객 선제 안내
- 배송 우선순위 조정 후보 선정
- CS 알림 또는 보상 후보 등록

이 해석이 중요한 이유는 모델 성능을 단순 accuracy로 평가하지 않고, 제한된 운영 리소스를 어디에 먼저 쓸지 정하는 우선순위 모델로 연결할 수 있기 때문이다.

### 18.2 Track C 확장 필요성

현재 Track B는 낮은 만족도 리뷰 위험을 예측하지만, 위험 주문에 대해 “예상 배송일을 며칠로 제안해야 하는가”를 직접 답하지는 않는다. 이 질문은 별도의 목표 변수를 갖는 Track C로 분리하는 것이 적절하다.

Track C의 역할:

```text
Track B: 어떤 주문이 위험한가?
Track C: 위험 주문에 대해 예상 배송일을 어떻게 조정할 것인가?
```

Track C는 모든 주문에 적용하기보다, Track B에서 위험 확률이 높은 주문에만 추가 실행하는 2단계 운영 구조가 적합하다.

```text
주문 발생
→ 기존 예상 배송일 포함 Track B 실행
→ 고위험 주문 선별
→ 고위험 주문에만 Track C 실행
→ 추천 예상 배송일 산출
→ 특별 조치로 예상 배송일 조정 또는 고객 선제 안내 반영
```

이 구조는 모든 주문의 예상 배송일을 불필요하게 늘리지 않으면서, 리뷰 리스크가 큰 주문에만 더 보수적인 약속을 적용할 수 있다는 장점이 있다.

### 18.3 Track C 모델 설계안

Track C는 현재의 낮은 만족도 리뷰 분류 모델과 목표가 다르므로, `review_score`나 낮은 만족도 리뷰 여부를 직접 타깃으로 두기보다 실제 배송 소요일을 예측하는 모델로 설계하는 것이 자연스럽다.

기본 설계:

| 항목 | 내용 |
| --- | --- |
| 사용 시점 | 주문 직후 또는 배송 시작 전 |
| 적용 대상 | Track B가 고위험으로 분류한 주문 |
| 입력 feature | 주문 시점에 이미 알 수 있는 가격, 배송비, 상품 스펙, 카테고리, 거리, 지역 정보 |
| target | `delivery_days` |
| 출력 | 주문별 예상 실제 배송 소요일 |
| 운영 결과 | 추천 예상 배송일 또는 고객 안내용 보수적 배송 약속 |

추천 산식 후보:

```text
recommended_expected_days = ceil(predicted_delivery_days_90_percentile)
```

90% 분위수를 기본안으로 둔 이유는 Track C의 목적이 평균 배송일 예측이 아니라 고위험 주문의 지연 tail risk를 줄이는 데 있기 때문이다. 평균 예측값은 “보통의 주문”을 설명하는 데는 유용하지만, 낮은 만족도 리뷰로 이어질 가능성이 큰 늦은 배송 케이스를 충분히 보호하지 못할 수 있다.

단, 90%는 통계적으로 고정된 정답 임계값이 아니라 초기 운영 기준이다. 실제 운영에서는 분위수 기준별 trade-off를 비교해야 한다.

| 분위수 기준 | 운영 의미 | 장점 | 주의점 |
| --- | --- | --- | --- |
| 80% | 덜 보수적인 추천일 | 예상 배송일 증가 폭이 작음 | 지연 불만이 남을 수 있음 |
| 90% | 기본 추천안 | 지연 리스크와 고객 기대 관리의 균형 | 일부 주문은 예상일이 길어질 수 있음 |
| 95% | 매우 보수적인 추천일 | 예상일 초과 가능성을 더 낮춤 | 배송 약속이 길어져 구매 전환율에 불리할 수 있음 |

따라서 Track C 구현 시 90% 분위수를 기본값으로 사용하되, 결과표에는 분위수별 `3일 초과 지연율`, `평균 추천 조정일수`, `현재 예상 배송일 대비 증가 폭`을 함께 비교하는 것이 바람직하다.

또는 3일 초과 지연 기준을 직접 반영하는 정책:

```text
recommended_expected_days = ceil(predicted_delivery_days_90_percentile - 3)
```

판단 기준:

- 평균 예측값은 실제보다 늦어지는 주문을 충분히 보호하지 못할 수 있다.
- 고위험 주문은 리뷰 리스크 관리가 목적이므로 평균보다 보수적인 분위수 예측이 더 적합하다.
- 90% 분위수는 확정된 정답이 아니라 초기 운영 기준이며, 최종 기준은 운영 비용과 고객 경험 지표를 함께 보며 조정해야 한다.
- EDA에서 4일 이상 지연 주문의 평균 리뷰 점수가 1–3일 지연 주문보다 유의하게 낮았으므로, 3일 초과 지연을 피하는 방향으로 예상 배송일을 추천하는 것이 프로젝트의 통계 결과와 연결된다.

### 18.4 구현 시 주의사항

Track C를 구현할 때는 다음을 주의한다.

- `delivery_days`는 배송 완료 후에만 알 수 있는 값이므로 Track C의 target으로는 사용할 수 있지만, 입력 feature로 사용하면 안 된다.
- `delay_days`, `is_delayed`, `delay_days_cat`, `delivery_ratio` 등 배송 완료 후 정보 또는 그 파생 변수는 Track C 입력에서 제외한다.
- 주문-아이템 단위 데이터에서는 같은 `order_id`가 여러 행에 반복되므로, 모델 평가 시 `order_id` 기준 group split을 유지한다.
- 예상 배송일 추천이 실제 리뷰 개선으로 이어진다고 단정하지 않는다. 실제 운영 효과는 A/B 테스트 또는 파일럿 운영으로 검증해야 한다.

### 18.5 포트폴리오 반영 문장

Track C가 아직 구현 전이라면 “향후 확장 제안”으로 표현한다.

> 사전예측모델 Track B로 낮은 만족도 리뷰 위험 주문을 선별하고, 고위험 주문에 대해서는 Track C 배송 소요일 예측 모델을 추가 적용해 보수적인 예상 배송일을 추천하는 운영 전략을 제안했다. 특히 4일 이상 지연 구간에서 리뷰 점수가 유의하게 하락한다는 통계검정 결과를 Track C의 추천 기준으로 연결했다.

Track C를 실제 구현한 뒤에는 다음처럼 표현을 강화할 수 있다.

> Track B로 고위험 주문을 선별한 뒤, Track C 분위수 기반 배송 소요일 예측 모델로 주문별 추천 예상 배송일을 산출했다. 이를 통해 낮은 만족도 리뷰 위험 예측을 운영자가 실행 가능한 배송 약속 조정 액션으로 연결했다.

### 18.6 Track C 1차 구현 결과

구현일: 2026-08-15

추가 파일:

- `src/olist_delivery_models.py`
- `scripts/generate_track_c_outputs.py`
- `streamlit_app.py`
- `outputs/tables/track_c_quantile_results.csv`
- `outputs/tables/track_c_recommendation_examples.csv`

구현한 함수와 역할:

| 함수 | 입력 | 출력 | 역할 |
| --- | --- | --- | --- |
| `load_ml_data` | `data/processed/ml_data.csv` 경로 | `DataFrame` | 모델링 직전 데이터를 불러온다. |
| `add_pre_order_features` | 원본 주문-아이템 데이터 | 파생 feature가 추가된 `DataFrame` | Track B와 Track C가 공통으로 쓰는 주문 시점 feature를 만든다. |
| `add_post_delivery_features` | 원본 주문-아이템 데이터 | 사후 배송 feature가 추가된 `DataFrame` | Track A가 사용하는 배송 완료 후 feature를 만든다. |
| `review_risk_target` | `review_score`, 라벨 정책 | `Series` | 라벨 정책에 따라 낮은 만족도 리뷰와 긍정 리뷰를 이진화한다. |
| `prepare_labeled_model_frame` | 원본 데이터, 라벨 정책 | `DataFrame` | 3점 제외 등 라벨 정책을 실제 모델링 데이터에 적용한다. |
| `build_track_a_pipeline` | Track A 입력 feature | scikit-learn `Pipeline` | 사후 원인분석용 LightGBM 분류 파이프라인을 만든다. |
| `build_track_b_pipeline` | Track B 입력 feature | scikit-learn `Pipeline` | 낮은 만족도 리뷰 위험도를 예측하는 LightGBM 분류 파이프라인을 만든다. |
| `build_track_c_pipeline` | Track C 입력 feature, 분위수 기준 | scikit-learn `Pipeline` | `delivery_days`의 분위수 예측을 수행하는 LightGBM 회귀 파이프라인을 만든다. |
| `risk_classification_metrics` | 실제 라벨, 긍정 클래스 확률, threshold | `dict` | risk precision, risk recall, PR-AUC, flagged rate를 계산한다. |
| `evaluate_track_a_vs_b` | 전체 데이터, 라벨 정책 | `DataFrame` | Track A, Track B, baseline을 같은 라벨 정책에서 비교한다. |
| `evaluate_label_policies` | 전체 데이터 | `DataFrame` | 3점 처리 기준별 라벨 분포와 Track B 성능을 비교한다. |
| `predict_order` | 학습된 모델, 주문 1건 feature | 위험도, 추천 예상 배송일 딕셔너리 | UI에서 단일 주문의 Track B/C 결과를 계산한다. |
| `evaluate_track_c_quantiles` | 전체 데이터, 분위수 후보 | 분위수별 성능 비교표 | 80%, 90%, 95% 기준의 지연율과 조정일수를 비교한다. |

Track C 입력 feature는 Track B와 동일한 주문 시점 feature 22개를 사용한다. `delivery_days`는 target으로만 사용하며 입력 feature에는 넣지 않는다.

1차 산출 결과:

| 분위수 기준 | 현재 3일 초과 지연율 | 검토값 적용 시 3일 초과 지연율 | 평균 조정일수 | 조정된 주문·아이템 행 비율 |
| --- | ---: | ---: | ---: | ---: |
| 80% | 5.04% | 4.51% | 0.24일 | 7.78% |
| 90% | 5.04% | 3.50% | 1.22일 | 23.86% |
| 95% | 5.04% | 2.44% | 3.06일 | 43.85% |

해석:

- 90% 분위수 기준은 3일 초과 지연율을 5.04%에서 3.50%로 낮추는 시뮬레이션 결과를 보였다.
- 그 대가로 평균 추천 조정일수는 1.22일 증가하고, 전체 테스트 아이템 행의 약 23.86%에서 현재 예상 배송일보다 긴 추천일이 제시된다.
- 95% 기준은 지연율을 더 낮추지만 평균 조정일수와 조정 대상 아이템 행 비율이 커집니다.
- 90% 기준은 아직 검증되지 않은 초기 운영 가정이며, 최종 기준은 구매 전환율, 고객 안내 비용, CS 처리 비용을 함께 고려해 조정해야 한다.

Streamlit UI:

- 실행 파일: `streamlit_app.py`
- 실행 명령: `python -m streamlit run streamlit_app.py`
- 기능: 테스트 주문 예시 선택 또는 입력값 조정, Track B 위험도 확인, Track C 추천 예상 배송일 확인, 운영 액션 확인

### 18.7 UI 운영 기준 보정

보정일: 2026-08-15

초기 UI의 `샘플 주문`은 `ml_data.csv` 전체에서 `order_id` 중복을 제거한 예시 주문이었다. 이는 체험용 UI로는 충분하지만, 모델 평가와 운영 시뮬레이션을 더 명확히 하려면 학습에 사용하지 않은 주문을 보여주는 것이 더 적절하다.

수정 기준:

- UI의 주문 선택 라벨을 `테스트 주문 예시`로 변경했다.
- 선택 가능한 주문은 `order_id` 기준 holdout split 이후 test 영역에 있는 주문으로 제한했다.
- 모델은 train 영역으로 학습하고, UI 예시는 test 영역에서 가져온다.
- Track B가 고위험으로 판정한 주문만 별도 관리 목록으로 보여준다.
- Track C 추천 예상 배송일은 고위험 주문에만 운영 액션으로 적용한다.

Track B 판정 기준:

| 판정 | 낮은 만족도 리뷰 위험 확률 | 운영 의미 | Track C 적용 |
| --- | ---: | --- | --- |
| 고위험 | 46% 초과 | 판매자가 별도 확인해야 할 주문 | 적용 |
| 주의 | 35% 초과부터 46% 이하 | 배송 상태 모니터링 대상 | 미적용 |
| 일반 | 35% 이하 | 일반 처리 | 미적용 |

46% 기준은 Track B에서 validation 기준으로 선택한 긍정 리뷰 확률 threshold `0.54`를 낮은 만족도 리뷰 위험 확률로 뒤집은 값이다.

```text
낮은 만족도 리뷰 위험 확률 = 1 - 긍정 리뷰 확률
고위험 기준 = 1 - 0.54 = 0.46
```

주의 구간의 35% 기준은 운영자가 즉시 예상 배송일을 조정하지는 않지만, 배송 상태를 모니터링할 필요가 있는 중간 위험군을 분리하기 위한 UI 기준이다. 이 값은 실제 운영 비용과 모니터링 가능 인력에 따라 조정할 수 있다.

### 18.8 운영 기준 조정 기능 추가

수정일: 2026-08-15

Track B 고위험 기준과 Track C 분위수 기준을 UI에서 운영자가 직접 조정할 수 있도록 변경했다.

수정 이유:

- 고위험 기준은 CS 처리 가능 인력, 판매자 확인 가능량, 낮은 만족도 리뷰 미탐 비용에 따라 달라질 수 있다.
- 분위수 기준은 고객 기대 관리와 구매 전환율 사이의 trade-off에 따라 달라질 수 있다.
- 고정값만 제공하면 설명은 단순하지만 실제 운영 시나리오의 유연성이 부족하다.
- 따라서 기본값은 프로젝트 검증 기준으로 유지하고, 운영자는 시뮬레이션 목적으로 기준을 조정할 수 있게 하는 것이 더 실무적이다.

UI 기본값:

| 기준 | 기본값 | 근거 |
| --- | ---: | --- |
| Track B 고위험 기준 | 46% | 검증셋에서 선택한 긍정 리뷰 threshold `0.54`를 낮은 만족도 리뷰 위험 확률로 변환 |
| Track B 주의 기준 | 35% | 즉시 예상 배송일 조정 대상은 아니지만 모니터링할 중간 위험군 분리 |
| Track C 분위수 기준 | 90% | 고위험 주문의 tail risk를 줄이기 위한 초기 운영 가정 |

UI 동작:

- 고위험 기준을 올리면 더 위험한 주문만 Track C 적용 대상으로 남는다.
- 고위험 기준을 낮추면 판매자가 확인해야 할 주문 수가 늘어난다.
- Track C 분위수를 높이면 추천 예상 배송일이 더 보수적으로 길어진다.
- Track C 분위수를 낮추면 예상 배송일 증가 폭은 줄지만 지연 리스크가 더 남을 수 있다.

포트폴리오 표현:

> 운영자가 CS 리소스와 고객 경험 전략에 따라 낮은 만족도 리뷰 위험 기준과 예상 배송일 보수성을 직접 조정할 수 있는 의사결정 지원 UI를 구현했다. 기본값은 검증셋 threshold와 90% 분위수 운영 가정에 기반하되, 실제 운영에서는 기준별 지연 감소 효과와 예상 배송일 증가 폭을 비교해 조정할 수 있다.

## 19. 지티씨솔루션 면접 이후 라벨 기준 보완

작성일: 2026-09-06

지티씨솔루션 1차 면접에서 Olist 모델링의 타깃 정의에 대한 꼬리질문을 받았다. 핵심 질문은 “5점 리뷰가 대부분인 불균형 데이터에서 리뷰를 어떻게 부정과 긍정으로 나눌 것인가”, “NPS는 0점부터 10점 기준인데 1점부터 5점 별점에 그대로 적용할 근거가 있는가”였다.

보완 판단:

- Olist 리뷰 점수는 NPS가 아니므로 NPS 기준을 그대로 적용하지 않는다.
- 3점은 불만족이라고 단정하기 애매하므로 주 모델에서는 제외한다.
- 주 모델은 1점·2점을 낮은 만족도 리뷰, 4점·5점을 긍정 리뷰로 정의한다.
- 기존의 1점부터 3점까지 리스크 기준은 폐기하지 않고 민감도 분석용 정책으로 남긴다.
- 불균형 데이터에서는 accuracy보다 낮은 만족도 리뷰 기준의 `risk_recall`, `risk_precision`, `risk_pr_auc`, `flagged_rate`를 함께 본다.

면접 답변 프레임:

> 처음에는 리뷰 점수를 단순히 1점부터 3점까지와 4점부터 5점까지로 나누었지만, 다시 보니 Olist 별점은 NPS가 아니기 때문에 3점을 부정으로 단정하기 어렵다고 판단했습니다. 그래서 주 모델에서는 1점·2점을 낮은 만족도 리뷰로 정의하고 3점은 중립으로 제외했습니다. 대신 3점을 비위험으로 포함하거나 리스크로 포함하는 정책도 비교해서, 라벨 기준에 따라 클래스 비율과 운영 대상 규모가 어떻게 달라지는지 확인했습니다.

최신 산출물:

- `outputs/tables/track_a_vs_b_comparison.csv`: 주 라벨 정책 기준 Track A, Track B, baseline 성능
- `outputs/tables/label_policy_comparison.csv`: 라벨 정책별 클래스 비율과 Track B 성능
- `outputs/metadata/refreshed_search_summary.json`: 선택 라벨 정책, threshold, 핵심 지표 요약

현재 주 모델 기준에서 Track B는 낮은 만족도 라벨을 가진 평가 아이템 행의 약 62%를 탐지합니다. 다만 precision은 0.235이므로 예측 결과를 자동 조치로 쓰기보다 운영자가 먼저 확인할 후보 주문을 좁히는 모델로 설명해야 한다.

## 20. 면접 후 Track B 개선 백로그

작성일: 2026-09-07

상태: 개선 계획. 아래 진단 과정에서는 코드를 수정하지 않았으며, 면접 종료 후 순서대로 구현하고 결과를 다시 확정해야 한다.

### 20.1 개선 목적

Track B의 현재 임계값에서는 전체 평가 행의 41.3%가 위험 후보로 표시되지만, 표시된 후보 중 실제 낮은 만족도 리뷰는 23.5%다. 수동 검토나 보상처럼 비용이 드는 조치에 바로 연결하면 정상 주문을 확인하는 업무가 과도하게 발생할 수 있다.

이 개선 작업의 목적은 단순히 accuracy를 높이는 것이 아니다. 실제 운영 단위인 주문별로 위험도를 산출하고, 담당자가 처리할 수 있는 범위 안에서 낮은 만족도 주문을 최대한 먼저 찾도록 모델과 운영 기준을 다시 설계하는 것이다.

### 20.2 현재 확인된 사실

기존 공식 결과:

| 지표 | 값 | 해석 |
| --- | ---: | --- |
| 낮은 만족도 비율 | 0.157 | 행 단위 무작위 모델의 PR-AUC 기준선에 해당한다. |
| Risk Precision | 0.235 | 후보 100건 중 약 24건이 실제 낮은 만족도다. |
| Risk Recall | 0.619 | 실제 낮은 만족도 행 100건 중 약 62건을 찾는다. |
| Risk PR-AUC | 0.289 | 기준선 0.157보다는 높지만 자동 조치에 충분하다고 단정할 수 없다. |
| Flagged Rate | 0.413 | 전체 평가 행의 약 41%를 확인해야 한다. |

현재 평가 구조에서 추가로 확인한 한계:

- 입력 데이터는 주문-아이템 단위이고 리뷰 라벨은 주문 단위다.
- `order_id` 그룹 분할로 같은 주문이 학습과 테스트에 동시에 들어가는 누수는 막았지만, 최종 metric은 주문-아이템 행 단위로 계산된다.
- 여러 아이템이 있는 주문은 평가에서 더 큰 가중치를 갖기 때문에 0.235 precision과 0.413 flagged rate를 주문 업무량으로 그대로 해석하면 안 된다.
- 현재 위험 임계값 0.46은 검증셋에서 risk recall과 balanced accuracy의 차이가 가장 작은 지점으로 선택한다. CS 처리 가능량이나 오탐 비용을 직접 최적화한 기준은 아니다.

### 20.3 주문 단위 임시 진단 결과

기존 test split의 아이템별 위험 확률을 주문별로 임시 집계해 확인했다. 이 결과는 정식 주문 단위 모델이 아니라 현재 문제의 크기를 확인하기 위한 진단값이다.

| 주문 위험도 집계 | 위험 임계값 | Flagged Rate | Risk Precision | Risk Recall | Risk PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| 주문 내 최대 위험 확률 | 0.46 | 0.392 | 0.206 | 0.596 | 0.255 |
| 주문 내 평균 위험 확률 | 0.46 | 0.379 | 0.207 | 0.577 | 0.254 |

테스트 영역은 19,985개 아이템 행과 17,575개 주문으로 구성됐고, 주문 단위 낮은 만족도 비율은 0.135였다. 집계 방법에 따라 결과가 달라지며, 단순 사후 집계에서도 PR-AUC가 기존 0.289보다 낮아졌다. 따라서 다음 구현에서는 아이템 모델의 출력값을 임의로 축약하는 데서 끝내지 않고 처음부터 주문 단위 학습 데이터를 만들어야 한다.

### 20.4 개선 우선순위

#### 우선순위 1. 주문 단위 데이터와 평가 파이프라인 재구축

해야 할 일:

1. `order_id`당 한 행만 갖는 Track B 데이터셋을 만든다.
2. 아이템 정보는 주문 상품 수, 셀러 수, 총 상품 금액, 총 배송비, 최대 무게, 총 부피, 최대 거리, 최대 예상 배송일 등의 주문 단위 피처로 집계한다.
3. 여러 카테고리와 셀러가 포함된 주문은 대표값 하나로 임의 선택하지 않고 개수, 최대값, 구성 비율처럼 재현 가능한 집계 규칙을 정한다.
4. 학습·검증·테스트의 모든 metric을 고유 주문 기준으로 계산한다.

완료 기준:

- 모델링 데이터에서 `order_id`가 유일하다.
- 학습·검증·테스트 주문 ID가 서로 겹치지 않는다.
- 테스트 주문 수, 실제 낮은 만족도 주문 수, 후보 주문 수가 함께 출력된다.
- README와 산출물에서 행 단위와 주문 단위 표현이 섞이지 않는다.

#### 우선순위 2. 운영 용량과 비용을 반영한 threshold 선택

현재의 `risk recall과 balanced accuracy 차이 최소화` 기준을 주 운영 기준에서 내린다. 다음 지표를 validation 영역에서 비교한다.

- `Precision@5%`, `Precision@10%`, `Precision@20%`
- `Recall@5%`, `Recall@10%`, `Recall@20%`
- 후보 주문 수와 전체 주문 대비 후보 비율
- 정상 주문 검토 비용과 낮은 만족도 주문 미탐 비용을 반영한 예상 비용

비용 기준 예시:

```text
예상 운영 비용
= FP 수 × 정상 주문 검토 비용
+ FN 수 × 낮은 만족도 주문 미탐 비용
+ 실제 개입 주문 수 × 안내·보상 비용
```

CS 담당자가 하루에 확인할 수 있는 주문 수를 먼저 정하고, 그 범위에서 precision과 recall이 가장 적절한 cutoff를 선택한다. 위험 확률 상위 5%, 10%, 20%를 사용하는 top-k 정책도 고정 확률 임계값과 비교한다.

주의: threshold를 변경하면 precision, recall, flagged rate는 달라지지만 같은 예측 점수의 PR-AUC는 바뀌지 않는다. PR-AUC를 높이려면 위험 주문의 순위 자체를 더 잘 구분하도록 데이터와 모델을 개선해야 한다.

#### 우선순위 3. 미래 주문을 가정한 시간 분할 검증

현재 `GroupShuffleSplit`은 주문 중복 누수는 막지만 미래 배포 상황을 재현하지는 않는다. 주문 시각을 기준으로 과거 주문은 학습, 그다음 기간은 검증, 가장 최근 기간은 테스트로 분리한다.

완료 기준:

- threshold와 모델 선택은 과거 학습·검증 구간에서만 수행한다.
- 가장 최근 테스트 구간은 최종 평가 전까지 사용하지 않는다.
- 무작위 그룹 분할 결과와 시간 분할 결과를 함께 남기되, 포트폴리오의 주 성능은 시간 분할 결과로 설명한다.
- 기간별 낮은 만족도 비율과 주요 피처 분포 변화를 확인한다.

#### 우선순위 4. 주문 전에 이용 가능한 이력 피처 추가

후보 피처:

- 해당 주문 이전 판매자의 배송 지연률과 낮은 만족도 비율
- 고객의 과거 주문 횟수, 평균 리뷰, 이전 불만 이력
- 고객 지역과 판매자 지역 조합의 과거 지연률
- 카테고리별 과거 낮은 만족도 비율
- 주문 당시 판매자의 처리 주문량
- 주문 내 상품 수, 셀러 수, 카테고리 수
- 결제 수단 수, 할부 개월 수 등 주문 단위로 먼저 집계한 결제 정보
- 예상 배송 기간이 동일 경로의 과거 실제 배송 분포에 비해 촉박한 정도

모든 이력 피처는 해당 주문 발생 이전 기록만으로 계산한다. 전체 기간의 판매자 저평가율이나 지연률을 사용하면 미래 리뷰가 과거 주문 피처에 섞이는 leakage가 발생한다.

#### 우선순위 5. 예측 대상을 실행 가능한 문제로 좁히기

현재 타깃은 원인과 관계없이 모든 1점·2점 리뷰다. 상품 품질, 파손, 설명 불일치, 판매자 응대처럼 주문 시점 물류 피처로 구분하기 어려운 원인까지 포함돼 있어 Track B의 성능 상한이 낮을 수 있다.

다음 대안을 비교한다.

1. 전체 낮은 만족도 리뷰 예측을 유지한다.
2. 배송 지연 가능성이 높은 주문을 먼저 예측한다.
3. 배송 문제와 관련된 낮은 만족도 리뷰만 별도 타깃으로 정의한다.
4. 주문 시점, 결제 승인, 택배사 인계, 약속 배송일 임박 시점에 위험도를 다시 계산하는 동적 재예측 구조를 만든다.

Olist 데이터만으로 리뷰 원인을 완전하게 분리하기 어렵다면 이를 데이터 한계로 명시한다. 임의로 배송 관련 리뷰라고 단정하지 않는다.

#### 우선순위 6. 모델과 확률 품질 개선

주문 단위와 시간 분할을 먼저 확정한 뒤 다음을 비교한다.

- LightGBM 하이퍼파라미터를 average precision 기준으로 재탐색
- `class_weight` 적용 여부 비교
- 로지스틱 회귀 등 단순 모델과 LightGBM 비교
- 확률 calibration 전후 Brier score와 reliability curve 비교
- 여러 시간 구간에서 PR-AUC와 Precision@K의 변동 확인

리샘플링이나 class weight는 자동으로 precision을 높여주는 방법이 아니므로 주문 단위 기준선보다 실제 검증 성능이 좋아질 때만 채택한다. Calibration은 확률 0.6의 의미를 해석하기 쉽게 하지만 순위가 같다면 PR-AUC 자체를 크게 높이지 않는다.

#### 우선순위 7. 운영 파일럿과 효과 검증 설계

모델 성능이 개선돼도 선제 안내나 배송 약속 조정이 실제 리뷰와 재구매를 개선한다고 바로 결론 내리지 않는다. 고위험 주문을 대상으로 무작위 실험군과 홀드아웃을 구성한다.

- 실험군: 선제 배송 안내 또는 운영 개입 적용
- 홀드아웃: 기존 처리 유지
- 최종 지표: 낮은 만족도 리뷰율, CS 문의율, 취소율, 재구매율
- 운영 지표: 개입 주문 수, 주문당 처리 시간, 보상 비용, 오탐 주문 대응 비용

모델의 `prediction quality`와 개입의 `business impact`를 별도로 평가한다.

### 20.5 면접 후 실행 순서

- [ ] 현재 코드와 `outputs/`를 재실행해 기존 결과가 재현되는지 확인한다.
- [ ] 주문 단위 집계 규칙과 최종 피처 목록을 문서로 먼저 확정한다.
- [ ] 주문 단위 학습·검증·테스트 파이프라인을 구현한다.
- [ ] 시간 분할과 무작위 그룹 분할 결과를 비교한다.
- [ ] threshold별 precision, recall, 후보 주문 수 표를 생성한다.
- [ ] Precision@K·Recall@K와 비용 기반 운영 기준을 추가한다.
- [ ] 과거 이력 피처를 하나씩 추가하고 증분 성능을 기록한다.
- [ ] 동적 재예측 또는 타깃 재정의 실험을 진행한다.
- [ ] README, 노트북, Streamlit 설명, 산출물 CSV를 같은 기준으로 갱신한다.
- [ ] 최종 선택과 폐기한 대안의 이유를 이 문서에 이어서 기록한다.

### 20.6 피해야 할 수정

- test 성능을 본 뒤 threshold를 선택하지 않는다.
- 주문-아이템 metric을 주문 업무량으로 설명하지 않는다.
- 단순히 threshold를 높인 결과를 PR-AUC 개선이라고 표현하지 않는다.
- SMOTE나 class weight를 적용했다는 사실만으로 불균형 문제가 해결됐다고 판단하지 않는다.
- 미래 주문의 리뷰나 배송 결과로 과거 판매자 이력 피처를 만들지 않는다.
- 현재 성능으로 자동 보상, 자동 차단, 배송 약속 변경을 바로 실행하지 않는다.


## 21. 제출 이후 설명 정비 — 2026-09-23

### 21.1 사실과 해석 기준

- 전처리 `delay_days`는 도착 시각 차이를 일수로 환산하고 소수 첫째 자리로 반올림한 값입니다. 기존 범주값은 모델 입력 호환성을 위해 유지하고 표시에서는 `0일 초과·3일 이하`, `3일 초과·6일 이하`, `6일 초과·9일 이하`, `9일 초과`로 읽습니다.
- 핵심 리뷰 통계는 주문별 1행, 모델 성능은 주문·아이템 행 기준입니다. `order_id` 그룹 분할은 주문 간 데이터 누수를 막기 위한 설계이며 주문 단위 성능 평가를 뜻하지 않습니다.
- 3일 경계 비교의 1.83점은 관찰된 평균 차이입니다. Mann–Whitney U는 리뷰 점수 분포를 비교하며, rank-biserial `r=0.594`는 짧은 지연 구간의 점수가 높은 방향의 순위 기반 효과크기입니다. 원시 `p_value=0.0`은 보고에서 `p<0.001`로 표시합니다.
- 전처리는 `review_answer_timestamp`로 정렬한 뒤 주문별 마지막 리뷰를 남깁니다. 통계 셀은 병합 데이터에서 주문별 첫 행을 남깁니다. 복수 리뷰·동률 처리와 표본 일관성의 추가 검증은 별도 과제입니다.
- EDA는 그룹 분할 후 `df2 = X_train_valid.copy()`로 학습·검증 영역에 제한합니다. 1단계에서 전체 데이터를 사용한다고 적었던 설명은 이 재할당을 누락한 오류여서 교정했습니다. 전처리 탐색에서도 경계를 검토했으므로 현재 검정을 독립적인 경계 확정 검증이나 최적 임계값의 증거로 확대하지 않습니다.

### 21.2 Track C 수치의 적용 범위

`evaluate_track_c_quantiles`의 표는 고위험 주문 필터 없이 전체 테스트 아이템 행에 예상 배송일 검토값을 적용한 시뮬레이션입니다. `current_over_3_delay_rate`와 `recommended_over_3_delay_rate`는 정수형 배송 소요일과 예상 소요일의 차이가 3을 초과하는 비율입니다. EDA의 반올림된 지연일 정의와 통합하지 않습니다.

`share_orders_adjusted`는 이름과 달리 조정된 아이템 행의 비율이며, 평균 조정일수도 전체 테스트 행 기준입니다. 5.04%에서 3.50%로의 변화는 실제 배송 소요일을 줄인 성과가 아닙니다. 고위험 주문에만 검토값을 제시하는 UI 흐름과 전체 행 결과표를 구분합니다. 구매 전환 비용·고객 수용성·실제 개입 효과는 검증되지 않았습니다.

### 21.3 이번 단계와 후속 단계

1단계는 README와 노트북 설명을 원본 코드에 맞추는 작업입니다. 기존 미커밋 Track B 백로그와 그 안의 임시 진단값은 보존하며 이번 작업에서 재현했다고 주장하지 않습니다. 원본 데이터, 모델 코드, 입력 범주값, 라벨 정책, 임계값과 성능 CSV는 변경하지 않습니다.

2단계에서 관련 EDA 셀만 실행해 표시용 매핑, 표·차트와 노트북 출력을 갱신했습니다. 2026-09-24에 3단계 최종 대조를 완료하고 Company-Research의 `docs/olist-follow-up-todo.md`와 `docs/application-handoff.md`에 전체 수정표, 검증 결과, 포트폴리오 반영 과제와 미검증 항목을 기록했습니다. 관련 EDA 결과표의 모든 수치·열 구조, 모델 코드·성능표·전처리 데이터와 기존 20절 백로그가 보존됐습니다. 범용 포트폴리오와 제출 NICE 자산은 이번 정비에서 수정하지 않았습니다.

지원자 관점에서는 지표 정의와 실제 계산 단위를 구분한 판단을 설명할 수 있습니다. 이 설명 교정을 신규 실험이나 모델 성능 개선 성과로 제시하지 않습니다.


### 21.4 2단계 통계 재현과 표시 교정

- EDA의 학습·검증 영역은 87,482개 아이템 행·76,627개 주문이며, 테스트와 주문 ID 교집합은 0입니다. 이는 EDA 분할 확인이며 Track A/B의 최종 실행 분할까지 검증한 것은 아닙니다.
- 지연 여부 비교: 정상·조기 70,558건, 지연 6,069건, 평균 4.2915점·2.5538점, 차이 1.7377점입니다. 지연 그룹을 먼저 입력한 rank-biserial은 -0.5565입니다.
- 이전 설명 1.68점·-0.54는 전체 데이터 아이템 행 계산 1.6786점·-0.5350과 반올림 결과가 일치합니다. 현재 주문 기준 결과와 섞지 않습니다.
- 3일 경계 비교: 2,086건·3,983건, 평균 차이 1.82670140041485, U=6,621,470, rank-biserial=0.5938953399503017로 기존 CSV 수치가 재현됐습니다.
- 결과표의 `group` 표시명만 정확한 경계로 교정합니다. 기존 열 `mean_diff_1_3_minus_4_plus`는 호환성을 위해 유지하며 실제 의미는 `0 < delay_days <= 3` 구간 평균에서 `delay_days > 3` 구간 평균을 뺀 값입니다. 원시 `p_value`는 보존합니다.
- EDA 표시용 매핑은 모델 입력 `delay_days_cat`나 저장 전처리 데이터를 변경하지 않습니다. `delay_review_threshold_bar.png`는 주문 기준 셀에서만 저장합니다.

## 22. 기존 모델 검증 — 2026-09-30

### 22.1 범위와 재현 근거

현재 데이터와 기존 전처리·라벨·분할 함수를 메모리에서 실행하고 저장 출력·CSV·메타데이터를 대조했습니다. 전체 노트북, 모델 학습, 결과 생성 스크립트는 실행하지 않았습니다. 아래 내용은 현재 코드의 재현과 저장 기록의 일관성 검사이며 과거 학습 실행 전체의 재현을 뜻하지 않습니다. 기존 20절 백로그·21절 교정 기록은 보존합니다.

근거는 `src/olist_delivery_models.py`, `notebooks/01_preprocessing.ipynb`, `notebooks/03_ml_classifier.ipynb`, `scripts/generate_track_c_outputs.py`, `outputs/tables/track_a_vs_b_comparison.csv`, `outputs/tables/model_comparison_results.csv`, `outputs/tables/label_policy_comparison.csv`, `outputs/metadata/refreshed_search_summary.json`, `streamlit_app.py`입니다. 셀 번호는 JSON 배열 기준 0부터 셉니다. 실행 환경은 pandas 2.3.3, scikit-learn 1.6.1, LightGBM 4.6.0입니다. 임시 진단 JSON은 `/tmp/olist-model-audit-stage1-20260930.json`, `/tmp/olist-model-audit-stage2-20260930.json`, `/tmp/olist-model-audit-stage3-20260930.json`에 있으며 영구 기록은 이 절입니다.

### 22.2 데이터·분할 — 확인 완료

전체 109,294행에서 준비 함수의 추가 결측 제외는 0행이며, 3점 리뷰 9,187행을 제외한 100,107행·87,872개 주문으로 분할했습니다.

| 영역 | 아이템 행 | 고유 주문 | 낮은 만족도 행 | 낮은 만족도 비율 |
|---|---:|---:|---:|---:|
| 학습 | 64,103 | 56,237 | 10,327 | 16.1100% |
| 검증 | 16,019 | 14,060 | 2,678 | 16.7176% |
| 테스트 | 19,985 | 17,575 | 3,128 | 15.6517% |

세 영역 간 주문 교집합은 모두 0이며 모든 모델링 행이 한 번씩 배정됐습니다. Track A/B/기준선의 각 영역은 행 인덱스·정답·주문 ID가 동일하고 저장 CSV의 테스트 규모와 일치합니다. 같은 주문 내 라벨 충돌도 0입니다. 과거 실행의 분할 ID 파일은 확보하지 못했으므로 과거 실행 자체를 입증하는 결과로 확대하지 않습니다.

### 22.3 피처·전처리 — 문제와 한계

전처리 노트북의 필요한 셀만 메모리에서 재현했습니다. `merged_data.csv`에서 리뷰 결측을 제외한 109,296행에 기존 처리를 적용하고 출고일 이상치 2행을 제외한 결과, 109,294행의 순서와 Track B 22개 피처가 저장 데이터와 일치했습니다(숫자는 부동소수점 허용오차 비교). 파일 저장 셀은 실행하지 않았습니다.

| 피처 | 원천·계산 | 주문 확정 시점 판단 |
|---|---|---|
| `order_item_id`, `price`, `freight_value` | 주문 아이템 순번·가격·배송비 | 주문 구성 확정 후 사용 가능하다는 설계입니다. |
| `freight_ratio`, `total_price` | 배송비/가격, 가격+배송비 | 행별 계산이며 다른 표본의 정보를 사용하지 않습니다. |
| `product_weight_g`, `product_length_cm`, `product_height_cm`, `product_width_cm` | 상품 카탈로그 | 주문 당시 카탈로그 값인지 변경 이력 확인이 필요합니다. |
| `order_purchase_dayofweek`, `order_purchase_month` | 주문 시각의 요일·월 | 주문 시점에 계산 가능합니다. |
| `expected_delivery_days` | 예상 배송일과 주문 시각 차이의 정수 일수 | 실제 배송일을 사용하지 않지만 예상일의 사후 갱신 여부는 확인 필요입니다. |
| `main_category`, `sub_category` | 상품 카테고리의 고정 사전 매핑 | 타깃 통계는 사용하지 않습니다. 카탈로그 시점은 확인 필요입니다. |
| `distance_km`, `distance_cat` | 고객·판매자 좌표의 직선거리와 고정 구간 | 위치 기반 설계이며 지리 참조자료의 시점·결측 대체 범위는 별도 제약입니다. |
| `cross_state`, `is_sp_customer`, `is_sp_seller`, `sp_route_type`, `sp_route_type_customer`, `sp_route_type_seller` | 고객·판매자 주 코드의 비교와 SP 여부 | 주문 주소·판매자 위치가 확정된 뒤 계산 가능한 규칙입니다. |

실제 배송일·출고일·지연일·리뷰 점수는 Track B 입력 목록에 없습니다. 이는 코드 의존성 확인이며 원자료가 실제 주문 당시 스냅샷임을 입증한 것은 아닙니다.

**확인된 전처리 문제:** 전처리 셀 92·94·95·121이 분할 전 전체 데이터로 최빈값·중앙값을 계산합니다. 학습 파이프라인의 `SimpleImputer`는 학습 영역에만 맞춰지지만, 앞에서 이미 채운 값에는 효과가 없습니다.

| 처리 | 전체 최종 데이터에서 실제 대체 | 주 모델 학습/검증/테스트에서 실제 대체 | 학습 영역 통계와 비교 |
|---|---:|---|---|
| 고객 좌표 | 위도·경도 각각 238행 | 각각 124 / 40 / 56행 | 일부 대체값이 달라지거나 학습 도시 통계만으로 채울 수 없습니다. |
| 판매자 좌표 | 각각 249행 | 각각 146 / 29 / 44행 | 이번 비교의 대체값은 동일합니다. |
| 상품 무게·크기 | 각 열 18행 | 각각 13 / 4 / 1행 | 이번 비교의 중앙값은 동일합니다. |
| 거리 중앙값 | 49행 | 30 / 7 / 9행 | 전체 중앙값 432.9149949km, 현재 학습 영역의 대체 전 거리 중앙값 428.2057623km입니다. |

전체와 주 모델 합계의 차이는 3점 리뷰 제외 때문입니다. 거리의 학습 영역 중앙값 비교도 기존 전체 기반 좌표 대체 뒤의 값이므로, 모든 전처리를 학습 전용으로 바꾼 최종 대체값을 뜻하지 않습니다. 성능 영향의 방향·크기는 재학습하지 않아 미확인입니다. 리뷰 정답을 피처로 사용한 누수와 테스트 분포를 전처리에 사용한 문제를 구분합니다.

원천 주문은 배송 완료 필터, 배송 시각 결측·모순 제거 및 리뷰 확보 조건을 거칩니다(전처리 셀 38·45·47·88·105). 따라서 현재 성능은 취소·미배송을 포함한 전체 주문 모집단의 성능이 아닙니다. 이 표본 선택은 적용 대상의 한계로 기록하며 즉시 코드 오류로 단정하지 않습니다.

### 22.4 평가·선택 과정 — 확인과 불일치

- 평가 함수는 원래 라벨 0을 위험 양성으로 변환하고 `1 - positive_proba`로 위험 점수를 계산합니다. 고정 4행 예시에서 노트북과 공용 함수의 지표가 일치했고 임계값과 정확히 같은 긍정 확률은 비위험으로 처리됐습니다. 임계값 후보의 점수가 동률이면 먼저 나온 후보를 선택합니다.
- 현재 A/B 코드는 검증 확률로 0.10부터 0.90까지 0.01 간격 후보를 비교합니다. `abs(risk_recall - balanced_acc)` 최소화는 두 클래스 재현율 차이 최소화에 해당하며 처리 용량·비용 최적화 기준이 아닙니다. CSV·메타데이터의 긍정 임계값은 A 약 0.61, B 약 0.54입니다.
- 현재 탐색 코드는 학습 영역의 `StratifiedGroupKFold`와 macro F1을 사용합니다. 하지만 과거 전체 탐색 로그·예측 파일은 확보하지 못했습니다. 모델 비교 CSV에는 라벨 정책·표본·실행 식별자가 없고, LightGBM 임계값 0.57은 현재 노트북의 0.61과 다릅니다. 과거 비교표를 현재 주 정책에서의 재탐색 완료 증거로 단정하지 않습니다.
- XGBoost 비교 CSV의 `min_child_weight`는 1이나 노트북 셀 35의 후속 학습은 3을 하드코딩합니다. 탐색의 `best_estimator_`를 그대로 평가하는 경로가 아니므로 비교표를 그대로 재생성한다고 보장할 수 없습니다.
- 라벨 정책은 목적에 따른 정책으로 설명돼 있으며 현재 코드에 테스트 성능 최대화로 정책을 자동 선택하는 경로는 없습니다. 다만 정책별 테스트 결과가 공개된 이후 과거 의사결정에 영향을 줬는지까지 증명할 이력은 없습니다.

| 저장 결과 | 대조 결과 |
|---|---|
| Track B 테스트 | 노트북 셀 45·46과 현재 CSV 일치. 저장 혼동행렬은 `[[1937,1191],[6320,10537]]`입니다. |
| Track A 테스트 | 노트북 셀 38·46은 `[[2121,1007],[5307,11550]]`, 현재 CSV의 지표가 함의하는 행렬은 `[[2121,1007],[5271,11586]]`입니다. 오탐 36행 차이이며 원인은 미확정입니다. |
| 라벨 정책 | 셀 48의 저장 출력에서 주 정책 재현율 0.591752, 현재 CSV는 0.619246입니다. 나머지 정책 출력도 다릅니다. |
| CSV 자체 | A/B/기준선의 정확도·균형 정확도·정밀도·재현율·후보 비율·macro F1은 테스트 클래스 수와 산술적으로 일치합니다. |

행렬 순서는 실제/예측 라벨 `[0,1]`입니다. CSV 행렬은 집계값으로 역산한 것이며 개별 예측을 재실행한 결과가 아닙니다. AP는 확률 순위가 필요하므로 직접 재현하지 않았습니다. 기존 열 `risk_pr_auc`는 코드상 `average_precision_score`이며 사다리꼴 PR 곡선 면적 계산과 구분합니다. 저장 출력·CSV 차이는 확정했지만 어느 과거 실행이 원인인지는 단정하지 않습니다.

UI는 `train_console_artifacts`에서 80,122행으로 별도 학습하고, 성능 비교 경로는 64,103행으로 학습합니다. UI 학습 영역은 평가 경로의 학습+검증 영역이며 테스트 19,985행은 같습니다. 기본 임계값을 재사용하지만 동일한 학습 모델은 아니므로 저장 재현율을 UI 모델의 직접 검증 성능으로 붙이지 않습니다. UI의 테스트 주문 목록은 첫 아이템 행, 고위험 목록은 위험도 정렬 후 주문 중복 제거를 사용하므로 아이템 평가표와 주문별 UI 표현도 구분해야 합니다.

### 22.5 후속 해결 순서 — 아직 실행하지 않음

1. 저장 결과의 정본·실행 경로를 정리합니다. Track A와 라벨 정책의 이전 노트북 출력에는 출처·실행 조건을 붙이고, 불일치 원인이 미확정인 수치를 임의 교체하지 않습니다. 탐색 파라미터와 실제 평가 모델을 연결합니다.
2. 같은 분할·라벨·모델·임계값을 유지한 비교 실험을 설계합니다. 학습 주문만으로 좌표·상품·거리 대체값을 구하고 검증·테스트에 적용합니다. 학습에 없는 도시의 처리 규칙과 과거 전처리 비교군을 먼저 확정합니다. 교차검증을 다시 수행한다면 대체 통계도 각 fold의 학습 영역 안에서 계산해야 합니다.
3. 필요 재학습의 이유는 수정 전처리가 피처 값을 바꾸기 때문입니다. 기존 데이터·성능표를 보존한 별도 산출물에 분할 식별자, 실행 설정·버전, 아이템별 예측 확률, 혼동행렬·AP와 비교 결과를 저장하는 계획을 먼저 승인받습니다. 이번 기록은 재학습 실행 승인이 아닙니다.
4. UI 학습 모델과 평가 모델의 대응을 별도로 검증합니다. 이후 주문 단위 평가·시간 분할·운영 상위 건수·비용 기준 임계값은 전처리 수정 효과와 섞지 않고 새 실험으로 다룹니다.

현재 결론은 분할·평가 함수 확인 완료, 분할 전 대체 및 저장 결과 불일치 확인, 성능 영향과 과거 실행 전체 재현 미검증입니다. 포트폴리오의 기존 수치는 보존하며 향후 검증 조건·실험 버전을 확정한 뒤 반영 여부를 판단합니다. NICE 제출본은 보존합니다.

## 23. 저장 결과 출처 정리 — 2026-09-30

### 23.1 결과별 사용 기준

현재 보고 수치의 기준은 라벨·표본 정보가 포함된 현행 CSV로 유지합니다. 이는 과거 실행의 완전한 재현 인증이 아니라 문서 간 참조 기준입니다. 노트북의 다른 출력은 삭제·교체하지 않고 과거 실행 기록으로 보존합니다.

| 결과 | 저장 이력·생성 경로 | 사용 기준 |
|---|---|---|
| `track_a_vs_b_comparison.csv` | 2026-09-06 `23d9009`에서 주 라벨로 변경. ML 노트북 셀 46과 생성 스크립트 두 곳이 같은 파일에 저장 | 현재 보고 기준. 어떤 실행이 마지막으로 저장했는지는 커밋만으로 확정 불가입니다. |
| `label_policy_comparison.csv` | `23d9009`에서 추가. 생성 스크립트가 `evaluate_label_policies` 결과를 저장 | 정책별 민감도 참고. 노트북 셀 48은 CSV를 읽지만 저장된 출력은 현재 CSV와 다릅니다. |
| `refreshed_search_summary.json` | `23d9009`에서 주 정책·임계값·요약 갱신. 생성 스크립트가 저장 | 현재 CSV와 연결된 요약. 전체 탐색 과정의 실행 로그는 아닙니다. |
| `model_comparison_results.csv` | 2026-08-02 `7634166`에서 추가된 뒤 현행 HEAD까지 변경 없음 | 당시 1–3점 위험/4–5점 긍정 정책 문맥의 과거 모델 비교 자료. 현재 3점 제외 정책의 재탐색 증거로 사용하지 않습니다. |
| 노트북 셀 38·45·46·48의 저장 출력 | 현행 출력이 `23d9009`의 출력과 동일함을 비교 확인 | 최근 표현 교정에서 생긴 차이가 아닙니다. 현재 CSV와 다른 출력의 실제 생성 시각·환경은 미확인입니다. |

`7634166`의 ML 노트북 라벨 함수는 `score >= 4`만 1로, 나머지를 0으로 만듭니다. 과거 비교 CSV 자체에는 라벨·표본·실행 ID가 없으므로 같은 커밋의 코드 문맥을 근거로 분류하되 정확한 실행 과정을 복구했다고 표현하지 않습니다. 저장소의 현재 검색 범위에서 이 비교 CSV를 직접 생성하는 저장 코드도 찾지 못했습니다.

### 23.2 Track A 불일치 원인 후보

모델을 학습하지 않고 노트북의 데이터 준비 셀 6·11·13·15·19와 공용 준비 함수를 실행해 입력을 비교했습니다.

- A의 33개 컬럼 집합·각 값·자료형·행 인덱스는 같지만 컬럼 순서는 다릅니다. 노트북은 기존 데이터 순서에서 제외 컬럼을 제거하고, 공용 코드는 `TRACK_A_COLS` 순서를 명시합니다.
- B는 두 경로 모두 `PRE_ORDER_COLS`를 사용하므로 컬럼 순서까지 같습니다.
- A의 피처 일부 선택 설정(`colsample_bytree`)과 입력 순서가 함께 존재하므로 열 순서는 재현 실험에서 통제할 후보입니다. 오탐 36행 차이의 원인으로 확정하지 않습니다. 노트북과 공용 코드의 스레드 설정도 같지 않아 실행 환경을 함께 고정해야 합니다.
- 공용 생성 스크립트는 모델을 다시 학습하면서 현재 CSV와 메타데이터를 덮어씁니다. 출처 정리를 위해 이 스크립트를 실행하지 않습니다.

### 23.3 완료 범위와 다음 설계 조건

출처 분류와 두 입력 경로 대조는 완료했습니다. 저장 출력 차이의 정확한 원인·과거 확률값 복구는 미완료입니다. 기존 성능 수치·CSV·노트북은 변경하지 않았습니다.

다음 비교 실험 설계에서는 공용 코드의 피처 순서·모델 설정·실행 환경을 고정하고, 같은 실행 조건에서 기존 전처리 비교군과 학습 전용 대체 실험군을 비교해야 합니다. 과거 CSV와 새 실행의 차이를 곧바로 전처리 효과로 해석하지 않습니다. Track A 입력 순서 원인 확인은 별도 진단으로 분리합니다. 학습에 없는 도시 처리, 비교군 구성, 저장할 예측·설정·분할 근거를 정한 후 재학습 범위를 제시합니다.

## 24. 학습 영역 전용 결측치 처리 비교 실험 설계 — 실행 전

작성일: 2026-09-30. 실험 목적은 분할 전 전체 기반 대체가 현재 A/B 평가에 미친 영향을 비교하는 것입니다. 이 절은 실행 계획이며 신규 학습·코드 변경은 아직 수행하지 않았습니다.

### 24.1 고정 조건과 비교군

| 조건 | 실행 계획 |
|---|---|
| 대상 | 현재 주 정책의 100,107개 아이템 행. 배송 완료·리뷰 확보 필터와 라벨은 유지 |
| 분할 | 현재 64,103 / 16,019 / 19,985행. 주문 ID 기준 기존 분할을 한 번 만들고 네 실행에서 같은 인덱스 재사용 |
| 모델 | 공용 `build_track_a_pipeline`, `build_track_b_pipeline`의 설정·피처 순서 유지. random_state=42, n_jobs=1. A/B 각각 독립 파이프라인 |
| 비교군 | 현재 `ml_data.csv`로 기존 전처리를 재현한 A/B 두 모델 |
| 실험군 | 동일 행에서 좌표·상품·거리 대체 통계만 학습 영역으로 제한한 A/B 두 모델 |
| 임계값 | 현재 A/B CSV의 `positive_threshold` 원시 값을 그대로 읽어 네 실행에 고정. 반올림·재선택하지 않음 |
| 실행량 | A/B × 기존/수정 전처리 = 총 4회 학습. CV·모델 탐색·임계값 탐색·UI·Track C 학습 제외 |

재학습이 필요한 이유는 대체값과 그 파생 피처가 달라지면 학습된 모델도 달라지기 때문입니다. 과거 CSV와 새 실험군을 직접 비교하지 않고, 같은 환경에서 새로 실행한 비교군과 실험군의 차이를 봅니다. 비교군과 과거 CSV의 차이는 재현 오차로 따로 기록합니다. 입력·분할 검증이 통과하면 과거 지표와의 불일치만으로 실험군을 튜닝하지 않으며, 고정된 두 조건의 비교를 진행합니다.

### 24.2 행 연결과 대체 규칙

`merged_data.csv`와 현재 모델링 데이터는 `(order_id, order_item_id)` 복합키가 각각 유일하며, 100,107개 모델 행 모두 원본에 일대일 연결됨을 확인했습니다. 키 연결 후 모델링 인덱스·행 순서를 복구하며, 새 행 제거·추가는 금지합니다.

1. 원본의 대체 전 고객·판매자 위도·경도 및 상품 무게·크기를 복원합니다. 기존 병합·우편번호별 평균 좌표·상품 매핑은 유지합니다. 이 실험은 원천 지리정보의 역사적 가용성까지 해결하는 설계가 아닙니다.
2. 학습에 배정된 아이템 행만으로 기존 도시 키 기준 위도·경도 최빈값을 계산합니다. 동률은 기존 방식인 정렬된 `mode()` 첫 값을 사용합니다. 관측된 좌표는 변경하지 않습니다.
3. 학습에 도시가 없거나 해당 좌표가 모두 결측이면 좌표는 결측으로 둡니다. 새 지역 단계나 임의 좌표를 추가하지 않습니다. 현재 데이터에서 해당 고객 좌표 행은 학습 30·검증 8·테스트 9행이고 판매자 좌표는 0행입니다.
4. 상품 4개 수치 열은 학습 원본 행의 열별 중앙값으로 채웁니다. 아이템 가중 방식을 유지하며 주문별로 재집계하지 않습니다.
5. 좌표 대체 후 기존 Haversine 식으로 거리를 계산합니다. 남은 거리 결측은 학습 영역의 관측 가능한 거리 중앙값으로 채웁니다. 검증·테스트는 이 중앙값만 적용받습니다. 22절의 428.21km는 이전 좌표 대체 뒤 비교값이므로 새 중앙값으로 하드코딩하지 않습니다.
6. 기존 경계로 `distance_cat`을 다시 계산하고, A의 거리 파생값 `delivery_speed`, `day_per_km`, `delivery_distance`를 갱신합니다. 그 밖의 가격·시각·라벨·주/카테고리 피처는 비교군과 동일해야 합니다. 무한값 처리와 파이프라인의 학습 전용 대체·인코딩은 기존 방식을 유지합니다.

### 24.3 구현·산출물 계획

승인 후 `scripts/audit_imputation_comparison.py`를 별도 진입점으로 추가합니다. 공개 모델 모듈·기존 노트북·생성 스크립트는 변경하지 않습니다. 새 스크립트는 기존 순수 준비·분할 함수와 모델 빌더를 사용하며 기존 결과를 저장하는 함수를 호출하지 않습니다. 함수 구성과 입출력은 구현 전에 설명합니다.

산출물은 `outputs/experiments/imputation-audit-YYYYMMDD-HHMMSS/`에만 저장하며 이미 있는 디렉터리는 덮어쓰지 않습니다.

- `manifest.json`: 실행 시각·Git 상태, 코드/입력 SHA256, Python·패키지 버전, 모델 실제 파라미터·피처 순서·임계값, 라벨·분할 설정, 기존 CSV와 비교 여부.
- `split_assignments.csv`: 원래 행 인덱스, 주문·아이템 키, 분할, 정답. 원자료 주소·좌표는 내보내지 않습니다.
- `imputation_audit.json`: 학습에서 계산한 통계, 대체·미해결·변경 행 수, 그룹별 집계 및 모든 입력 불변 검증 결과.
- `predictions.csv`: 비교군/실험군·A/B·검증/테스트·행 키·정답·긍정 및 위험 확률·고정 임계값 예측·피처 변경 여부.
- `metrics.csv`: 아이템 행 기준 정확도·균형 정확도·위험 정밀도/재현율·AP·후보 비율·macro F1 및 명시적 위험 TP/FN/FP/TN. 검증과 테스트를 분리합니다.
- `comparison.md`: 같은 환경의 실험군−비교군 차이와 과거 CSV−새 비교군 차이를 구분합니다. 예측 변경 행 수·전처리 직접 변경 행 수도 함께 보고합니다. 학습이 달라지면 직접 대체하지 않은 행의 예측도 바뀔 수 있습니다.

이번 실험에서 모델 파일을 배포하거나 UI 모델을 교체하지 않습니다. 개별 확률과 실행 설정은 보존하여 AP와 혼동행렬을 재계산할 수 있게 합니다.

### 24.4 학습 전·후 완료 기준

- 학습 전: 복합키 유일성·행 연결 100%, 라벨/분할/행 순서 동일, 주문 교집합 0, 각 피처 자료형·순서 일치, 허용한 대체·거리 관련 피처 외 변경 0을 검사합니다. 실패하면 학습 전에 중단합니다.
- 대체 규칙 검증: 학습에 없는 도시·모든 좌표 결측·최빈값 동률·거리 결측 예시를 확인합니다. 검증·테스트의 좌표·상품 값을 바꾼 진단 입력에서도 학습 통계와 학습 변환 결과가 그대로인지 확인합니다. 이 진단은 메모리에서만 수행합니다.
- 학습 후: 저장 예측으로 지표·혼동행렬 재계산, 유한한 0–1 확률, 분할별 예측 개수, 모델·조건별 설정 일치를 확인합니다. 지표 변화 방향이나 크기를 미리 성공 조건으로 정하지 않습니다.
- 기존 코드·CSV·전처리 데이터·NICE·포트폴리오의 해시 보존, 추가 파일 범위와 `git diff --check`를 확인하고 인계 문서를 갱신합니다.
- 기존 테스트는 여러 번 결과를 확인한 표본이므로 새 실험 결과를 완전히 미관측인 최종 일반화 성능이라고 주장하지 않습니다. 이번 결과는 전처리 수정 민감도 비교입니다. 향후 시간 분할·주문 단위 평가·임계값 재선택은 별도 설계입니다.

## 25. 학습 전용 결측치 처리 비교 실험 완료 — 2026-09-30

사용자가 기존 코드를 보존한 별도 실험 구현·학습을 승인했습니다. 24절은 실행 전 계획이며 이 절이 실제 결과입니다.

- 추가 코드: `scripts/audit_imputation_comparison.py`. 기존 모델 모듈·전처리/ML 노트북·생성 스크립트는 변경하지 않았습니다.
- 실행: `python3 scripts/audit_imputation_comparison.py --validate-only` 통과 후 옵션 없이 총 4회 학습했습니다. A/B 각각 기존 전처리와 학습 전용 대체 조건이며, 추가 튜닝·CV·UI/Track C 학습은 하지 않았습니다.
- 결과: [비교 보고서](../outputs/experiments/imputation-audit-20260930-220848/comparison.md). 같은 폴더의 `manifest.json`, `split_assignments.csv`, `imputation_audit.json`, `predictions.csv`, `metrics.csv`에 설정·분할·학습 대체 통계·예측·지표를 보관했습니다. 개별 예측은 검증/테스트 합계 144,016행입니다.

### 25.1 재현과 변경 범위

공용 코드 순서·설정·환경으로 다시 학습한 기존 전처리 A/B는 과거 CSV의 테스트 정확도·균형 정확도·정밀도·재현율·AP·후보 비율·macro F1과 모두 일치했습니다. 이번에는 저장 예측으로 AP와 혼동행렬을 직접 재계산할 수 있습니다. 이는 현행 CSV 수치 재현 성공이며 과거 노트북 출력 불일치의 정확한 발생 원인까지 입증한 것은 아닙니다.

행·라벨·분할·피처 순서·자료형·고정 임계값을 유지했습니다. 실제 피처 변경은 학습 41행·검증 14행·테스트 13행입니다. 전체 68행에서 `distance_km`와 A의 `delivery_speed`, `day_per_km`, `delivery_distance`만 달라졌습니다. 상품 4개 중앙값과 거리 범주는 이번 데이터에서 변화가 없습니다. 학습 거리 중앙값은 428.20576234972293km입니다.

거리 재계산은 기존 저장값과 `rtol=1e-12`, `atol=1e-9km` 이내이면 기존 값을 유지해 직렬화·부동소수점 오차를 처치 차이로 세지 않았습니다. 실제 실험에서는 좌표 최빈값을 학습 영역에서 새로 구한 뒤 거리 중앙값을 계산했습니다.

### 25.2 테스트 결과 — 아이템 행 기준

| 모델 | 전처리 | 위험 정밀도 | 위험 재현율 | AP | 균형 정확도 | 위험 TP/FN/FP/TN |
|---|---|---:|---:|---:|---:|---|
| A | 기존 | 0.286932 | 0.678069 | 0.505432 | 0.682690 | 2121/1007/5271/11586 |
| A | 학습 전용 | 0.284381 | 0.672315 | 0.504897 | 0.679190 | 2103/1025/5292/11565 |
| B | 기존 | 0.234589 | 0.619246 | 0.289262 | 0.622164 | 1937/1191/6320/10537 |
| B | 학습 전용 | 0.234436 | 0.616368 | 0.289564 | 0.621437 | 1928/1200/6296/10561 |

A 재현율은 0.5754%p, B는 0.2877%p 낮아졌습니다. B AP는 약 0.000302 높아졌으므로 모든 지표가 같은 방향으로 바뀐 것은 아닙니다. 검증 영역에서는 A 재현율 0.678118→0.676998·AP 0.513561→0.515370, B 재현율 0.616878→0.613891·AP 0.304349→0.301947입니다. 검증/테스트의 상세 지표를 함께 저장했고 유리한 지표만으로 실험군을 선택하지 않았습니다.

테스트 예측 라벨은 A 727행, B 295행이 달라졌습니다. 직접 피처가 바뀐 테스트 13행의 예측 라벨은 동일했고, 다른 행에서 변경이 발생했습니다. 학습 행의 값 변화로 학습된 트리가 달라지므로 영향이 직접 대체된 행에만 한정되지 않습니다. 개별 확률은 라벨 변화와 별개입니다.

### 25.3 검증과 해석

- 원본 행의 일대일 연결, 분할 규모·주문 비중첩, 허용 피처 외 불변 검사를 통과했습니다.
- 미등록 도시·도시 내 전부 결측·최빈값 동률·거리 중앙값 대체 예시를 확인했습니다. 검증·테스트 값 교란에도 학습 통계·학습 변환 결과가 동일했습니다.
- 저장 확률을 다시 읽어 지표·혼동행렬을 검증했고, 별도 집계로 예측 수·분할·TP/FN/FP/TN·정밀도·재현율을 대조했습니다. 산출물 해시와 기존 입력·코드 해시를 확인했습니다.
- 학습 전용 처리는 평가 독립성 측면의 교정입니다. 이번 고정 분할에서는 위 정도의 지표 변화가 관찰됐지만, 영향이 항상 작다거나 통계적으로 동등하다는 검정 결과는 아닙니다.
- 기존에 선택된 모델·임계값과 이미 열람한 테스트를 사용한 민감도 비교입니다. 주문 단위·시간 분할 일반화 성능, 당시 카탈로그·지리정보의 가용성, 취소·미배송 모집단, 운영 효과는 해결하지 않았습니다.
- 기존 코드·CSV·데이터·NICE·포트폴리오는 그대로입니다. 새 결과를 기존 성능표나 UI에 반영하지 않았습니다. 본 전처리 경로 반영은 별도 변경 범위를 정한 뒤 진행합니다.

### 25.4 실험 코드 함수 설명

| 함수 | 입력 → 출력 | 설계 역할·핵심 규칙 |
|---|---|---|
| `distance` | 좌표 4열 → 거리 Series | 기존 Haversine 식과 6,371km 반경 유지 |
| `fit_imputation` | 학습 원본 행 → 통계 dict | 도시 최빈값·상품 중앙값·학습 거리 중앙값만 계산 |
| `transform_imputation` | 원본 행·통계 → 처리 DataFrame | 관측값 유지, 미등록 도시 좌표는 결측, 남은 거리만 학습 중앙값 대체 |
| `make_candidate` | 기존 모델 프레임·처리 결과 → 실험군 프레임 | 거리·상품과 거리 파생값만 갱신, 순서·자료형·나머지 값 불변 검사 |
| `check_imputation` | 원본·학습 인덱스·통계 → 검사 상태 dict | 고정 경계 예시와 holdout 교란 불변성 확인 |
| `measure` | 정답·긍정 확률·임계값 → 지표 dict | 기존 평가 함수와 위험 TP/FN/FP/TN, 확률 범위 검사 |
| `sha256` | 파일 경로 → 해시 문자열 | 입력 보존과 산출물 무결성 근거 |
| `main` | CLI 옵션·기존 파일 → 새 폴더, 반환 없음 | 사전 검증 후 4회 학습·저장 예측 재검산. `--validate-only`는 학습·산출물 쓰기 없음 |


## 26. A/B 본 학습 경로 수정 — 2026-10-01

사용자가 기존 코드 수정과 단계별 보고·Git push를 승인했습니다. 앞선 정비·실험을 `a1145c7`로 먼저 보존하고 이번 구현을 별도 커밋으로 분리합니다. 24·25절의 기존 코드 보존은 당시 실험 범위이며, 현재 A/B 경로에는 아래 변경이 적용됩니다.

### 변경과 설계 근거

전체 데이터에서 대체값을 구하면 평가 영역이 학습 입력에 영향을 줄 수 있습니다. 원본 값을 복원한 뒤 파이프라인의 `fit`에서 통계를 학습하도록 변경했습니다. 전역 분할뿐 아니라 탐색 CV의 각 fold에도 같은 원칙을 적용합니다. 기존 분할·라벨·피처·LightGBM 설정·검증 임계값 선택 규칙은 유지했습니다.

| 함수·메서드 | 입력 → 출력 | 역할·판단 규칙 |
|---|---|---|
| `attach_raw_imputation_columns` | 모델 프레임·원본 → 원본 보조 열 추가 프레임 | 복합키 일대일·인덱스 유일성·누락 검사, 행 순서 보존 |
| `haversine_distance` | 좌표 프레임 → km Series | 기존 반경 6,371km와 계산식 유지 |
| `TrainOnlyImputer.__init__` | 피처 열 → 추정기 | clone 가능한 고정 피처 명세 보관 |
| `_fill_coordinates` | 보조 열 포함 행 → 원본 좌표·상품 프레임 | 학습 도시 최빈값 적용, 미등록 도시 좌표는 결측 유지 |
| `fit` | 학습 행·선택적 y → self | 학습 도시 최빈값·상품 중앙값·좌표 처리 후 거리 중앙값 산출; 동률은 정렬 첫 값 |
| `transform` | 처리할 행 → 기존 피처 프레임 | 학습 통계만 적용, 기존 거리 경계·A 거리 파생식·자료형 유지, 원본 보조 열 제거 |
| `get_feature_names_out` | 선택적 입력 열 → 출력 열 배열 | 인코더에 기존 피처 이름 전달 |
| `prepare_corrected_model_frame` | 데이터·라벨 정책·선택적 원본 → 모델 프레임 | 라벨 정책 적용 후 원본 연결, 여기서는 통계 학습 없음 |
| `build_corrected_track_pipeline` | 입력 명세·A/B → Pipeline | 학습 전용 대체 후 기존 인코딩·분류기 연결 |
| `evaluate_track_a_vs_b` / `evaluate_track_b` / `evaluate_label_policies` | 기존 데이터·선택적 원본 → 평가표 또는 지표 | 수정 경로 사용, 검증에서 임계값 선택 유지 |

노트북 03의 전처리를 같은 경로에 연결하고 과거 학습 출력을 제거했습니다. 노트북 01은 EDA 스냅샷과 모델 전처리 차이에 관한 설명만 추가했습니다. 노트북의 데이터 덮어쓰기를 제거하고 노트북 비교표·생성 스크립트 출력은 실행별 폴더로 분리했습니다. 과거 CSV를 현재 정책의 재탐색 증거로 표현하던 주석도 교정했습니다. UI·Track C 학습 함수·기존 저수준 빌더는 유지했습니다.

### 검증 결과와 실제 학습

- 회귀 테스트 6개: 일대일 원본 연결, 도시 미등록·최빈값 동률·결측 처리, holdout 교란 불변성, fold별 독립 학습, 전체 피처의 앞선 실험 일치, 공개 평가 함수 연결을 확인했습니다.
- 노트북 관련 준비·분할 셀만 실행했습니다. 학습 변환 결과는 공용 코드와 원소별 일치했습니다(A 64,103×73, B 64,103×57, 인코딩 후 열 수). 전체 노트북·탐색·생성 스크립트는 실행하지 않았습니다.
- 실제 새 파이프라인 A/B 각 1회 학습을 완료했습니다. 앞선 실험의 train_only 검증·테스트 확률과 최대 차이 0이었습니다. 검증 선택 임계값도 A 약 0.61·B 약 0.54로 일치했습니다.
- 수정 후 테스트 정밀도/재현율/AP/균형 정확도: A 0.284381/0.672315/0.504897/0.679190, B 0.234436/0.616368/0.289564/0.621437입니다. 기존 표를 교체하지 않았습니다.
- 결과·확률·실행 스크립트·환경·해시는 `outputs/experiments/ab-pipeline-verification-20261001-101407/`에 보관합니다. 실행 후 노트북 주석만 교정한 이력은 manifest의 검증 당시 해시와 최종 해시로 구분합니다.

### 남은 범위

새 일반화 성능·최적 하이퍼파라미터·통계적 동등성을 주장하지 않습니다. 이미 확인한 테스트를 활용한 구현 재현 검증입니다. UI·Track C의 학습 전용 처리 이관과 UI 입력 계약·평가 대응은 다음 변경 범위를 먼저 설계합니다. 원천 위치 데이터·상품 카탈로그·예상 배송일의 당시 가용성, 표본 선택, 주문 단위·시간 분할 평가도 별도 과제입니다. 포트폴리오에는 전처리 검증 경험을 추가할 수 있으나 기존 수치를 자동 교체하지 않습니다.

최종 검사: Python 회귀 테스트 6개, Company-Research 테스트 92개, 노트북 코드 구문 검사와 `git diff --check`를 통과했습니다. 사전 해시 목록의 데이터·기존 결과표/메타데이터·포트폴리오 보호 대상 132개 파일이 일치했습니다.


## 27. UI·Track C 본 학습 경로 반영 — 2026-10-02 기록

사용자가 2026-10-01 검토안의 구현·전후 10회 학습·화면 검증·문서화·Git push를 승인했습니다. UI는 Streamlit 운영 콘솔 화면을 의미합니다. 실행과 비교 학습은 10월 1일 완료했으며 권한 안내 및 문서화를 10월 2일 이어갔습니다. 26절의 UI·Track C 미반영 상태는 아래 결과로 갱신합니다.

### 함수별 변경 근거와 입력 계약

| 함수 | 입력 → 출력 | 목적·핵심 규칙 |
|---|---|---|
| `distance_category` | 거리 Series → 구간 배열 | 기존 50·250·750·1,500km 이하 경계를 배치·화면에서 공유 |
| `transform_prepared_features` | 수동 입력 피처 → 처리 피처 | 입력 거리를 사용, 결측 상품·거리만 학습 통계 적용, 거리 구간 재계산. 가짜 좌표를 생성하지 않음 |
| `prepare_corrected_model_frame` | 데이터·정책·원본 → 원본 보조 열 포함 프레임 | `label_policy=None`은 C 평가의 기존 전체 표본 유지 |
| `build_corrected_track_pipeline` | A/B/C·분위수·선택 피처 → Pipeline | C도 학습 전용 대체를 적용. 선택 피처는 기존 모델 인코더에서 선택하며 대체 입력은 필요한 사전 피처 전체 사용 |
| `train_models` | 데이터·설정 → 학습 B/C | 입력 전체를 학습하는 기존 용도 유지. 그 자체가 holdout 평가가 아님 |
| `train_console_artifacts` | 데이터·설정 → 모델·테스트 아이템·고위험 주문 | 기존 80/20 주문 분할 유지. 예시에는 모든 테스트 아이템, 고위험 목록에는 주문별 최대 위험 아이템 표시 |
| `predict_order` | 학습 B/C·수동 피처 → 확률·배송일·대상 여부 | 모델의 수동 입력 처리 후 인코더·예측기 적용. 저장 원본 배치 경로와 구분 |
| `evaluate_track_c_quantiles` | 데이터·분위수 → 평가표 | 리뷰 3점도 포함하는 기존 C 표본 유지, train-only 대체 적용 |
| `make_recommendation_examples` | 데이터·설정 → 예시표 | 학습 통계로 교정한 피처와 예측 출력 |
| UI `build_input_row` | 입력 dict → 22개 피처 | 기존 파생식 유지, 거리 구간 자동 계산 |
| 비교 스크립트 `delivery_metrics` | 실제·예측·기존 예상일·분위수 → 지표 dict | MAE·pinball·coverage·전체 테스트 행의 검토값 시뮬레이션 |
| 비교 스크립트 `main` | 기존 파일·선택 캐시 경로 → 새 산출물 폴더 | 기준 커밋과 10회 학습, 분할·예측·환경·해시 기록 |

UI 학습 행 80,122/테스트 19,985, C 평가 학습 행 87,482/테스트 21,812를 보존했습니다. 대체 통계는 경로별 학습 표본에서 따로 구하며 거리 중앙값은 각각 428.997489km, 431.950088km입니다. 직접 피처 변경은 UI 학습49/테스트12, C 평가 학습43/테스트13행입니다. 하이퍼파라미터·임계값은 탐색하지 않았습니다.

### 결과·검증·수정 중 발견 사항

[비교 결과](../outputs/experiments/console-imputation-20261001-105834/comparison.md)에 10회 학습의 상세 수치가 있습니다. UI B 분류 변경은 0행, 확률 최대 차이는 약 0.001086, 고위험 주문은 양쪽 모두 6,923건입니다. C 90% 평가의 검토값 적용 후 3일 초과율은 3.5027%→3.5118%이며 실제 배송 속도 개선이 아닙니다. 고위험만의 효과나 인과 효과도 아닙니다.

Python 회귀 테스트 8개가 통과했습니다. 기존 A/B 검사와 새 경로·수동 결측·거리 경계·지역 열·화면 실행·아이템 선택·거리 변경을 포함합니다. 실제 학습 모델의 배치/수동/표시 확률 일치도 확인했습니다. UI 검사는 학습 완료 모델을 재사용했으며 비교 이후 추가 LightGBM 학습은 없습니다.

처음 UI 실행에서 `customer_state` 누락을 발견했습니다. 모델 프레임에는 없는 표시용 메타데이터를 원본에서 유지하고 교정 피처만 대입하도록 수정했습니다. 최종 함수의 화면 데이터 조립과 기존 예측의 일치를 확인했습니다. 처음 임계값 경계 테스트는 리터럴 0.46과 기존 `1 - 0.54`의 부동소수점 차이로 실패하여 원시 기본값과 화면 명시값을 각각 검사하도록 고쳤습니다. 운영 규칙은 변경하지 않았습니다.

### 미검증·다음 단계

브라우저 육안 검사는 컴퓨터 사용의 화면 기록 권한 대기로 미검증입니다. Streamlit AppTest 통과와 레이아웃 육안 확인을 구분합니다. UI의 다른 분위수 선택에 따른 실제 재학습은 실행하지 않았습니다. 기존 입력창·슬라이더 조건 및 캐시 구조를 유지했습니다.

원천 피처의 당시 가용성·표본 선택·주문 단위 성능·시간 분할·운영 비용 기반 임계값은 별도 검증 과제입니다. UI 기본 46%는 별도 A/B 평가에서 가져온 값으로 UI 모델의 최적 기준이 아닙니다. NICE·포트폴리오·기존 성능표는 그대로 보존합니다. 범용 포트폴리오에는 학습과 추론의 전처리 연결·결측 대체 검증 경험을 추가할 후보로 남기며 기존 수치를 자동 교체하지 않습니다.

최종 확인: Company-Research 테스트 92개·두 저장소 diff 검사 통과. 저장 예측에서 모든 비교 지표를 다시 계산해 일치 확인. 기존 데이터·결과표/메타데이터·포트폴리오 보호 대상 132개 해시 일치.


### 2026-10-03 권한 복구 후 브라우저 확인

재시작 후 Chrome 앱의 접근성 읽기·스크린샷·클릭·텍스트 입력이 성공했습니다. 확장 프로그램 기반 연결은 목록에 없지만 네이티브 컴퓨터 사용으로 게스트 창에서 로컬 앱을 확인했습니다. 임시 모델 캐시가 재시작으로 사라져 실제 앱 시작 시 기본 UI B/C 2개 모델을 다시 학습했습니다. 이는 앞선 10회 비교 실험과 별도의 화면 실행 확인입니다.

초기 화면에서 위험 확률 47.7%, 90% 배송 소요일 16.9일, 고위험 주문 6,923건을 확인했습니다. 거리를 60km로 변경하면 Short-Haul과 위험 확률 47.6%, 배송 소요일 16.4일로 갱신됐습니다. 확대된 창에서 한글·지표 카드·고위험 목록·하단 안내 표시를 직접 확인했습니다. 작은 창에서는 긴 카드 라벨이 말줄임 표시되며 넓은 표는 가로 스크롤이 필요한 한계가 있습니다. 모든 화면 크기나 분위수 선택의 재학습까지 검증한 것은 아닙니다. 앱 코드·기존 결과표는 이번 확인에서 수정하지 않았습니다.

## 28. 저장 예측의 주문 단위 평가 점검 — 2026-10-03

사용자의 후속 진행 요청에 따라 기존 저장 예측을 메모리에서 집계했습니다. 새 학습·피처 변경·임계값 선택·기존 성능표 교체는 수행하지 않았습니다. 20절의 주문 단위 피처 재구축과는 별개의 진단이며, 아이템 단위로 학습한 모델의 예측을 주문별로 평가한 것입니다.

### 표본·집계 규칙

A/B 테스트 및 UI B 테스트는 각각 19,985개 아이템 행·17,575개 주문입니다. 복수 아이템 주문은 1,706개, 주문당 최대 아이템 수는 14개입니다. 주문·아이템 복합키 중복은 없고 같은 주문 안의 라벨도 모두 일치했습니다. 위험 클래스는 리뷰 1·2점(라벨0), 리뷰3점은 제외합니다. 위험 아이템 행은 3,128개, 위험 주문은 2,380개입니다.

- 아이템 위험 점수는 `1 - positive_probability`입니다. 주 진단은 기존 UI의 목록 생성 규칙에 맞춰 주문별 최대값을 사용합니다. 한 상품이라도 기준을 넘으면 주문을 확인 대상으로 포함하는 의미입니다. 이 최대값을 보정된 주문 불만 확률로 해석하지 않습니다.
- 평균값은 집계 민감도 비교로만 사용합니다. 테스트 성능으로 최대/평균 중 유리한 규칙을 선택하지 않습니다. 아이템 수가 많은 주문은 최대값이 커질 기회도 많으므로 아이템 수별 평가가 후속 과제입니다.
- A/B는 저장된 `threshold_validation_selected`의 원시값을 유지합니다. 위험 기준은 A 약0.39, B 약0.46이며 `risk > 1 - positive_threshold`를 적용합니다. UI B는 기존 양성 기준0.54에서 계산한 위험 기준을 사용합니다. 주문 집계 후 임계값을 재탐색하지 않았습니다.

### 진단 결과

| 모델 | 평가·집계 | 정밀도 | 재현율 | AP | 확인 대상 수 |
|---|---|---:|---:|---:|---:|
| A | 아이템 행 | 0.284381 | 0.672315 | 0.504897 | 7,395행 |
| A | 주문 최대값 | 0.261760 | 0.671008 | 0.514989 | 6,101주문 |
| A | 주문 평균값 | 0.264358 | 0.657563 | 0.507980 | 5,920주문 |
| B | 아이템 행 | 0.234436 | 0.616368 | 0.289564 | 8,224행 |
| B | 주문 최대값 | 0.205342 | 0.591176 | 0.254582 | 6,852주문 |
| B | 주문 평균값 | 0.206246 | 0.574370 | 0.253466 | 6,628주문 |
| UI B | 아이템 행 | 0.232707 | 0.617327 | 0.294421 | 8,298행 |
| UI B | 주문 최대값 | 0.202513 | 0.589076 | 0.256994 | 6,923주문 |
| UI B | 주문 평균값 | 0.203836 | 0.576050 | 0.254374 | 6,726주문 |

B의 주문 최대값 기준 TP=1,407, FP=5,445, FN=973, TN=9,750입니다. UI B는 TP=1,402, FP=5,521, FN=978, TN=9,674이며 예전 UI 고위험 목록 6,923주문과 합계가 일치합니다. UI와 B는 학습 표본 수가 다른 모델이므로 지표를 혼용하지 않습니다.

분모와 가중치·집계 점수가 함께 달라지므로 아이템 성능과 주문 성능의 차이를 모델 성능의 하락이나 개선으로 단정하지 않습니다. 주문 기준에서는 B가 실제 위험 주문의 약59.1%를 탐지하며, 확인 대상으로 고른 주문 중 실제 위험 주문은 약20.5%입니다. 모델 피처에서 상품 정보를 제거한 결과가 아닙니다. 향후 주문 단위 모델을 학습한다면 모든 상품 정보를 명시적 집계 피처로 반영하는 별도 설계가 필요합니다.

### 재현 근거

입력 파일은 `outputs/experiments/ab-pipeline-verification-20261001-101407/predictions.csv`와 `outputs/experiments/console-imputation-20261001-105834/predictions.csv`입니다. 아래 코드는 저장 예측만 읽으며 파일을 쓰거나 모델을 학습하지 않습니다.

```python
import pandas as pd
from sklearn.metrics import precision_score, recall_score, average_precision_score

ab = pd.read_csv('outputs/experiments/ab-pipeline-verification-20261001-101407/predictions.csv', float_precision='round_trip')
ui = pd.read_csv('outputs/experiments/console-imputation-20261001-105834/predictions.csv', float_precision='round_trip')
sources = [(t, ab[(ab.track == t) & (ab.split == 'test')].copy()) for t in ['A', 'B']]
sources.append(('UI B', ui[(ui.population == 'UI') & (ui.variant == 'train_only')].copy()))
for name, x in sources:
    assert not x.duplicated(['order_id', 'order_item_id']).any()
    assert x.groupby('order_id').review_label.nunique().eq(1).all()
    x['risk'] = 1 - x.positive_probability
    threshold = 1 - (x.threshold_validation_selected.iloc[0] if name != 'UI B' else .54)
    for rule in ['item', 'max', 'mean']:
        if rule == 'item':
            truth, score = x.review_label.eq(0), x.risk
        else:
            truth = x.groupby('order_id').review_label.first().eq(0)
            score = x.groupby('order_id').risk.agg(rule)
        flag = score > threshold
        print(name, rule, precision_score(truth, flag), recall_score(truth, flag),
              average_precision_score(truth, score), int(flag.sum()))
```

### 시간 분할 전 확인과 다음 범위

모델 표본100,107행을 원본과 일대일 연결했으며 구매 시각 누락0, 주문 내 구매 시각 불일치0을 확인했습니다. 구매 기간은 2016-09-15 12:16:38부터 2018-08-29 15:00:37까지입니다. 시간 분할을 구성할 수 있으나 아직 기준일을 선택하거나 학습하지 않았습니다.

다음은 구매 시각뿐 아니라 학습 마감 시점에 배송·리뷰 정답이 확보돼 있었는지 확인하고, 기간별 표본·라벨 분포를 바탕으로 시간 분할안을 제시하는 것입니다. `review_answer_timestamp`의 정의·실제 라벨 가용성 대리변수 적합성부터 확인합니다. 당시 카탈로그·지리 정보의 완전한 시점 재현은 별도 한계입니다. 기존 테스트는 이미 반복 열람했으므로 미관측 최종 검증으로 주장하지 않습니다. 시간 분할 재학습의 이유·범위·산출물은 실행 전에 제시합니다.
