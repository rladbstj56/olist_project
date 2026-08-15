# Olist Project 대화 및 작업 맥락 기록

작성일: 2026-07-27  
프로젝트 최종 위치: `/Users/yoonseokim/olist_project`  
GitHub repo: `https://github.com/rladbstj56/olist_project`

## 1. 이 문서의 목적

이 문서는 기존 `2025_main_bootcamp/2nd_advanced_project` 경로에서 진행하던 작업이 `olist_project` 독립 repo로 분리되면서, 대화에서 결정한 분석 기준과 수정 이력이 사라지지 않도록 남긴 기록이다.

실제 채팅 스레드 자체를 파일 시스템으로 옮길 수는 없으므로, 이 문서에는 대화에서 합의한 핵심 판단, 수정 사항, repo 이전 과정, 남은 주의사항을 정리했다.

## 2. 최종 프로젝트 구조 결정

기존 구조:

```text
/Users/yoonseokim/2025_main_bootcamp/2nd_advanced_project
```

최종 구조:

```text
/Users/yoonseokim/olist_project
```

최종 GitHub 구조:

```text
rladbstj56/olist_project
```

기존 `2025-main-bootcamp` repo 안의 `2nd_advanced_project` 폴더는 삭제했고, `olist_project`를 독립 repo로 유지하기로 했다.

## 3. repo 이전 작업 기록

### 3.1 히스토리 분리

처음에는 `git subtree split --prefix=2nd_advanced_project`로 기존 `2nd_advanced_project` 폴더의 git history를 분리했다.

이 방식은 기존 커밋 메시지를 보존한다는 장점이 있다. 다만 당시 git에 실제로 추적되고 있던 파일은 `code_final_report`, `data/original_dataset.zip` 중심이었기 때문에, 과거 팀 작업물이 충분히 복원되지 않았다.

### 3.2 팀 작업물 복원

GitHub 과거 화면에는 아래 폴더와 파일이 있었지만, 최신 `main`에서는 삭제된 상태였다.

- `analytical_thinking`
- `code_eda`
- `code_ml`
- `code_preprocessed`
- `script`
- 루트 노트북
  - `eda_final.ipynb`
  - `eda_통합_머신러닝.ipynb`
  - `ml.ipynb`
  - `ml_shared.ipynb`
  - `ml_shared copy.ipynb`
  - `preprocessed_final.ipynb`
- `image.png`
- 발표 PDF
- `data/others`

따라서 과거 커밋 `94acf74`에서 해당 작업물들을 복원해 `olist_project`에 새 커밋으로 추가했다.

복원 커밋:

```text
e9ef745 Restore team project artifacts
```

### 3.3 제외한 파일

`analysis_guide`는 사용자가 필요 없다고 판단해 제외했다.

제외 이유:
- 현재 Olist 최종 프로젝트와 직접 관련 없는 H&M 분석 가이드 성격
- 일부 파일이 GitHub 일반 push 제한에 걸릴 정도로 큼
  - `analysis_guide/processed_data/df_full.csv`: 약 511MB
  - `analysis_guide/data/transactions_hm.csv`: 약 101MB

추가한 `.gitignore` 기준:

```gitignore
.DS_Store
.venv/
__pycache__/
.ipynb_checkpoints/
.claude/
analysis_guide/
```

## 4. GitHub 작업 기록

### 4.1 `olist_project`

최종 push 완료:

```text
e9ef745 Restore team project artifacts
```

GitHub push 중 50MB 초과 파일에 대한 경고는 있었지만, 100MB 초과 파일은 없어 push는 성공했다.

경고 대상 예:
- `code_final_report/finalcode/data/merged_data.csv`: 약 53MB
- `code_final_report/finalcode/data/olist_geolocation_dataset.csv`: 약 58MB

이는 GitHub 권장 크기 50MB를 넘지만, 하드 제한 100MB 미만이므로 push 가능했다.

### 4.2 `2025-main-bootcamp`

`2nd_advanced_project`를 독립 repo로 이전한 뒤 기존 repo에서 삭제했다.

삭제 커밋:

```text
9991bfc Remove migrated second advanced project
```

기존 로컬 폴더도 삭제했다.

삭제된 경로:

```text
/Users/yoonseokim/2025_main_bootcamp/2nd_advanced_project
```

보존된 새 경로:

```text
/Users/yoonseokim/olist_project
```

## 5. EDA 검증 및 수정 기록

대상 파일:

```text
code_final_report/finalcode/eda_final.ipynb
```

주요 수정 사항:

- `merged_final_data.csv`가 없을 경우 `preprocessed_final.ipynb`를 먼저 실행해야 한다는 안내 추가
- train/test 분할에서 `stratify=y` 적용
- `delay_days` 부호와 해석 설명 수정
- 인과처럼 보이는 표현 완화
- 리뷰 점수 차이에 대한 과도한 배수 표현 제거
- `data/merged_train_data.csv` 생성 확인

판단 기준:
- EDA는 모델링보다 앞 단계이므로 데이터 정의와 해석 문구가 정확해야 한다.
- 상관관계나 그룹 차이를 인과로 표현하면 분석 신뢰도가 떨어진다.
- 클래스 불균형이 있는 타깃은 stratify로 분포를 유지해야 평가가 안정적이다.

## 6. 전처리 검증 및 수정 기록

대상 파일:

```text
code_final_report/finalcode/preprocessed_final.ipynb
```

주요 확인 사항:

- Olist 원천 CSV 로드
- `product_category_name_translation.csv` 확보
- `data/merged_final_data.csv` 생성
- 시간 순서 이상치 처리
- 리뷰 중복 축약
- 최종 모델링용 데이터 생성

## 7. ML 분류 모델 검증 및 수정 기록

대상 파일:

```text
code_final_report/finalcode/ml_classifier.ipynb
```

주요 수정 사항:

- `lightgbm` 의존성 반영
- `requirements.txt` 추가
- `nbclient`, `nbformat` 추가
- `safe_divide` 함수 추가
- `data/ml_data.csv` 저장 시 `index=False` 적용
- 타깃 설명을 “NPS 기준”이 아니라 “CS 리스크 기준”으로 수정
- 행 단위 split을 `order_id` 기준 group split으로 변경
- 하이퍼파라미터 탐색 CV를 `StratifiedGroupKFold`로 변경
- Track A 3개 모델 전체 재탐색
- Track B LightGBM 하이퍼파라미터 재탐색
- Track B에 상품 무게/크기 컬럼 추가
- `OneHotEncoder(handle_unknown='ignore', sparse_output=True)` 유지
- 최신 모델 결과 CSV와 해석 표 갱신
- 전체 노트북 실행 검증 완료

## 8. Track A와 Track B 결정

### 8.1 Track A

Track A는 사후 원인분석 모델이다.

사용 시점:
- 배송 완료 후

목적:
- 부정 리뷰 발생 원인 진단
- CS 우선순위 판단
- 보상/환불 정책 참고

사용 가능 정보:
- 배송 완료 후에 확정되는 배송 결과 컬럼 포함 가능

예:
- `delivery_days`
- `delay_days`
- `is_delayed`
- `delay_days_cat`
- `delivery_speed`
- `delivery_ratio`
- `delivery_distance`
- `delivery_price`
- `day_per_km`

### 8.2 Track B

Track B는 사전 예측 모델이다.

사용 시점:
- 주문 확정 또는 배송 시작 전

목적:
- 고위험 주문 사전 선별
- 선제 안내
- 배송 우선 처리

사용 가능 정보:
- 주문 시점에 이미 알 수 있는 정보만 사용

배송 완료 후 정보는 제외한다.

### 8.3 상품 무게/크기 컬럼 판단

Track B에 추가한 컬럼:

- `product_weight_g`
- `product_length_cm`
- `product_height_cm`
- `product_width_cm`

판단 기준:
- 상품 카탈로그 정보이므로 주문 시점에 이미 알 수 있다.
- 배송 완료 후 생성되는 미래 정보가 아니므로 data leakage가 아니다.
- 무게와 크기는 배송 난이도, 배송비, 파손 가능성, 지연 가능성과 연결될 수 있다.

Track A에서도 해당 컬럼은 이미 사용 중이다.

## 9. Data Leakage 방지 기준

현재 최종 데이터의 한 행은 주문 1건이 아니라 주문-아이템 1건이다.

정확한 인스턴스 기준:

```text
order_id + order_item_id
```

문제:
- 하나의 주문에 여러 아이템이 있으면 같은 `order_id`가 여러 행에 반복된다.
- 리뷰 점수는 `order_id` 단위로 붙는다.
- 행 단위 split을 하면 같은 주문의 일부 아이템이 train에, 다른 아이템이 test에 들어갈 수 있다.
- 이 경우 모델이 사실상 같은 주문 정보를 학습한 뒤 평가받게 된다.

수정 기준:
- 모델 입력에서는 `order_id`를 제외한다.
- split과 CV에서는 `order_id`를 group 기준으로 사용한다.

적용:

- train/valid/test 분리: `GroupShuffleSplit`
- 하이퍼파라미터 탐색: `StratifiedGroupKFold`

## 10. 인스턴스 및 병합 기준

최종 데이터 `merged_final_data.csv`는 주문-아이템 단위다.

병합 흐름:

```python
df_final = pd.merge(df_orders, df_customers_geo, on="customer_id", how="left")
df_final = pd.merge(df_final, df_items, on="order_id", how="left")
df_final = pd.merge(df_final, df_reviews, on="order_id", how="left")
df_final = pd.merge(df_final, df_products, on="product_id", how="left")
df_final = pd.merge(df_final, df_sellers_geo, on="seller_id", how="left")
```

핵심:
- `orders`에 `order_items`를 붙이는 순간 데이터 단위가 주문 단위에서 주문-아이템 단위로 확장된다.
- 상품 카테고리, 무게, 크기, 셀러, 가격, 배송비는 아이템 단위 정보이므로 이 기준이 분석 목적에 더 적합하다.

## 11. 리뷰 중복 축약 기준과 한계

리뷰 테이블은 `order_id` 기준으로 병합했다. 같은 `order_id`에 리뷰가 여러 개 있는 경우, 최신 리뷰 1건만 남겼다.

```python
df_reviews = df_reviews.sort_values(by=['order_id', 'review_answer_timestamp'])
df_reviews = df_reviews.drop_duplicates(subset=['order_id'], keep='last')
```

처리 이유:
- 리뷰를 중복 그대로 두면 `orders × items × reviews` 형태로 행이 증식된다.
- 특정 주문이 리뷰 수만큼 과대표집될 수 있다.
- 따라서 row explosion을 막기 위해 주문당 대표 리뷰 1건을 선택했다.

한계:
- 최신 리뷰가 반드시 대표 리뷰라고 보장할 수 없다.
- 다중 상품 주문에서 리뷰가 특정 상품에 대한 평가인지, 전체 주문 경험에 대한 평가인지 구분할 수 없다.
- 상품 단위 원인 해석에는 주의가 필요하다.

## 12. 시간 순서 이상치 처리

일반적인 시간 순서:

```text
order_purchase_timestamp
→ order_approved_at
→ order_delivered_carrier_date
→ order_delivered_customer_date
```

검증 및 처리:

| 검증 조건 | 의미 | 처리 |
|---|---|---|
| `purchase > approved` | 구매 시각이 승인 시각보다 늦음 | 발견 0건 |
| `approved > carrier` | 승인 시각이 택배사 인도 시각보다 늦음 | 제거하지 않음 |
| `carrier > delivered` | 택배사 인도 시각이 고객 도착 시각보다 늦음 | 23건 제거 |
| `delivered < purchase` | 고객 도착 시각이 구매 시각보다 빠름 | 발견 0건 |
| `dispatch_days < 0` | 구매 후 출고 소요일이 음수 | 2행 제거 |

판단 기준:
- `carrier > delivered`는 물류 흐름상 성립하기 어려워 삭제했다.
- `approved > carrier`는 결제 승인 기록 지연 가능성이 있어 삭제하지 않았다.
- `dispatch_days < 0`은 파생 피처 해석을 깨뜨리므로 최종 제거했다.

## 13. 모델 재탐색 결과

Track A 모델 비교:

| model | cv_best_macro_f1 | threshold | neg_recall | balanced_acc | macro_f1 | accuracy |
|---|---:|---:|---:|---:|---:|---:|
| LightGBM | 0.646 | 0.57 | 0.625 | 0.636 | 0.592 | 0.642 |
| RandomForest | 0.648 | 0.57 | 0.623 | 0.633 | 0.588 | 0.638 |
| XGBoost | 0.616 | 0.82 | 0.614 | 0.620 | 0.575 | 0.623 |

최종 Track A 모델:

```text
LightGBM
```

선택 이유:
- RandomForest가 CV macro F1은 아주 근소하게 높았다.
- 하지만 validation 기준 `negative recall`, `balanced_accuracy`, `macro_f1`은 LightGBM이 더 좋았다.
- 프로젝트 목적이 부정 리뷰 탐지이므로 validation 목적 지표를 더 우선했다.

## 14. 최종 성능

| track | n_features | accuracy | balanced_acc | neg_recall | macro_f1 |
|---|---:|---:|---:|---:|---:|
| Track A LightGBM, 사후 원인분석 | 33 | 0.652 | 0.651 | 0.648 | 0.603 |
| Track B LightGBM, 사전 예측 | 22 | 0.600 | 0.599 | 0.597 | 0.553 |
| Baseline, 다수 클래스 | 0 | 0.767 | 0.500 | 0.000 | 0.434 |

해석:
- baseline은 accuracy가 높지만 부정 리뷰를 하나도 잡지 못한다.
- Track B는 accuracy는 낮지만 부정 리뷰 탐지 관점에서는 baseline보다 훨씬 낫다.
- Track A는 배송 결과 정보를 쓰기 때문에 Track B보다 성능이 높은 것이 자연스럽다.

## 15. 주요 산출 파일

최종 분석 문서:

```text
code_final_report/finalcode/modeling_decision_log.md
docs/conversation_context_log.md
```

모델 결과:

```text
code_final_report/finalcode/outputs/tables/model_comparison_results.csv
code_final_report/finalcode/outputs/tables/track_a_vs_b_comparison.csv
code_final_report/finalcode/outputs/metadata/refreshed_search_summary.json
```

의존성:

```text
code_final_report/finalcode/requirements.txt
```

## 16. 남은 주의사항

`2025_main_bootcamp` 상위 repo에는 이 작업과 무관한 변경이 남아 있었다.

- `.gitignore`
- `3rd_practice_project`
- `4th_final_project` 쪽 일부 output JSON

이 변경들은 `olist_project` 이전 및 삭제 커밋에는 포함하지 않았다.

`olist_project`에는 50MB가 넘는 CSV가 일부 들어 있다. GitHub push는 성공했지만, 장기적으로는 대용량 데이터는 외부 저장소나 release asset, 또는 Git LFS로 분리하는 것이 더 좋다.

## 17. 면접/포트폴리오 핵심 설명

이 프로젝트에서 강조할 포인트:

- 단순히 모델을 만든 것이 아니라, 모델 사용 시점에 따라 Track A와 Track B를 분리했다.
- 사후 원인분석과 사전 예측을 구분해 feature leakage를 통제했다.
- 주문-아이템 단위 데이터 구조에서 `order_id` 기반 group split으로 data leakage를 막았다.
- 리뷰 granularity와 상품 granularity 불일치를 인식하고 한계로 명시했다.
- accuracy만 보지 않고 `negative recall`, `balanced_accuracy`, `macro_f1`을 중심으로 불균형 분류 문제를 평가했다.
- 기존 팀 작업 과정과 최종 정리본을 모두 보존하기 위해 `olist_project`를 독립 repo로 분리했다.

## 18. Track B 운영 활용과 Track C 확장 아이디어

작성일: 2026-08-15

이번 대화에서는 현재 사전 예측 모델 Track B를 운영자가 실제로 어떻게 활용할 수 있는지, 그리고 그 결과를 기반으로 예상 배송일 추천 모델 Track C로 확장할 수 있는지 검토했다.

### 18.1 현재 Track B feature 확인

`notebooks/03_ml_classifier.ipynb` 기준으로 Track B는 주문 시점에 이미 알 수 있는 22개 feature를 사용한다. 이 안에는 `expected_delivery_days`가 포함되어 있다.

Track B feature:

- `order_item_id`
- `price`
- `freight_value`
- `freight_ratio`
- `total_price`
- `product_weight_g`
- `product_length_cm`
- `product_height_cm`
- `product_width_cm`
- `order_purchase_dayofweek`
- `order_purchase_month`
- `expected_delivery_days`
- `main_category`
- `sub_category`
- `distance_km`
- `distance_cat`
- `cross_state`
- `is_sp_customer`
- `is_sp_seller`
- `sp_route_type`
- `sp_route_type_customer`
- `sp_route_type_seller`

따라서 현재 Track B는 “판매자 또는 플랫폼이 설정한 기존 예상 배송일을 포함해, 해당 주문이 부정 리뷰로 이어질 위험이 높은지 사전에 판단하는 모델”로 해석할 수 있다.

### 18.2 Track B의 운영 활용 해석

Track B는 리뷰 점수를 정확히 맞히는 모델이라기보다, 주문 직후 운영자가 먼저 확인해야 할 위험 주문을 좁혀주는 조기경보 모델이다.

운영 흐름:

```text
주문 발생
→ 주문 시점 feature 입력
→ Track B로 부정 리뷰 위험 확률 예측
→ 고위험 주문 선별
→ 출고 확인, 배송 모니터링, 고객 선제 안내, 보상 후보 등록 등 운영 액션 연결
```

위험도별 운영 액션 예시는 다음과 같다.

| 위험도 | 운영 액션 | 목적 |
| --- | --- | --- |
| 낮음 | 일반 처리 | 운영 리소스 절약 |
| 중간 | 배송 상태 모니터링 강화 | 문제 발생 전 조기 감지 |
| 높음 | 판매자 출고 확인, 고객 선제 안내 | 지연 불만 예방 |
| 매우 높음 | 우선 출고 요청, CS 알림, 쿠폰·보상 후보 등록 | 부정 리뷰 전환 방지 |

핵심 해석은 다음이다.

- Track B는 모든 주문에 동일한 조치를 하자는 모델이 아니다.
- 운영 리소스가 제한되어 있으므로 위험 확률 상위 주문부터 우선 관리하는 것이 현실적이다.
- 특히 4일 이상 지연 주문의 평균 리뷰 점수가 1–3일 지연 주문보다 유의하게 낮았으므로, Track B 고위험 주문은 “3일 초과 지연으로 넘어가기 전에 개입해야 할 후보”로 볼 수 있다.

### 18.3 Track C 확장 아이디어

사용자가 제안한 확장 방향은 다음과 같다.

```text
주문 발생
→ 기존 예상 배송일을 포함해 Track B 실행
→ 부정 리뷰 위험이 높다고 예측된 주문 선별
→ 고위험 주문에만 Track C 실행
→ 해당 주문에 더 적절한 추천 예상 배송일 산출
→ 특별 조치로 예상 배송일 조정 또는 고객 선제 안내에 반영
```

이 아이디어의 장점은 모든 주문의 예상 배송일을 일괄적으로 늘리지 않는다는 점이다. Track B에서 위험하다고 판단된 주문에만 Track C를 추가 실행하면, 고객에게 불필요하게 긴 배송 약속을 제시하는 문제를 줄이면서도 고위험 주문에는 더 보수적인 배송 약속을 적용할 수 있다.

Track C의 핵심 질문:

```text
이 주문이 4일 이상 지연 구간으로 넘어가지 않으려면 예상 배송일을 며칠로 제안하는 것이 적절한가?
```

또는 운영 관점에서는 다음 질문으로 바꿀 수 있다.

```text
현재 예상 배송일로는 리뷰 리스크가 높으니, 이 주문은 예상 배송일을 몇 일로 보수적으로 조정해야 하는가?
```

### 18.4 Track C 모델 설계 후보

Track C는 현재 Track B와 목표 변수가 다르다. Track B는 부정 리뷰 여부를 분류하지만, Track C는 예상 배송일 추천을 위해 실제 배송 소요일 또는 지연 위험을 예측해야 한다.

가장 자연스러운 설계:

```text
Track C target = delivery_days
입력 feature = 주문 시점에 이미 알 수 있는 정보
출력 = 주문별 예상 실제 배송 소요일
추천 예상 배송일 = 예측 배송 소요일 + 안전 여유일
```

더 보수적인 방식으로는 평균 예측이 아니라 분위수 예측을 사용할 수 있다.

```text
recommended_expected_days = ceil(predicted_delivery_days_90_percentile)
```

90% 분위수는 통계적으로 고정된 정답 임계값이 아니라, 고위험 주문에는 평균보다 보수적인 배송 약속을 제시한다는 초기 운영 기준이다. 평균 예측은 일반적인 배송 소요일을 맞히는 데는 적합하지만, 늦어지는 일부 주문의 tail risk를 충분히 반영하지 못할 수 있다. Track C의 목적은 평균 배송일을 맞히는 것보다 부정 리뷰로 이어질 수 있는 큰 지연 리스크를 줄이는 데 있으므로, 90% 분위수를 기본안으로 둔다.

다만 실제 운영에서는 80%, 85%, 90%, 95% 분위수 기준을 비교해 최종 기준을 조정해야 한다.

| 분위수 기준 | 장점 | 주의점 |
| --- | --- | --- |
| 80% | 예상 배송일이 과하게 길어지는 문제를 줄임 | 지연 불만이 여전히 남을 수 있음 |
| 90% | 지연 리스크 관리와 고객 기대 관리의 균형 | 일부 주문은 기존 예상일보다 길게 제안될 수 있음 |
| 95% | 지연 가능성을 더 낮춤 | 배송 약속이 지나치게 길어져 구매 전환율에 불리할 수 있음 |

따라서 포트폴리오와 면접에서는 “90% 분위수는 초기 운영 가정이며, 실제 서비스에서는 지연 감소 효과와 예상 배송일 증가에 따른 전환 손실을 함께 비교해 조정해야 한다”고 설명한다.

또는 프로젝트의 3일 초과 지연 기준을 직접 반영하면 다음과 같은 정책도 가능하다.

```text
recommended_expected_days = ceil(predicted_delivery_days_90_percentile - 3)
```

첫 번째 방식은 지연 자체를 줄이는 보수적 정책이고, 두 번째 방식은 리뷰 점수가 급락한 “4일 이상 지연” 구간을 피하는 데 초점을 둔 정책이다.

### 18.5 포트폴리오 표현 방향

Track C는 아직 구현 완료 모델이 아니라, 현재 분석 결과에서 자연스럽게 이어지는 운영 확장 제안이다. 따라서 포트폴리오에서는 다음처럼 표현하는 것이 적절하다.

> 사전예측모델 Track B로 부정 리뷰 위험 주문을 선별하고, 고위험 주문에 대해서는 Track C 배송 소요일 예측 모델을 추가 적용해 보수적인 예상 배송일을 추천하는 운영 전략을 제안했다. 특히 4일 이상 지연 구간에서 리뷰 점수가 유의하게 하락한다는 통계검정 결과를 Track C의 추천 기준으로 연결했다.

면접에서 강조할 포인트:

- 예측 모델을 만드는 데서 끝나지 않고, 예측 결과를 운영 액션으로 연결했다.
- Track B는 “어떤 주문이 위험한가”를 답하고, Track C는 “위험 주문에 어떤 예상 배송일을 제안할 것인가”를 답한다.
- 3일 초과 지연 기준은 단순 직관이 아니라 주문 단위 통계검정 결과에 기반한다.
- 아직 실제 정책 효과는 A/B 테스트나 파일럿 운영으로 검증해야 하므로, 인과 효과로 단정하지 않는다.
