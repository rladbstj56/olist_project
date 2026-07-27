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
code_final_report/finalcode/outputs/model_comparison_results.csv
code_final_report/finalcode/outputs/track_a_vs_b_comparison.csv
code_final_report/finalcode/outputs/refreshed_search_summary.json
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
