# Data

이 프로젝트의 원천 데이터는 Kaggle의 Brazilian E-Commerce Public Dataset by Olist를 사용합니다.

- Dataset: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
- Local raw data path: `data/raw/`
- Local processed data path: `data/processed/`

## Raw Files

Kaggle에서 데이터를 내려받은 뒤 아래 CSV 파일을 `data/raw/`에 둡니다.

```text
data/raw/
├── olist_customers_dataset.csv
├── olist_geolocation_dataset.csv
├── olist_order_items_dataset.csv
├── olist_order_payments_dataset.csv
├── olist_order_reviews_dataset.csv
├── olist_orders_dataset.csv
├── olist_products_dataset.csv
├── olist_sellers_dataset.csv
└── product_category_name_translation.csv
```

## Processed Files

`notebooks/01_preprocessing.ipynb`와 `notebooks/02_eda.ipynb`를 순서대로 실행하면 `data/processed/` 아래에 분석용 CSV가 생성됩니다.

```text
data/processed/
├── merged_data.csv
├── merged_final_data.csv
├── merged_train_data.csv
└── ml_data.csv
```

## Git Policy

원천 CSV와 가공 CSV는 용량이 커서 Git 추적 대상에서 제외합니다. 재현은 Kaggle 원본 데이터와 노트북 실행 순서로 관리합니다.
