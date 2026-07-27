# Olist Customer Satisfaction Risk Analysis

Brazilian Olist e-commerce 데이터를 기반으로 고객 불만족 리뷰를 탐지하기 위한 EDA, 전처리, 분류 모델링 프로젝트입니다.

## Project Structure

```text
olist_project/
├── notebooks/              # 최종 실행 노트북
├── data/
│   ├── raw/                # 원본 Olist CSV
│   └── processed/          # 전처리 및 모델링용 생성 CSV
├── outputs/                # 모델 비교 및 검증 결과
├── docs/                   # 의사결정, 한계점, 대화 맥락 기록
├── archive/                # 과거 팀 작업물과 중간 노트북
├── analytical_thinking/    # 부트캠프 학습 자료
├── script/                 # 발표 스크립트
└── requirements.txt
```

## Main Notebooks

1. `notebooks/01_preprocessing.ipynb`
2. `notebooks/02_eda.ipynb`
3. `notebooks/03_ml_classifier.ipynb`
4. `notebooks/04_ml_regressor.ipynb`

## Data Files

- `data/raw/`: 원본 Olist 데이터와 상품 카테고리 번역 테이블
- `data/processed/merged_data.csv`: 1차 병합 데이터
- `data/processed/merged_final_data.csv`: EDA와 ML 공통 최종 분석 데이터
- `data/processed/merged_train_data.csv`: EDA 이후 모델링 후보 데이터
- `data/processed/ml_data.csv`: 모델링 직전 기준 데이터 스냅샷

## Archive Policy

`archive/`는 삭제 대상이 아니라 프로젝트 발전 과정을 보존하는 공간입니다.

- `archive/team_work/`: 팀원별 EDA, 전처리, ML 작업 과정
- `archive/legacy_root_notebooks/`: 현재 최종 노트북의 조상 또는 중간본

최종 재현과 검토는 `notebooks/`, `data/`, `outputs/`, `docs/`를 기준으로 합니다.
