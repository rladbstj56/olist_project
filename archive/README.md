# Archive

`archive/`는 최종 재현 기준이 아니라 프로젝트 진행 과정과 팀 작업 맥락을 보존하는 공간입니다.

면접관이나 리뷰어가 먼저 확인해야 하는 파일은 루트 [README.md](../README.md), [notebooks](../notebooks), [outputs](../outputs), [docs](../docs)입니다.

## Folder Guide

| 폴더 | 의미 |
| --- | --- |
| `legacy_root_notebooks/` | 독립 저장소로 정리하기 전 루트에 있던 초기·중간 노트북 |
| `reference_models/` | 최종 모델로 채택하지 않은 참고 모델 |
| `team_work/` | 팀 프로젝트 당시 전처리, EDA, ML 작업 기록 |

## Review 기준

- 최종 실행 기준: `notebooks/01_preprocessing.ipynb`부터 `notebooks/03_ml_classifier.ipynb`까지
- 최종 결과 기준: `outputs/charts/`, `outputs/tables/`, `outputs/metadata/`
- 판단 근거 문서: `docs/modeling_decision_log.md`

Archive 파일은 프로젝트 발전 과정을 설명할 때만 참고합니다.
