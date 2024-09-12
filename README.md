# BOOSTCAMP_BASELINE

## 설치 방법
이 프로젝트는 Poetry를 사용하여 의존성을 관리합니다. 설치 방법은 다음과 같습니다:

```bash
# Poetry 설치
pip install poetry

# 프로젝트 의존성 설치
poetry install
```

## 사용 방법

## 프로젝트 구조
```
project-root/
├── LICENSE
├── README.md
├── configs/
│   ├── data_configs/
│   ├── ensemble_configs/
│   ├── loss_configs/
│   ├── model_configs/
│   └── optimizer_configs/
├── data/
│   ├── processed/
│   └── raw/
├── notebooks/
│   ├── eda/
│   ├── model_exploration/
│   └── results_analysis/
├── poetry.lock
├── pyproject.toml
├── results/
│   ├── ensembles/
│   └── individual_models/
├── scripts/
│   ├── ensemble_predict.py
│   ├── evaluate.py
│   ├── predict.py
│   └── train.py
├── src/
│   ├── data/
│   ├── ensemble/
│   ├── experiments/
│   ├── loss_functions/
│   ├── models/
│   ├── optimizers/
│   └── utils/
└── tests/
```

### 주요 디렉토리 설명
- `configs/`: 다양한 구성 요소에 대한 설정 파일
- `data/`: 원본 및 전처리된 데이터 저장
- `notebooks/`: 분석 및 탐색을 위한 주피터 노트북
- `results/`: 모델 결과 출력 디렉토리
- `scripts/`: 훈련, 평가 등을 위한 유틸리티 스크립트
- `src/`: 주요 소스 코드
- `tests/`: 다양한 구성 요소에 대한 단위 테스트
