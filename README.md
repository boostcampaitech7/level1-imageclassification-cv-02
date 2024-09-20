# Sketch Image Classification

# 설치 방법

## git 설치
```bash
apt update
apt install -y git
git clone https://github.com/boostcampaitech7/level1-imageclassification-cv-02.git
cd level1-imageclassification-cv-02
```

## 데이터셋 설치
```bash
apt-get install wget
wget https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000307/data/data.tar.gz
tar -zxvf data.tar.gz
```

## Python package 설치
이 프로젝트는 Poetry를 사용하여 의존성을 관리합니다. 설치 방법은 다음과 같습니다:

```bash
# Poetry 설치
python -m venv competition1
source competition1/bin/activate
pip install poetry
```

### 1. pyproject.toml 파일 수정
- python="^3.10"으로 수정. (3.12 설치 안됨)
### 2. poetry 의존성 update
```bash
poetry lock
```
### 3. 설치
```bash
poetry install
```
- 추가로 설치한 패키지들도 있어서, pip도 해주기
```bash
pip install -r requirements.txt
```


## 사용 방법
### Train
- **config 파일 수정 잘 해서 돌리기**
```bash
python train.py --config configs/train_configs/train/config.yaml
```

### Test (Inference)
- **config 파일 수정 잘 해서 돌리기**
```bash
python test.py --config configs/train_configs/test/config.yaml
```

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
│   ├── train/
│   ├── test/
│   ├── train.csv
│   └── test.csv
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
