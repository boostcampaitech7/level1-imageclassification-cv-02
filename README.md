# Sketch Image Classification

# 설치 방법

## git 설치
```bash
apt-get update -y
apt-get install -y libgl1-mesa-glx #OpenGL (Open Graphics Library) 그래픽 API를 지원하는 라이브러리
apt-get install -y libglib2.0-0 #GLib 라이브러리는 다양한 시스템 및 플랫폼에서 사용되는 기본 라이브러리 
apt install -y git
git clone https://github.com/boostcampaitech7/level1-imageclassification-cv-02.git
cd level1-imageclassification-cv-02
```

## 데이터셋 설치
```bash
apt-get install wget
wget https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000307/data/data.tar.gz
tar -zxvf data.tar.gz 
rm -rf data.tar.gz # data 압축 다 풀었으면 .gz 파일 삭제.
cd data #dataset 폴더에 들어가줌
find . -name '._*' -type f -delete # ._으로 시작하는 파일 지워줌(숨김파일이거나 캐싱파일이라 필요없음.)
cd .. # dataset 폴더에서 나오고 level1폴더로 다시 회귀.
rm -rf ._data #._data라는 캐싱파일 삭제.
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
pip install -r settings/requirements.txt #requirements.txt가 settings에 있음. m
```


## 사용 방법
### Train
- **config 파일 수정 잘 해서 돌리기**
```bash
python train.py --config configs/train_configs/train/config.yaml
```
만약에 `No module names 'src'`라고 뜬다면(현재 경로를 제대로 입력받지 않은것이니,)
홈 디렉토리에 `.bashrc`만들고 그 안에 `export PYTHONPATH=$(pwd)`라고  입력후 저장하면 됨.

### Test (Inference)
- **config 파일 수정 잘 해서 돌리기**
```bash

python test.py --config configs/train_configs/test/config.yaml
```

## 프로젝트 구조
```
.
|-- competition1
|   |-- bin
|   |-- include
|   |-- lib
|   |-- lib64 -> lib
|   |-- pyvenv.cfg
|   `-- share
|-- configs
|   |-- augmentation_configs
|   |-- data_configs
|   |-- ensemble_configs
|   `-- train_configs
|-- data
|   |-- sample_submission.csv
|   |-- test
|   |-- test.csv
|   |-- train
|   `-- train.csv
|-- output # 본인의 결과물 csv, checkpoint, 가 자동으로 저장됨
|   `-- lightning_logs
|-- poetry.lock
|-- pyproject.toml
|-- pytest.ini
|-- settings
|   |-- LICENSE
|   |-- README.md
|   `-- requirements.txt
|-- src  # dataset, train에 필요한 요소들(loss, optimizer, scheduler,model 등)이 있습니다.
|   |-- data
|   |-- ensemble
|   |-- experiments
|   |-- loss_functions
|   |-- models
|   |-- optimizers
|   |-- plmodules
|   |-- scheduler
|   `-- utils
|-- test.py
|-- tests
|   |-- __init__.py
|   |-- conftest.py
|   |-- test_datamodules.py
|   |-- test_ensemble_predict.py
|   |-- test_ensembles.py
|   |-- test_losses.py
|   |-- test_models.py
|   `-- test_optimizers.py
`-- train.py
```

### 주요 디렉토리 설명
- `configs`
다양한 설정 파일들을 포함하는 디렉토리입니다.
데이터 증강, 데이터셋, 앙상블, 학습 등에 관한 설정 파일들이 포함되어 있습니다.

- `output`
프로젝트의 결과물인 CSV 파일들을 저장하는 디렉토리입니다.

- `settings`
프로젝트의 의존성 관리와 라이선스 관련 파일들이 위치한 디렉토리입니다.
Poetry를 사용한 의존성 관리 파일(pyproject.toml)과 라이선스 파일이 포함되어 있습니다.

- `src`
프로젝트의 핵심 소스 코드가 위치한 디렉토리입니다.
데이터셋, 모델, 손실 함수, 옵티마이저, 스케줄러 등 학습에 필요한 모든 요소들이 포함되어 있습니다.

- `tests`
프로젝트의 테스트 코드가 위치한 디렉토리입니다.
다양한 컴포넌트에 대한 단위 테스트와 통합 테스트 파일들이 포함되어 있습니다.

<font size ="5"> **src 디렉토리 상세 구조**</font>
- `data`
데이터 관련 모듈들이 위치합니다. 데이터 로딩, 전처리, 증강 등의 기능을 담당합니다.

- `ensemble`
앙상블 학습 관련 모듈들이 위치합니다. 배깅, 스태킹, 보팅 등 다양한 앙상블 기법을 구현한 클래스들이 포함되어 있습니다.

- `experiments`
실험 설정 및 실행과 관련된 모듈들이 위치합니다. 실험 구성, 앙상블 실행, 단일 실험 실행 등의 기능을 담당합니다.

- `loss_functions`
손실 함수 관련 모듈들이 위치합니다. 기본 손실 함수와 사용자 정의 손실 함수들이 포함되어 있습니다.

- `models`
모델 관련 모듈들이 위치합니다. TIMM(Torch Image Models) 라이브러리를 사용한 모델 구현이 포함되어 있습니다.

- `optimizers`
옵티마이저 관련 모듈들이 위치합니다. 기본 옵티마이저와 사용자 정의 옵티마이저들이 포함되어 있습니다.

- `plmodules`
PyTorch Lightning 모듈들이 위치합니다. 스케치 모듈 등 특정 태스크를 위한 Lightning 모듈들이 포함되어 있습니다.

- `scheduler`
학습률 스케줄러 관련 모듈들이 위치합니다. 기본 스케줄러와 사용자 정의 스케줄러들이 포함되어 있습니다.



- `utils`
프로젝트 전반에서 사용되는 유틸리티 함수들이 위치합니다. 데이터, 평가, 모델 관련 유틸리티 함수들이 포함되어 있습니다.
