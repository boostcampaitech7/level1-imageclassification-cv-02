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
pip install -r requirements.txt
```
## train.py, test.py  돌리기전에 !!
먼저 Wandb https://kr.wandb.ai/ 에 들어가서 회원가입하고 본인 로그인해야합니다.

따라서 `pip install -r requirements.txt`를 하면 `wandb`가 다 설치되어 있을텐데, 이후에 
terminal에서 `wandb login` 치고 나서 본인의 API 입력하면 됨. 


## 사용 방법
### Train
- **config 파일 수정 잘 해서 돌리기**
```bash
python train.py --config configs/train_configs/train/config.yaml
```
- **Wandb 쓰는 방법**
```bash
python train.py --config configs/train_configs/train/config.yaml --use_wandb
```
`--use_wandb`를 붙여쓰면 됩니다.
### Test (Inference)
- **config 파일 수정 잘 해서 돌리기**
```bash
python test.py --config configs/train_configs/test/config.yaml
```



<font size ="5">**Wandb Sweep 쓰는 방법**</font>
### sweep 기능 적용!
1. sweep을 사용할 때는, configs/train_configs/train/config.yaml 내에 use_sweep을 **True**로 바꿔주세요!
2. config.yaml 내에 optimizer부분과 scheduler 부분은 **주석 처리**해주세요!
<img width="658" alt="스크린샷 2024-09-20 오후 5 12 29" src="https://private-user-images.githubusercontent.com/81224613/369334526-3aeb3166-a69a-46bc-9648-f84f046d4048.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MjcyNjk3OTcsIm5iZiI6MTcyNzI2OTQ5NywicGF0aCI6Ii84MTIyNDYxMy8zNjkzMzQ1MjYtM2FlYjMxNjYtYTY5YS00NmJjLTk2NDgtZjg0ZjA0NmQ0MDQ4LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA5MjUlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwOTI1VDEzMDQ1N1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWM1MzdkY2IyYWY3YWEzNDg0ZWUzYTI5MTA2MGQwOGU4NzVmZWQxN2UxMDUwMWQwNmRiZGY3NDkzYmYwZmFiMmUmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.yLAztJgBTu_7zE86Azrj3xPcqoL4VK3TDnJx6XKsLAQ">





## 📣 To Reviewers
<!-- 참고 사항을 적어주세요. 없으면 지워주세요. -->
### sweep 사용 방법
1. agent 생성
```bash
wandb sweep configs/train_configs/train/sweep.yaml
```
2. agent 적용
<img width="926" alt="스크린샷 2024-09-20 오후 5 17 17" src="https://private-user-images.githubusercontent.com/81224613/369333651-96f7373d-f7ba-462f-b875-6e989b835247.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MjcyNjk3OTcsIm5iZiI6MTcyNzI2OTQ5NywicGF0aCI6Ii84MTIyNDYxMy8zNjkzMzM2NTEtOTZmNzM3M2QtZjdiYS00NjJmLWI4NzUtNmU5ODliODM1MjQ3LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA5MjUlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwOTI1VDEzMDQ1N1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTNkOTY5OTc3ZjM4NzJiMjkyZDk2ZTJlZGUxNjRhYTRlYWRmYjY2M2E5NGUzYTU0MjYyNDM1YTE3YWEwYjQzNTkmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.IZ344_22mGudMAHg56IRIwj_uR-pD-FWWnMMu6qfPrQ">

```bash
wandb agent <your wandb agent> -- count 5
```
- 파란색 부분 복붙 후 count 추가하기 (count: sweep 적용 횟수)


### sweep 실행 결과
<img width="1037" alt="스크린샷 2024-09-20 오후 5 18 14" src="https://private-user-images.githubusercontent.com/81224613/369330280-baa337e0-bfd3-4e9a-b8a2-4f19beaecf08.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MjcyNjk3OTcsIm5iZiI6MTcyNzI2OTQ5NywicGF0aCI6Ii84MTIyNDYxMy8zNjkzMzAyODAtYmFhMzM3ZTAtYmZkMy00ZTlhLWI4YTItNGYxOWJlYWVjZjA4LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA5MjUlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwOTI1VDEzMDQ1N1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWI2ZjNmMWM1MjkyMDI0ODVjNzdhYjBiZWRjOWIwZTY0YjFlMTk1ODAwNWQ2NDU0Y2Q5ZWNjMzI4NTU1Y2ZmNDcmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.1TVZnKevmKi_kjsyUci0ux_Rwfo4xK7HbkM0lYJ0GYA">


## 프로젝트 구조
```
.
|-- README.md
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
|-- output
|   |-- eva02_large_patch14_448.csv
|   `-- lightning_logs
|-- poetry.lock
|-- pyproject.toml
|-- pytest.ini
|-- requirements.txt
|-- settings
|   `-- LICENSE
|-- src
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

### + checkpoint_path를 일일히 지정하기가 귀찮다면?
- `test.py`의 코드를 지우고 밑에 코드 복붙.
-> 하지만 이는 최신 체크포인트롤 이용해 test하는 것일뿐 validation_test가 가장 높은 것을 이용한게 아니므로 최적의 모델이 아닐 수 있음.
```python
import argparse
import os
import pytorch_lightning as pl
from omegaconf import OmegaConf

from src.data.custom_datamodules.sketch_datamodule import SketchDataModule
from src.plmodules.sketch_module import SketchModelModule


def get_latest_checkpoint(checkpoint_dir):
    checkpoint_paths = []
    for root, dirs, files in os.walk(checkpoint_dir):
        for file in files:
            if file.endswith('.ckpt'):
                checkpoint_paths.append(os.path.join(root, file))
    if not checkpoint_paths:
        return None
    return max(checkpoint_paths, key=os.path.getctime)


def main(config_path, checkpoint_path=None):
    # YAML 파일 로드
    config = OmegaConf.load(config_path)
    
    # model_name에서 '.' 이전 부분 추출하여 name 필드 설정
    model_name = config.model.model_name
    name_prefix = model_name.split('.')[0]
    
    if not config.get('name'):  # name 필드가 비어있다면 설정
        config.name = name_prefix
    
    print(f"Name from config: {config.name}")

    # 최신 체크포인트 경로 업데이트
    if checkpoint_path is None:
        checkpoint_dir = config.checkpoint_path
        checkpoint_path = get_latest_checkpoint(checkpoint_dir)
    
    if checkpoint_path is None:
        raise ValueError("No checkpoint found. Please specify a valid checkpoint path.")

    print(f"Using checkpoint: {checkpoint_path}")

    # 데이터 모듈 설정
    data_config_path = config.data_config_path
    augmentation_config_path = config.augmentation_config_path
    seed = config.get("seed", 42)  # 시드 값을 설정 파일에서 읽어오거나 기본값 42 사용
    data_module = SketchDataModule(data_config_path, augmentation_config_path, seed)
    data_module.setup()

    # 모델 설정
    model = SketchModelModule.load_from_checkpoint(checkpoint_path, config=config)

    # 트레이너 설정
    trainer = pl.Trainer(
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        precision=16,
        default_root_dir=config.trainer.default_root_dir  # output 폴더로 저장하게끔
    )

    # 평가 시작
    trainer.test(model, datamodule=data_module)

    # csv 파일에 output 저장하기
    output_path = f"{config.trainer.default_root_dir}/{config.name}.csv"  # output 폴더에 저장
    test_info = data_module.test_info
    predictions = model.test_predictions
    test_info['target'] = predictions
    test_info = test_info.reset_index().rename(columns={"index": "ID"})

    # 결과를 csv 파일로 저장
    test_info.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a model with PyTorch Lightning"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=False, help="Path to the model checkpoint"
    )
    args = parser.parse_args()

    main(args.config, args.checkpoint)

```