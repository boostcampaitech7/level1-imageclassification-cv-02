# Sketch Image Classification


## git 설치
```bash
apt-get update -y
apt-get install -y libgl1-mesa-glx libglib2.0-0
apt install -y git
git clone https://github.com/boostcampaitech7/level1-imageclassification-cv-02.git
cd level1-imageclassification-cv-02
```


## Python package 설치
이 프로젝트는 Poetry를 사용하여 의존성을 관리합니다. 설치 방법은 다음과 같습니다:

```bash
# Poetry 설치
python -m venv competition1
source competition1/bin/activate
pip install poetry
```

### 1. poetry 의존성 update
```bash
poetry lock
```
### 2. 설치
```bash
poetry install
```
- 추가로 설치한 패키지들도 있어서, pip도 해주어야 합니다. (lock으로 설정 X)
```bash
pip install -r requirements.txt
```

## 사용 방법
### Train
```bash
python train.py --config configs/train_configs/train/config.yaml
```


### Test (Inference)
```bash
python test.py --config configs/train_configs/test/config.yaml
```

## Wandb
- Wandb: https://kr.wandb.ai/ 

터미널에서 `wandb login` 실행 후, 본인의 API를 입력합니다.
- **Wandb 쓰는 방법** 👉 `--use_wandb` 추가
```bash
python train.py --config configs/train_configs/train/config.yaml --use_wandb
```

## sweep
- sweep을 사용할 때는, configs/train_configs/train/config.yaml 내에 use_sweep을 **True**로 변경합니다.

### sweep 사용 방법
1. agent 생성
```bash
wandb sweep configs/train_configs/train/sweep.yaml
```
2. agent 적용
```bash
wandb agent <your wandb agent> -- count 5
```
