import argparse
import importlib
import wandb
import os
import glob
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import cv2
import numpy as np
import torch

from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from torch.utils.data import DataLoader
from torchcam.methods import GradCAM

def visualize_gradcam(
        model: torch.nn.Module,
        device: torch.device,
        dataloader: DataLoader,
        target_layer: str,
        image_index: int,
        save_dir: str
    ):

    # Grad-CAM 추출기를 초기화합니다.
    cam_extractor = GradCAM(model, target_layer)
    model.eval()  # 모델을 평가 모드로 설정합니다.
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 시각화를 위한 Figure를 생성합니다.
    
     # 저장 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)
    
    # 기존 파일 중 가장 큰 숫자 찾기
    existing_files = [f for f in os.listdir(save_dir) if f.startswith("gradcam_result_") and f.endswith(".png")]
    max_num = 0
    for file in existing_files:
        try:
            num = int(file.split("_")[-1].split(".")[0])
            max_num = max(max_num, num)
        except ValueError:
            continue
    
    # 새 파일 이름 생성
    new_num = max_num + 1
    save_path = os.path.join(save_dir, f"gradcam_result_{new_num:03d}.png")
    
    # 데이터 로더에서 배치를 반복합니다.
    current_index = 0
    for batch in dataloader:
        
        if isinstance(batch, list):# batch가 리스트인 경우 첫 번째 요소를 inputs로 가정
            inputs = batch[0]
        else:
            inputs = batch
        
        inputs = inputs.to(device)  # 입력 이미지를 장치로 이동합니다.       
        outputs = model(inputs)  # 모델을 통해 예측을 수행합니다.
        _, preds = torch.max(outputs, 1)  # 예측된 클래스 인덱스를 가져옵니다.


        
        # 배치 내의 각 이미지에 대해 처리합니다.
        for j in range(inputs.size()[0]):
            if current_index == image_index:
                cam = cam_extractor(preds[j].item(), outputs[j].unsqueeze(0))[0]# CAM을 가져옵니다.               
                cam = cam.mean(dim=0).cpu().numpy()   # CAM을 1채널로 변환합니다.                              
                cam = cv2.resize(cam, (inputs[j].shape[2], inputs[j].shape[1]))# CAM을 원본 이미지 크기로 리사이즈합니다.              
                cam = (cam - cam.min()) / (cam.max() - cam.min())  # CAM을 정규화합니다.# 정규화              
                cam = np.uint8(255 * cam)# CAM을 0-255 범위로 변환합니다.
                cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)# 컬러맵을 적용하여 RGB 이미지로 변환합니다.
                cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)  # BGR에서 RGB로 변환            

                # 입력 이미지가 1채널 또는 3채널인지 확인하고 처리합니다.
                input_image = inputs[j].cpu().numpy().transpose((1, 2, 0))
                if input_image.shape[2] == 1:  # 1채널 이미지인 경우
                    input_image = np.squeeze(input_image, axis=2)  # (H, W, 1) -> (H, W)
                    input_image = np.stack([input_image] * 3, axis=-1)  # (H, W) -> (H, W, 3)로 변환하여 RGB처럼 만듭니다.

                else:  # 3채널 이미지인 경우
                    input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())
                    input_image = (input_image * 255).astype(np.uint8)  # 정규화된 이미지를 8비트 이미지로 변환합니다.

                # 오리지널 이미지
                axes[0].imshow(input_image)
                axes[0].set_title("Original Image")
                axes[0].axis('off')              

                # Grad-CAM 이미지
                axes[1].imshow(cam)
                axes[1].set_title("Grad-CAM Image")
                axes[1].axis('off')
            
                # 오버레이된 이미지 생성
                overlay = cv2.addWeighted(input_image, 0.5, cam, 0.9, 0)
                axes[2].imshow(overlay)
                axes[2].set_title("Overlay Image")
                axes[2].axis('off')  

                # 이미지 저장
                plt.savefig(save_path)
                plt.close()
                print(f"GradCAM image saved to {save_path}")
                

            current_index += 1

    # 이미지 저장
    plt.savefig(save_path)
    plt.close()
    print(f"GradCAM image saved to {save_path}")


def main(config_path, use_wandb=True, sweep_dict=None):
    # YAML 파일 로드
    config = OmegaConf.load(config_path)
    print(config)


    # 데이터 모듈 동적 임포트
    data_module_path, data_module_class = config.data_module.rsplit(".", 1)
    # .을 기준으로 오른쪽에서 split하여 모듈 경로와 이름을 분리한다.
    DataModuleClass = getattr(
        importlib.import_module(data_module_path), data_module_class
    )

    # 데이터 모듈 설정
    data_config_path = config.data_config_path
    augmentation_config_path = config.augmentation_config_path
    seed = config.get("seed", 42)  # 시드 값을 설정 파일에서 읽어오거나 기본값 42 사용
    data_module = DataModuleClass(data_config_path, augmentation_config_path, seed)
    data_module.setup()  # 데이터 모듈에는 setup이라는 메소드가 존재한다.

    # 모델 모듈 동적 임포트
    model_module_path, model_module_class = config.model_module.rsplit(".", 1)
    ModelModuleClass = getattr(
        importlib.import_module(model_module_path), model_module_class
    )

    # 모델 설정
    model = ModelModuleClass(config)

    # Wandb 로거 설정 (use_wandb 옵션에 따라)
    logger = None
    if use_wandb:
        logger = WandbLogger(project="Sketch", name=config.wandb_name)

    # 콜백 설정
    checkpoint_callback = ModelCheckpoint(
        monitor=config.callbacks.model_checkpoint.monitor,
        save_top_k=config.callbacks.model_checkpoint.save_top_k,
        mode=config.callbacks.model_checkpoint.mode,
    )
    early_stopping_callback = EarlyStopping(
        monitor=config.callbacks.early_stopping.monitor,
        patience=config.callbacks.early_stopping.patience,
        mode=config.callbacks.early_stopping.mode,
    )

    # 트레이너 설정
    trainer = pl.Trainer(
        **config.trainer,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=logger,
        precision='16-mixed',
        # default_root_dir=config.trainer.default_root_dir #output 에 저장으로 변경.
    )

    # 훈련 시작
    trainer.fit(model, datamodule=data_module)

    # GradCAM 시각화 추가
    if config.get("visualize_gradcam", False):
        visualize_gradcam(
            model=model,
            device=model.device,
            dataloader=data_module.val_dataloader(),
            target_layer=config.gradcam.target_layer,
            image_index=config.gradcam.image_index,
            save_dir=config.gradcam.save_dir  # 저장 경로 추가
        )


if __name__ == "__main__":
    import torch, gc
    gc.collect()
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(description="Train a model with PyTorch Lightning")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument("--use_wandb", action="store_true", help="Use Wandb logger")
    args = parser.parse_args()

    main(args.config, args.use_wandb)
