import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import timm
import random

from torchvision import transforms
from torch.utils.data import DataLoader
from torchcam.methods import GradCAM
from typing import Callable
from timm import create_model
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform



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
                return #return을 적지 않으면 계속 함수가 돌아가는 무한 루프 빠짐.

            current_index += 1

    # 만약 원하는 인덱스의 이미지를 찾지 못했다면 메시지를 출력합니다.
    print(f"Image at index {image_index} not found in the dataloader.")



#이미지 파일을 무작위로 선택할 수 있게 합니다.(Eva모델에서는 필요.)
def get_random_image(train_dir):
    image_files = []
    for root, dirs, files in os.walk(train_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    
    if image_files:
        chosen_file = random.choice(image_files)
        print(f"Randomly selected: {chosen_file}")
        return chosen_file
    else:
        print(f"No image files found in the directory: {train_dir}")
        raise FileNotFoundError(f"No image files found in the directory: {train_dir}")




def visualize_attention_eva02(
        model:torch.nn.Module,
        transform: Callable,
        input_image_dir :str, 
        output_image_dir:str, 
        layer_index: int =0, 
        head_index: int =0):

    if not os.path.isfile(input_image_dir):
        raise FileNotFoundError(f"Input image not found: {input_image_dir}")

    os.makedirs(output_image_dir, exist_ok=True)

    # 이미지 로드 및 전처리
    image = Image.open(input_image_dir).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)

    # 어텐션 출력을 위한 훅 등록
    attention_maps = []

    def hook_fn(module, input, output):
        # EVA02 모델의 출력 구조에 맞게 수정
        attention_maps.append(output[0].detach())  # 어텐션 출력이 두 번째 요소

    # Register hook on all attention layers
    for name, module in model.named_modules():
        if isinstance(module, timm.models.eva.EvaAttention):  # EVA02 모델의 Attention 클래스 확인
            module.register_forward_hook(hook_fn)

    # 모델 추론
    with torch.no_grad():
        _ = model(input_tensor)
   
    
    if attention_maps:
        attention_map = attention_maps[layer_index]

        print(f"Attention map shape: {attention_map.shape}")

        # EVA02 모델의 어텐션 맵 구조에 맞게 수정
        print(f"Initial attention_map shape: {attention_map.shape}")
        print(f"head_index: {head_index}")
        
        if attention_map.shape[1] > head_index:
            attention_map_head = attention_map.cpu().numpy()
            print(f"attention_map_head shape: {attention_map_head.shape}")
            print(f"attention_map_head dtype: {attention_map_head.dtype}")
            
            # 어텐션 맵 크기 확인 및 조정
            if attention_map_head.shape == (1025, 1024):
                attention_map_no_cls = attention_map_head[1:,:]
                print(f"attemtion_map_head_no_cls: {attention_map_no_cls.shape}")
                num_patches = attention_map_head.shape[0]-1
                patch_size = int(num_patches**0.5)
                print(f"num_patches: {num_patches}, patch_size: {patch_size}")
                # 첫 번째 행(CLS 토큰)을 제외하고 나머지를 정사각형으로 재구성
                try:
                    attention_map_resized = attention_map_no_cls.reshape(patch_size, patch_size, patch_size, patch_size).mean(axis=(1, 3))
                    print(f"Resized attention map shape: {attention_map_resized.shape}")
                except ValueError as e:
                    print(f"Reshape error: {e}")
                    print(f"attention_map_no_cls[:, :-1] shape: {attention_map_no_cls[:, :-1].shape}")
                    attention_map_resized = attention_map_no_cls.reshape(patch_size, patch_size)
            else:
                print(f"Unexpected attention map shape: {attention_map_head.shape}")
                return
        else:
            print(f"Invalid head index: {head_index}. Available heads: {attention_map.shape[1]}")
            return
        

        # 시각화
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        ax1.imshow(image)
        ax1.set_title("Original Image")
        ax1.axis('off')
        if 'attention_map_resized' not in locals():
            print("Warning: attention_map_resized was not created. Using fallback.")
            attention_map_resized = np.zeros((32, 32))  # 또는 다른 적절한 대체 값

        
        im = ax2.imshow(attention_map_resized, cmap='viridis', interpolation='nearest')
        ax2.set_title(f"Attention Map (Layer {layer_index}, Head {head_index})")
        ax2.axis('off')

        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        
        # 파일 이름 생성 로직 수정
        existing_files = [f for f in os.listdir(output_image_dir) if f.startswith(f"attention_map_head{head_index}_layer{layer_index}_") and f.endswith(".png")]
        max_num = 0
        for file in existing_files:
            try:
                num = int(file.split("_")[-1].split(".")[0])
                max_num = max(max_num, num)
            except ValueError:
                continue
        
        new_num = max_num + 1
        output_filename = f"attention_map_head{head_index}_layer{layer_index}_{new_num:03d}.png"
        output_path = os.path.join(output_image_dir, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Attention map saved to: {output_path}")
    else:
        print("No attention maps were captured.")
        


#config 파일을 이용해 시각화 할 수 있게끔 합니다.
def perform_visualizations(config, model, data_module):
    #현재 실행준인 Py의 절대 경로를 기준으로, 루트 디렉토리 경로를 찾습니다.
    project_root =  os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
    if config.get("visualize_gradcam", False):
        visualize_gradcam(
            model=model,
            device=model.device,
            dataloader=data_module.val_dataloader(),
            target_layer=config.gradcam.target_layer,
            image_index=config.gradcam.image_index,
            save_dir=config.gradcam.save_dir
        )

    if config.get("visualize_attention", False):
        eva02_model = create_model('eva02_large_patch14_448.mim_m38m_ft_in22k_in1k', pretrained=True)
        eva02_model.eval()
        model_config = resolve_data_config({}, model=eva02_model)
        transform = create_transform(**model_config)

        input_image_dir = os.path.join(project_root, config["attention"]["input_image_dir"])
        print(f"Full path of input_image_dir: {input_image_dir}")
        
        if not os.path.exists(input_image_dir):
            print(f"Directory does not exist: {input_image_dir}")
            return
        #랜덤이미지 선택
        try:
            random_image_path = get_random_image(input_image_dir)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return

        visualize_attention_eva02(
            model=eva02_model,
            transform=transform,
            input_image_dir=random_image_path,
            output_image_dir=os.path.join(project_root, config["attention"]["output_image_dir"]),
            layer_index=config['attention']['layer_index'],
            head_index=config['attention']['head_index']
        )