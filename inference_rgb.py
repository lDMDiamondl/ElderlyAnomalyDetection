import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from collections import Counter

video_path = '/mnt/c/Users/k2i12/Desktop/video1.mp4'
model_path = '/mnt/c/Users/k2i12/Desktop/checkpoints_rgb/model_best.pth.tar'

# 라벨 목록
labels = ['falldown', 'lying', 'sit', 'throw', 'walk']

# GPU 사용 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 프레임을 변환하기 위한 트랜스폼 정의 (320x240 크기)
transform = transforms.Compose([
    transforms.Resize((240, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_video_frames(video_path):
    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # OpenCV는 BGR 형식을 사용하므로 RGB로 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # PIL 이미지로 변환
        pil_image = Image.fromarray(frame_rgb)
        # 변환 적용
        frame_tensor = transform(pil_image)
        frames.append(frame_tensor)
    
    cap.release()
    return frames


def load_model(model_path):
    # ResNet152 모델을 로드하고 마지막 레이어를 수정
    model = models.resnet152(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 5)  # 5개 클래스에 맞게 수정
    
    # 체크포인트 로드
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # fc_action을 fc로 변경 (오류 해결)
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    
    for key in state_dict.keys():
        new_key = key.replace("fc_action", "fc")
        new_state_dict[new_key] = state_dict[key]
    
    # 수정된 state_dict를 로드
    model.load_state_dict(new_state_dict)
    model.eval()
    model.to(device)
    return model


def predict_label(model, frames):
    predictions = []

    # 각 프레임에 대해 예측 수행
    for i, frame in enumerate(frames):
        # 배치 차원을 추가 (1, C, H, W)
        input_tensor = frame.unsqueeze(0).to(device)
        
        # 추론 실행
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            predictions.append(predicted.item())
        
        # 각 프레임의 예측된 라벨 출력
        print(f'Frame {i + 1}: Predicted label = {labels[predicted.item()]}')

    # 각 라벨들의 개수 출력
    label_counts = Counter(predictions)
    print("\nLabel counts:")
    for idx, count in label_counts.items():
        print(f'{labels[idx]}: {count}')
    
    # 가장 많이 예측된 라벨 출력
    predicted_label = max(set(predictions), key=predictions.count)
    print(f'Predicted label: {labels[predicted_label]}')


if __name__ == '__main__':
    print("Loading video frames...")
    frames = load_video_frames(video_path)
    
    print("Loading model...")
    model = load_model(model_path)
    
    print("Predicting label...")
    predict_label(model, frames)