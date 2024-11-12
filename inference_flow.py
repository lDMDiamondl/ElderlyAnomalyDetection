import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from collections import Counter
import argparse

labels = ['falldown', 'lying', 'sit', 'throw', 'walk']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((240, 320)),  # 320x240
    transforms.ToTensor(),
])

def compute_optical_flow(video_path, num_frames=10):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    
    if not ret:
        print(f"No video : {video_path}")
        return []
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    flow_frames = []
    frame_count = 0

    # Optical flow (10frame x 2channel) ###!!!!
    while cap.isOpened() and frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        prev_gray = gray

        flow_x, flow_y = cv2.split(flow)
        flow_x_norm = cv2.normalize(flow_x, None, 0, 255, cv2.NORM_MINMAX)
        flow_y_norm = cv2.normalize(flow_y, None, 0, 255, cv2.NORM_MINMAX)

        flow_x_pil = Image.fromarray(flow_x_norm).convert('L')
        flow_y_pil = Image.fromarray(flow_y_norm).convert('L')

        flow_x_tensor = transform(flow_x_pil)
        flow_y_tensor = transform(flow_y_pil)

        # 2채널 flow 데이터 save
        flow_frames.append(flow_x_tensor)
        flow_frames.append(flow_y_tensor)
        frame_count += 1

    cap.release()
    
    # Optical Flow 데이터를 결합하여 20채널 텐서로 만듦
    if len(flow_frames) >= 20:
        flow_tensor = torch.cat(flow_frames[:20], dim=0)
        return flow_tensor
    else:
        print(f"프레임이 부족합니다: {len(flow_frames)}개의 채널만 처리되었습니다.")
        return None


def load_model(model_path):
    # 첫 번째 레이어를 20채널로 수정
    model = models.resnet152(weights=None)
    model.conv1 = torch.nn.Conv2d(20, model.conv1.out_channels, kernel_size=model.conv1.kernel_size,
                                    stride=model.conv1.stride, padding=model.conv1.padding, bias=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 5)  ### num_class
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # state_dict에서 fc_action을 fc로 변경
    state_dict = checkpoint['state_dict']
    new_state_dict = {}

    for key in state_dict.keys():
        # fc_action으로 저장된 가중치를 fc로 변경
        new_key = key.replace("fc_action", "fc")
        new_state_dict[new_key] = state_dict[key]
    
    # 수정된 state_dict를 로드
    model.load_state_dict(new_state_dict)
    model.eval()
    model.to(device)
    return model

def predict_label(model, flow_tensor):
    input_tensor = flow_tensor.unsqueeze(0).to(device)  # 배치 차원 추가=>(1,20,240,320)
    
    # 추론 실행
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
    
    predicted_label = labels[predicted.item()]
    print(f'Predicted label: {predicted_label}') 
    return predicted_label


if __name__ == '__main__':
    # 비디오 파일 경로 및 학습된 모델 파일 경로를 직접 기입
    video_path = '/mnt/c/Users/k2i12/Desktop/video5.mp4'
    model_path = '/mnt/c/Users/k2i12/Desktop/checkpoints_flow/model_best.pth.tar'

    # 비디오에서 Optical Flow 추출
    print("Computing optical flow from video...")
    flow_tensor = compute_optical_flow(video_path)

    if flow_tensor is None:
        print("Optical Flow 데이터를 생성하지 못했습니다.")
        exit()

    # 모델 로드
    print("Loading model...")
    model = load_model(model_path)

    # 추론 수행
    print("Predicting label...")
    predict_label(model, flow_tensor)