import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from collections import Counter

model_path = '/mnt/c/Users/k2i12/Desktop/test/rgb2.tar'

# 라벨 목록
labels = ['falldown', 'lying', 'sit', 'throw', 'walk', 'punch']

# labels = ['falldown', 'lying', 'punch', 'sit', 'throw', 'walk']
# labels = ['sit', 'walk']

# GPU 사용 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 프레임을 변환하기 위한 트랜스폼 정의 (320x240 크기)
transform = transforms.Compose([
    transforms.Resize((240, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model(model_path):
    # ResNet152 모델을 로드하고 마지막 레이어를 수정
    model = models.resnet152(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 6)  # 6개 클래스에 맞게 수정
    # model.fc = torch.nn.Linear(model.fc.in_features, 2)
    
    # 체크포인트 로드
    checkpoint = torch.load(model_path, map_location=device)
    
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

def predict_label(model, frame):
    # 배치 차원을 추가 (1, C, H, W)
    input_tensor = frame.unsqueeze(0).to(device)
    
    # 추론 실행
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
    
    return predicted.item()

if __name__ == '__main__':
    print("Loading model...")
    model = load_model(model_path)

    # 웹캠에서 프레임을 읽기 위한 비디오 캡처 객체 생성
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('http://172.30.1.31:2023/video')

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # OpenCV는 BGR 형식을 사용하므로 RGB로 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # PIL 이미지로 변환
        pil_image = Image.fromarray(frame_rgb)
        # 변환 적용
        frame_tensor = transform(pil_image)
        
        # 예측 라벨 가져오기
        predicted_label_idx = predict_label(model, frame_tensor)
        predicted_label = labels[predicted_label_idx]
        
        # 예측된 라벨을 프레임에 표시
        cv2.putText(frame, f'Predicted label: {predicted_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        # 프레임을 출력
        cv2.imshow('Camera', frame)
        
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()