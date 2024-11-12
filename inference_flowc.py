import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from collections import deque
import time

labels = ['falldown', 'lying', 'sit', 'throw', 'walk', 'punch']

# labels = ['falldown', 'lying', 'punch', 'sit', 'throw', 'walk']
# labels = ['sit', 'walk']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((240, 320)),  # 320x240
    transforms.ToTensor(),
])

def compute_optical_flow_webcam(model, display_duration=3):
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('http://172.30.1.31:2023/video')
    ret, prev_frame = cap.read()
    
    if not ret:
        print("Error: Could not read from webcam.")
        return
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    flow_frames = deque(maxlen=20)
    frame_count = 0
    last_predicted_label = ""
    last_predicted_time = time.time()

    print("Starting Camera. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
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

        # 2채널 flow 데이터 저장
        flow_frames.append(flow_x_tensor)
        flow_frames.append(flow_y_tensor)
        frame_count += 1

        # 예측을 위해 필요한 프레임 수가 채워지면 예측 수행
        if len(flow_frames) == 20:
            flow_tensor = torch.cat(list(flow_frames), dim=0).unsqueeze(0).to(device)
            predicted_label = predict_label(model, flow_tensor)
            last_predicted_label = predicted_label
            last_predicted_time = time.time()
            flow_frames.clear()

        # 예측된 라벨을 프레임에 표시
        if time.time() - last_predicted_time < display_duration:
            cv2.putText(frame, f'Predicted label: {last_predicted_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        cv2.imshow('Camera', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def load_model(model_path):
    # 첫 번째 레이어를 20채널로 수정
    model = models.resnet152(weights=None)
    model.conv1 = torch.nn.Conv2d(20, model.conv1.out_channels, kernel_size=model.conv1.kernel_size,
                                    stride=model.conv1.stride, padding=model.conv1.padding, bias=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 6)  # 6개 클래스에 맞게 수정
    # model.fc = torch.nn.Linear(model.fc.in_features, 2)  # 2개 클래스에 맞게 수정
    
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
    # 추론 실행
    with torch.no_grad():
        output = model(flow_tensor)
        _, predicted = torch.max(output, 1)
    
    predicted_label = labels[predicted.item()]
    print(f'Predicted label: {predicted_label}') 
    return predicted_label

if __name__ == '__main__':
    model_path = '/mnt/c/Users/k2i12/Desktop/test/flow2.tar'
    
    print("Loading model...")
    model = load_model(model_path)
    
    print("Computing optical flow from webcam...")
    compute_optical_flow_webcam(model)