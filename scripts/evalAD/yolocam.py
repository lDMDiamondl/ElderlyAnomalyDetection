import os
import sys
import numpy as np
import cv2
from ultralytics import YOLO
import time
import torch
import torch.nn as nn
import argparse
import threading
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "../../")
from VideoSpatialPrediction_cam import VideoSpatialPrediction_cam
from VideoTemporalPrediction_cam import VideoTemporalPrediction_cam

import models

# YOLOv8 모델 로드
YOLOmodel = YOLO("/home/dmd/workspace/pyws/twos/scripts/evalAD/yolo11n.pt")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_model(model_path, num_categories, model_type='rgb'):
    model_start_time = time.time()
    params = torch.load(model_path)

    if model_type == 'rgb':
        model = models.rgb_resnet152(pretrained=False, num_classes=num_categories)
    else:
        model = models.flow_resnet152(pretrained=False, num_classes=num_categories)

    model.load_state_dict(params['state_dict'])

    if torch.cuda.is_available():
        model.cuda()
        print(f"Using GPU for {model_type} model.")
    else:
        print(f"Using CPU for {model_type} model.")

    model.eval()
    model_end_time = time.time()
    model_time = model_end_time - model_start_time
    print(f"Action recognition {model_type} model is loaded in {model_time:.4f} seconds.")

    return model

def load_class_labels(class_index_file):
    class_labels = {}
    with open(class_index_file, 'r', encoding='utf-8') as file:
        for idx, line in enumerate(file):
            text = line.strip().split(' ', 1)[1]
            class_labels[idx] = text
    return class_labels

def twostream_prediction(rgb_model, flow_model, num_categories, class_labels, frame_buffer, flow_buffer, optical_flow_frames, result_dict):
    '''
    비동기 스레드
    '''
    rgb_prediction = VideoSpatialPrediction_cam(frame_buffer[-optical_flow_frames:], rgb_model, num_categories)
    avg_rgb_pred = np.mean(rgb_prediction, axis=1)

    flow_x = [cv2.normalize(flow[:, :, 0], None, 0, 255, cv2.NORM_MINMAX) for flow in flow_buffer[-optical_flow_frames:]]
    flow_y = [cv2.normalize(flow[:, :, 1], None, 0, 255, cv2.NORM_MINMAX) for flow in flow_buffer[-optical_flow_frames:]]
    flow_prediction = VideoTemporalPrediction_cam(flow_x, flow_y, flow_model, num_categories)
    avg_flow_pred = np.mean(flow_prediction, axis=1)

    combined_pred = (avg_rgb_pred + avg_flow_pred) / 2
    pred_index = np.argmax(combined_pred)
    predicted_label = class_labels[pred_index]

    print(f'Two-Stream Prediction: {predicted_label} (Index: {pred_index})')
    result_dict['predicted_label'] = predicted_label

def webcam_two_stream_inference(rgb_model, flow_model, num_categories, class_labels, new_size=(320, 240)):
    cap = cv2.VideoCapture('http://172.30.1.10:2023/video')
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    prev_gray = None
    frame_buffer = []
    flow_buffer = []
    optical_flow_frames = 10
    result_dict = {'predicted_label': None}
    frame_count = 0
    detect_frame = 1

    # 한글 폰트 설정
    font = ImageFont.truetype('gulim.ttc', 30)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, new_size)
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        # 매 detect_frame번째 프레임마다 YOLO 모델 실행
        if frame_count % detect_frame == 0:
            # YOLO 모델로 사람 검출
            results = YOLOmodel(frame_resized)
            person_detected = False

            for result in results:
                for box in result.boxes:
                    if box.cls == 0:  # 사람 클래스
                        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        person_detected = True

        if person_detected:
            frame_buffer.append(frame_resized)

            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                flow_buffer.append(flow)
            prev_gray = gray

        # 예측에 사용할 프레임이 충분히 쌓이고 예측 스레드가 실행 중이 아닐 때 예측 수행
        if len(frame_buffer) >= optical_flow_frames and threading.active_count() == 1:
            prediction_thread = threading.Thread(target=twostream_prediction, args=(rgb_model, flow_model, num_categories, class_labels, frame_buffer, flow_buffer, optical_flow_frames, result_dict))
            prediction_thread.start()

            # 프레임 버퍼 유지
            frame_buffer = frame_buffer[-optical_flow_frames:]
            flow_buffer = flow_buffer[-optical_flow_frames:]

        # 한글 텍스트 추가, 예측 결과가 있을 경우 화면에 표시
        pil_image = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        if person_detected and result_dict['predicted_label'] is not None:
            draw.text((10, 10), f'예측 동작: {result_dict["predicted_label"]}', font=font, fill=(0, 255, 0, 255))
            frame_resized = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # 예측 결과가 있을 경우 화면에 표시
        # if person_detected and result_dict['predicted_label'] is not None:
        #     cv2.putText(frame_resized, f'Action: {result_dict["predicted_label"]}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        
        cv2.putText(frame_resized, f'Box: {person_detected}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        frame_count += 1
        cv2.imshow('Webcam Action Recognition', frame_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Two-Stream 모델을 사용한 웹캠 실시간 예측")
    parser.add_argument('--rgb_model_path', type=str, default='/mnt/c/Users/k2i12/Desktop/test/rgb3.tar',help='RGB 모델 파일 경로')
    parser.add_argument('--flow_model_path', type=str, default='/mnt/c/Users/k2i12/Desktop/test/flow3.tar', help='Flow 모델 파일 경로')
    parser.add_argument('--num_categories', type=int, default=6, help='카테고리 수')
    parser.add_argument('--class_ind_path', type=str, default='/home/dmd/workspace/pyws/twos/datasets/ucf101_splits/classIndH.txt', help='클래스 인덱스 파일 경로')
    args = parser.parse_args()

    if torch.cuda.is_available():
        print("GPU is available. Using GPU for inference.")
    else:
        print("GPU is not available. Using CPU for inference.")

    class_labels = load_class_labels(args.class_ind_path)
    print(class_labels)

    rgb_model = load_model(args.rgb_model_path, args.num_categories, model_type='rgb')
    flow_model = load_model(args.flow_model_path, args.num_categories, model_type='flow')

    webcam_two_stream_inference(rgb_model, flow_model, args.num_categories, class_labels, new_size=(320, 240))

if __name__ == "__main__":
    main()
