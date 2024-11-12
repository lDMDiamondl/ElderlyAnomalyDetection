import os
import sys
import numpy as np
import cv2
import time
import torch
import torch.nn as nn
import argparse
import threading

sys.path.insert(0, "../../")
from VideoSpatialPrediction_cam import VideoSpatialPrediction_cam
from VideoTemporalPrediction_cam import VideoTemporalPrediction_cam

import models

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
    with open(class_index_file, 'r') as f:
        for line in f:
            idx, class_name = line.strip().split()
            class_labels[int(idx) - 1] = class_name
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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, new_size)
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

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

        # 예측 결과가 있을 경우 화면에 표시
        if result_dict['predicted_label'] is not None:
            cv2.putText(frame_resized, f'Action: {result_dict["predicted_label"]}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Webcam Action Recognition', frame_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Two-Stream 모델을 사용한 웹캠 실시간 예측")
    parser.add_argument('--rgb_model_path', type=str, default='/mnt/c/Users/k2i12/Desktop/test/rgb2.tar',help='RGB 모델 파일 경로')
    parser.add_argument('--flow_model_path', type=str, default='/mnt/c/Users/k2i12/Desktop/test/flow2.tar', help='Flow 모델 파일 경로')
    parser.add_argument('--num_categories', type=int, default=6, help='카테고리 수')
    parser.add_argument('--class_ind_path', type=str, default='/home/dmd/workspace/pyws/twos/datasets/ucf101_splits/classInd.txt', help='클래스 인덱스 파일 경로')
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
