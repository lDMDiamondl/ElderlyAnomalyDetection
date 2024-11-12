# A sample function for classification using temporal network
# Customize as needed:
# e.g. num_categories, layer for feature extraction, batch_size

import glob
import os
import sys
import numpy as np
import math
import cv2

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

sys.path.insert(0, "../../")
import video_transforms

def VideoTemporalPrediction_cam(
        flow_x_frames,  # flow_x_frames list (numpy arrays)
        flow_y_frames,  # flow_y_frames list (numpy arrays)
        net,
        num_categories=6,
        start_frame=0,
        num_frames=0,
        num_samples=25,
        optical_flow_frames=10  # 10 frame
        ):

    if num_frames == 0:
        duration = len(flow_x_frames)
    else:
        duration = num_frames

    clip_mean = [0.5] * 20
    clip_std = [0.226] * 20
    normalize = video_transforms.Normalize(mean=clip_mean, std=clip_std)
    val_transform = video_transforms.Compose([
        video_transforms.ToTensor(),
        normalize,
    ])

    # 프레임 선택
    step = max(1, int(math.floor((duration - optical_flow_frames + 1) / num_samples)))
    dims = (256, 340, optical_flow_frames * 2, num_samples)
    flow = np.zeros(shape=dims, dtype=np.float64)
    flow_flip = np.zeros(shape=dims, dtype=np.float64)

    for i in range(num_samples):
        for j in range(optical_flow_frames):
            idx = i * step + j + start_frame
            if idx < len(flow_x_frames):
                img_x = flow_x_frames[idx]  
                img_y = flow_y_frames[idx]  
            else:
                img_x = flow_x_frames[-1]  
                img_y = flow_y_frames[-1]  

            if img_x is None or img_y is None:
                print(f"Warning: Missing image at index {idx}")
                continue

            img_x = cv2.resize(img_x, dims[1::-1])
            img_y = cv2.resize(img_y, dims[1::-1])

            flow[:, :, j * 2, i] = img_x
            flow[:, :, j * 2 + 1, i] = img_y

            flow_flip[:, :, j * 2, i] = 255 - img_x[:, ::-1]
            flow_flip[:, :, j * 2 + 1, i] = img_y[:, ::-1]

    # 자르기
    flow_1 = flow[:224, :224, :, :]
    flow_2 = flow[:224, -224:, :, :]
    flow_3 = flow[16:240, 60:284, :, :]
    flow_4 = flow[-224:, :224, :, :]
    flow_5 = flow[-224:, -224:, :, :]
    flow_f_1 = flow_flip[:224, :224, :, :]
    flow_f_2 = flow_flip[:224, -224:, :, :]
    flow_f_3 = flow_flip[16:240, 60:284, :, :]
    flow_f_4 = flow_flip[-224:, :224, :, :]
    flow_f_5 = flow_flip[-224:, -224:, :, :]

    flow = np.concatenate((flow_1, flow_2, flow_3, flow_4, flow_5, flow_f_1, flow_f_2, flow_f_3, flow_f_4, flow_f_5), axis=3)

    _, _, _, c = flow.shape
    flow_list = []
    for c_index in range(c):
        cur_img = flow[:, :, :, c_index].squeeze()
        cur_img_tensor = val_transform(cur_img.astype(np.uint8))
        flow_list.append(np.expand_dims(cur_img_tensor.numpy(), 0))

    flow_np = np.concatenate(flow_list, axis=0)

    batch_size = 25
    prediction = np.zeros((num_categories, flow.shape[3]))
    num_batches = int(math.ceil(float(flow.shape[3]) / batch_size))

    for bb in range(num_batches):
        span = range(batch_size * bb, min(flow.shape[3], batch_size * (bb + 1)))

        input_data = flow_np[span, :, :, :]
        imgDataTensor = torch.from_numpy(input_data).type(torch.FloatTensor).cuda()
        with torch.no_grad():  # 추론 모드에서 그래디언트 계산 비활성화
            output = net(imgDataTensor)
        result = output.data.cpu().numpy()
        prediction[:, span] = np.transpose(result)

    return prediction
