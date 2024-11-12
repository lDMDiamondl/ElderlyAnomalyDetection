# from __future__ import print_function

import os
import sys
import glob
import argparse
from multiprocessing import Pool, current_process
import cv2
import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

def run_optical_flow(vid_item):
    vid_path = vid_item[0]
    vid_id = vid_item[1]
    vid_name = vid_path.split('/')[-1].split('.')[0]
    out_full_path = os.path.join(out_path, vid_name)
    
    try:
        os.makedirs(out_full_path)
    except OSError:
        print('{} {} error'.format(vid_id, vid_name))
        pass

    current = current_process()
    dev_id = (int(current._identity[0]) - 1) % NUM_GPU

    # Create folders only for flow_x and flow_y
    flow_x_path = os.path.join(out_full_path, 'flow_x')
    flow_y_path = os.path.join(out_full_path, 'flow_y')

    os.makedirs(flow_x_path, exist_ok=True)
    os.makedirs(flow_y_path, exist_ok=True)

    cap = cv2.VideoCapture(vid_path)
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        prev_gray = gray

        if new_size[0] > 0 and new_size[1] > 0:
            frame = cv2.resize(frame, new_size)
            flow = cv2.resize(flow, new_size)

        # Save RGB image in the root folder of this video
        img_path = os.path.join(out_full_path, 'img_{:05d}.jpg'.format(frame_idx))
        cv2.imwrite(img_path, frame)

        # Save optical flow images in respective folders
        flow_x, flow_y = cv2.split(flow)
        flow_x_img = os.path.join(flow_x_path, 'flow_x_{:05d}.jpg'.format(frame_idx))
        flow_y_img = os.path.join(flow_y_path, 'flow_y_{:05d}.jpg'.format(frame_idx))

        # Normalize flow images to 0-255 and save
        flow_x_norm = cv2.normalize(flow_x, None, 0, 255, cv2.NORM_MINMAX)
        flow_y_norm = cv2.normalize(flow_y, None, 0, 255, cv2.NORM_MINMAX)

        cv2.imwrite(flow_x_img, flow_x_norm)
        cv2.imwrite(flow_y_img, flow_y_norm)

        frame_idx += 1

    cap.release()
    sys.stdout.flush()
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="extract optical flows")
    parser.add_argument("--src_dir", type=str, default='/mnt/c/Users/k2i12/Desktop/nda',
                        help='path to the video data')
    parser.add_argument("--out_dir", type=str, default='/mnt/c/Users/k2i12/Desktop/nda2',
                        help='path to store frames and optical flow')
    parser.add_argument("--new_width", type=int, default=0, help='resize image width')
    parser.add_argument("--new_height", type=int, default=0, help='resize image height')
    parser.add_argument("--num_worker", type=int, default=16)
    parser.add_argument("--num_gpu", type=int, default=1, help='number of GPU')
    parser.add_argument("--ext", type=str, default='avi', choices=['avi', 'mp4'],
                        help='video file extensions')

    args = parser.parse_args()

    out_path = args.out_dir
    src_path = args.src_dir
    num_worker = args.num_worker
    ext = args.ext
    new_size = (args.new_width, args.new_height)
    NUM_GPU = args.num_gpu

    if not os.path.isdir(out_path):
        print("creating folder: " + out_path)
        os.makedirs(out_path)

    vid_list = glob.glob(src_path + '/*/*.' + ext)
    print(len(vid_list))

    # Use tqdm to display the progress bar
    process_map(run_optical_flow, zip(vid_list, range(len(vid_list))), max_workers=num_worker)