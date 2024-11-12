import os
import glob
import random

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i+1 for i in range(len(classes))}
    return classes, class_to_idx

def create_train_test_files(data_path, output_path, num_splits=3):

    # 데이터셋 내 모든 하위 디렉토리 목록
    classes, class_to_idx = find_classes(data_path)
    
    # 라벨별로 파일 분류
    label_data = {}

    for label in classes:
        directory = os.path.join(data_path, label)
        video_files = glob.glob(os.path.join(directory, '*'))
        label_data[label] = video_files
        
    for label, files in label_data.items():
        print(f"{label}: {len(files)} files")

    # 라벨별 데이터 섞기
    for label in label_data:
        random.shuffle(label_data[label])

    # 학습 및 테스트 데이터로 균등하게 분할
    split_data = {
        'train': [[] for _ in range(num_splits)],
        'test': [[] for _ in range(num_splits)]
    }

    for label, files in label_data.items():
        num_files = len(files)
        split_size = num_files // num_splits  # 각 분할마다 균등하게 나누기
        remainder = num_files % num_splits  # 나머지 파일의 수
        label_idx = class_to_idx[label]

        for i in range(num_splits):
            start_idx = i * split_size
            end_idx = start_idx + split_size + (1 if i < remainder else 0)  # 남은 파일을 앞에서부터 하나씩 추가
            split_files = files[start_idx:end_idx]

            # 80% 학습, 20% 테스트로 분할
            split_point = int(len(split_files) * 0.8)

            # label, 파일 경로, label_idx저장
            split_data['train'][i].extend([(f, label, label_idx) for f in split_files[:split_point]])
            split_data['test'][i].extend([(f, label, label_idx) for f in split_files[split_point:]])

    for i in range(num_splits):
        with open(os.path.join(output_path, f'trainlist{i + 1:02d}.txt'), 'w') as train_file, \
            open(os.path.join(output_path, f'testlist{i + 1:02d}.txt'), 'w') as test_file:
            for train_item, label, label_idx in split_data['train'][i]:
                # label/파일명 label_idx 쓰기
                train_file.write(f"{label}/{os.path.basename(train_item)} {label_idx}\n")
            for test_item, label, label_idx in split_data['test'][i]:
                # label/파일명 label_idx 쓰기
                test_file.write(f"{label}/{os.path.basename(test_item)} {label_idx}\n")

    # 클래스 인덱스 파일 생성
    with open(os.path.join(output_path, 'classInd.txt'), 'w') as class_file:
        for label, idx in class_to_idx.items():
            class_file.write(f"{idx} {label}\n")

data_path = '/mnt/c/Users/k2i12/Desktop/nda3'
output_path = '/mnt/c/Users/k2i12/Desktop/nda4'

if not os.path.exists(output_path):
    os.makedirs(output_path)
create_train_test_files(data_path, output_path)