# 낙상 및 행동 감지 (Two‑Stream 기반 변형)

<!-- Demo GIFs -->

<p align="center">
  <img src="path/to/demo1.gif" width="280" />
  <img src="path/to/demo2.gif" width="280" />
  <img src="path/to/demo3.gif" width="280" />
</p>


이 저장소는 **[bryanyzhu/two-stream-pytorch](https://github.com/bryanyzhu/two-stream-pytorch)** 코드를 변형하여 만든 낙상/행동 감지 파이프라인입니다. YOLO로 사람을 먼저 검출한 뒤(검출된 사람 있을 때만) Two‑Stream(공간 + 옵티컬 플로우) 기반 분류기를 돌려 다음 **6가지 동작**을 프레임 단위로 예측합니다.

## 감지하는 행동 (클래스 번호)

1. 낙상 (Fall)
2. 누워 있음 (Lying)
3. 주먹질 (Punching)
4. 앉아 있음 (Sitting)
5. 물건 던짐 (Throwing)
6. 걷고 있음 (Walking)

---

## 주요 개요

1. **사람 검출 (YOLO)**: 입력 비디오에서 사람 객체가 감지되는지 확인합니다. 사람(또는 관심 영역)이 한 명이라도 검출되면 해당 프레임(또는 시퀀스)에 대해 행동 예측을 수행합니다.
2. **Two‑Stream 행동 분류기**: 공간(stream RGB) + 시간(stream Optical Flow) 두 입력을 사용하여 행동을 분류합니다. 원본 구현은 ResNet(혹은 변형 백본)을 사용합니다.
3. **후처리**:  원본 영상에 예측 라벨을 오버레이한 결과 비디오를 생성할 수 있습니다.

> ⚠️ **주의사항**: 개인 프로젝트에서 사용하던 코드를 기반으로 하여 일부 경로(path) 지정이 다소 지저분하거나 절대경로가 포함되어 있을 수 있습니다. 사용 환경에 따라 경로를 수정해 사용하는 것을 권장합니다.

