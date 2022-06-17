# Deep Learning-based Phase-only Hologram Generation
이 Repository는 경희대학교 2022년 1학기 소프트웨어융합 캡스톤디자인 수업의 일환으로 만들어졌습니다.
- 주저자: 차준영
- 공동저자: 반현민
- 지도교수: 황효석 교수님, 김휘용 교수님

## Overview
디지털 홀로그래피는 3차원 물체에서 나온 빛의 위상과 진폭 정보로 시각적인 정보를 기록하고 복원하는 기술이다. 이 홀로그램을 디스플레이 하기 위해선 디지털 공간광변조기(SLM; Spatial Light Modulator)가 필요하지만, 현재 기술적 한계로 빛의 진폭 혹은 위상만 디스플레이 할 수 있다. 이때 진폭 정보를 복원하는 진폭 공간광변조기보다 좀 더 빛을 효율적으로 사용하여 물체를 표현할 수 있는 위상 공간광변조기가 주로 사용되고 있으며, 이에 맞추어 빛의 위상만 기록하여 물체를 더 잘 표현할 수 있도록 위상 홀로그램(PoH; Phase-only Hologram)을 잘 만들기 위한 연구가 지속되고 있다. 

현재 위상 홀로그램을 잘 만드는 방법으로 대표적으로 반복 최적화 알고리즘인 GS algorithm (Gerchberg-Saxton algorithm)이 있다. 이 방법은 초반 0~1 사이의 Uniform Distribution으로 Random Phase를 만든 후, FFT / IFFT를 반복적으로 사용하여 입력 이미지의 진폭과 위상 정보가 위상만으로 적절히 표현되도록 만드는 기법이다. 이 방법을 충분히 반복하면 좋은 화질의 위상 홀로그램 이미지를 생성할 수 있다. 하지만 이런 반복 최적화 알고리즘은 계산 복잡도가 높고, 이미지 한 장씩 처리해야 하므로 병렬처리가 불가능하여 좋은 화질의 홀로그램 이미지를 생성하는 데 오랜 시간이 걸린다는 단점이 있다. 따라서 이 방법을 이용해 위상 홀로그램을 실시간으로 최적화하여 생성하는 데 한계가 있다. 이러한 이유로 딥러닝을 이용한 위상 홀로그램 생성 기법이 필요해지게 된다. 병렬처리가 가능한 딥러닝 방법을 이용하면 충분히 잘 학습된 신경망을 통해 실시간으로 좋은 품질의 위상 홀로그램을 생성할 수 있다. 따라서 본 캡스톤디자인에서는 딥러닝 기반으로 위상 홀로그램을 생성하는 신경망의 학습 전략을 수립하고자 한다.

------
## Training Strategy
![image](https://user-images.githubusercontent.com/34412522/174341940-e6a6309b-8dc6-4809-9683-9f1da5d92a1c.png)

1. 입력 이미지를 ASM을 통해 복소 홀로그램 생성
2. 생성된 복소 홀로그램을 DPAC를 통해 위상 홀로그램으로 변환
3. 만들어진 위상 홀로그램을 네트워크에 입력하여 최종 위상 홀로그램 결과 예측
4. 예측한 위상 홀로그램을 ASM을 통해 다시 복원하여 원본 이미지와 비교
5. L1 loss를 통해 error back propagation

------
## Experiment
![images_Experiment](https://user-images.githubusercontent.com/34412522/173362062-58c897a5-2a97-4b88-b5b8-7be416d7c58c.png)

1. Generate Phase-only Hologram with GS-algorithm
2. Generate Phase-only Hologram with Deep Neural Network training with our strategy
3. Numerically reconstruct the result of 1. and 2.
4. Compare the result of 3. with the original image in terms of Image Quality (PSNR)
5. Compare the 2. with 1. in terms of generating time of Phase-only Hologram

## Experiment Settings
- ASM Propagation:
  - SLM pixel pitch: 8㎛
  - Propagation distance: 1㎝
  - Wavelength: 520㎚ (Green)

- GS algorithm:
  - Initial Phase: 0~1 Uniform Random Distribution
  - Iteration: 500

- Neural Network:
  - Criterion: L1 loss
  - Optimizer: Adam optimizer
  - Learning Rate: 0.001 (decay 0.1 every 100 epoch)
  - Epoch: 200


------
## Result
![images_Result](https://user-images.githubusercontent.com/34412522/173363473-d500f32b-f18a-4464-83cd-0707fbca3dbe.png)

- (a): Original Image
- (b): Reconstruction Image of GS algorithm optimized 500 times.
- (c): Reconstruction Image of Neural Network trained 200 epochs.

GS 알고리즘은 원본 이미지에 비해 잡음이 짙게 끼고 Intensity가 많이 낮은 모습을 확인할 수 있다. 반면, 앞서 제안한 학습 전략으로 network를 training한 결과, checkerboard artifact가 끼어있지만, GS와는 다르게 밝기가 원본과 비슷하고 더 선명한 품질의 이미지를 얻을 수 있었다.

|  | GS algorithm | Proposed Method |
|---|---:|---:|
|Generating Time(s)|5.92|**0.027**|
|PSNR(dB)|11.89|**16.07**|


------
## Using Library
<img src="https://img.shields.io/badge/numpy-1.20.3-yellowgreen"/> 
<img src="https://img.shields.io/badge/opencv-4.5.5.64-yellowgreen"/> 
<img src="https://img.shields.io/badge/pytorch-1.11.0-red"/> 
<img src="https://img.shields.io/badge/torchvision-0.12.0-red"/> 
....


------
## Directory
- ASM_propagation.py: propagate light from image to holo plane (or holo to image plane) with Angular Spectrum Method (ASM)
- GSalgorithm.py: generate Phase-only Hologram with GS-algorhtm (iterative optimization)
- HGNTranspose: Hologram Generation Network using Transposed Convolution
- TestHGNTranspose.py: test trained network
- TrainHGNTranspose.py: train network
- dataset.py: load train or test dataset
- dataset_generator: generate dataset by DIV2K dataset
- utils: etc.


------
## Reference
[1] GERCHBERG, Ralph W. “A practical algorithm for the determination of phase from image and diffraction plane pictures.” Optik, 1972, 35: 237-246.

[2] HSUEH, Chung-Kai; SAWCHUK, Alexander A. “Computer-generated double-phase holograms.” Applied optics, 1978, 17.24: 3874-3883.

[3] LEE, Juhyun et al. “Deep neural network for multi-depth hologram generation and its training strategy.” Optics Express, 2020, 28.18: 27137-27154.

[4] MAIMONE, Andrew; GEORGIOU, Andreas; KOLLIN, Joel S. “Holographic near-eye displays for virtual and augmented reality.” ACM Transactions on Graphics (Tog), 2017, 36.4: 1-16.

[5] AGUSTSSON, Eirikur; TIMOFTE, Radu. “Ntire 2017 challenge on single image super-resolution: Dataset and study.” In: Proceddings of the IEEE conference on computer vision and pattern recognition workshops. 2017. p. 126-135.
