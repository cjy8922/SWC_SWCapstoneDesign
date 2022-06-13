# Deep Learning-based Phase-only Hologram Generation
This project is based on "Neural Holography with Camera-in-the-loop Training"

https://github.com/computational-imaging/neural-holography

## Overview
디지털 홀로그래피는 3차원 물체에서 나온 빛의 위상과 진폭 정보로 시각적인 정보를 기록하고 복원하는 기술이다. 이 홀로그램을 디스플레이 하기 위해선 디지털 공간광변조기(SLM; Spatial Light Modulator)가 필요하지만, 현재 기술적 한계로 빛의 진폭 혹은 위상만 디스플레이 할 수 있다. 이때 진폭 정보를 복원하는 진폭 공간광변조기보다 좀 더 빛을 효율적으로 사용하여 물체를 표현할 수 있는 위상 공간광변조기가 주로 사용되고 있으며, 이에 맞추어 빛의 위상만 기록하여 물체를 더 잘 표현할 수 있도록 위상 홀로그램(PoH; Phase-only Hologram)을 잘 만들기 위한 연구가 지속되고 있다. 

현재 위상 홀로그램을 잘 만드는 방법으로 대표적으로 반복 최적화 알고리즘인 GS algorithm (Gerchberg-Saxton algorithm)이 있다. 이 방법은 초반 0~1 사이의 Uniform Distribution으로 Random Phase를 만든 후, FFT / IFFT를 반복적으로 사용하여 입력 이미지의 진폭과 위상 정보가 위상만으로 적절히 표현되도록 만드는 기법이다. 이 방법을 충분히 반복하면 좋은 화질의 위상 홀로그램 이미지를 생성할 수 있다. 하지만 이런 반복 최적화 알고리즘은 계산 복잡도가 높고, 이미지 한 장씩 처리해야 하므로 병렬처리가 불가능하여 좋은 화질의 홀로그램 이미지를 생성하는 데 오랜 시간이 걸린다는 단점이 있다. 따라서 이 방법을 이용해 위상 홀로그램을 실시간으로 최적화하여 생성하는 데 한계가 있다. 이러한 이유로 딥러닝을 이용한 위상 홀로그램 생성 기법이 필요해지게 된다. 병렬처리가 가능한 딥러닝 방법을 이용하면 충분히 잘 학습된 신경망을 통해 실시간으로 좋은 품질의 위상 홀로그램을 생성할 수 있다. 따라서 본 캡스톤디자인에서는 딥러닝 기반으로 위상 홀로그램을 생성하는 신경망의 학습 전략을 수립하고자 한다.

------
## Using Library
<img src="https://img.shields.io/badge/numpy-1.20.3-yellowgreen"/> 
<img src="https://img.shields.io/badge/opencv-4.5.5.64-yellowgreen"/> 
<img src="https://img.shields.io/badge/pytorch-1.11.0-red"/> 
<img src="https://img.shields.io/badge/torchvision-0.12.0-red"/> 
<img src="https://img.shields.io/badge/glob-0.7-blue"/>
<img src="https://img.shields.io/badge/tqdm-4.62.3-blue"/>
....

------
## Directory
- ASM_propagation.py: propagate light from image to holo plane (or holo to image plane) with Angular Spectrum Method (ASM)
- GSalgorithm.py: generate Phase-only Hologram with GS-algorhtm (iterative optimization)
- **HGNTranspose**: Hologram Generation Network using Transposed Convolution
- TestHGNTranspose.py: test trained network
- TrainHGNTranspose.py: train network
- dataset.py: load train or test dataset
- dataset_generator: generate dataset by DIV2K dataset
- utils: etc.


------
## Experiment
![images_Experiment](https://user-images.githubusercontent.com/34412522/173362062-58c897a5-2a97-4b88-b5b8-7be416d7c58c.png)

1. Generate Phase-only Hologram with GS-algorithm
2. Generate Phase-only Hologram with Deep Neural Network training with our strategy
3. Numerically reconstruct the result of 1. and 2.
4. Compare the result of 3. with the original image in terms of Image Quality (PSNR)
5. Compare the 2. with 1. in terms of generating time of Phase-only Hologram

- ASM Propagation Settings:
  - SLM pixel pitch: 8㎛
  - Propagation distance: 1㎝
  - Wavelength: 520㎚ (Green)

- GS algorithm Settings:
  - Initial Phase: 0~1 Uniform Random Distribution
  - Iteration: 500

- Neural Network Settings:
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



