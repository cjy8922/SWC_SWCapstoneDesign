# Deep Learning-based Phase-only Hologram Generation
이 Repository는 경희대학교 2022년 1학기 소프트웨어융합 캡스톤디자인 수업의 일환으로 만들어졌습니다.
- 주저자: 차준영
- 공동저자: 반현민
- 지도교수: 황효석 교수님, 김휘용 교수님

## Overview 
광학 문자 인식 Task로 주어진 데이터를 보고 어떤 문제가 있고, OCR 모델의 정확도를 향상시키기 위해선 이 문제들을 어떻게 해결해야 하는지 위주로 접근했습니다.

## Reference
- TPS-Resnet-BiLSTM-Attn: BAEK, Jeonghun, et al. What is wrong with scene text recognition model comparisons? dataset and model analysis. In: Proceedings of the IEEE/CVF international conference on computer vision. 2019. p. 4715-4723. (code: https://github.com/clovaai/deep-text-recognition-benchmark)
- SATRN: LEE, Junyeop, et al. On recognizing texts of arbitrary shapes with 2D self-attention. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops. 2020. p. 546-547. (code: https://github.com/Media-Smart/vedastr)

## Using Library
<img src="https://img.shields.io/badge/lmdb-1.3.0-green"/> 
<img src="https://img.shields.io/badge/exrex-0.10.5-yellowgreen"/> 
<img src="https://img.shields.io/badge/nltk-3.7-yellowgreen"/> 
<img src="https://img.shields.io/badge/numpy-1.23.4-blue"/> 
<img src="https://img.shields.io/badge/opencv-4.6.0.66-blue"/> 
<img src="https://img.shields.io/badge/albumentations-1.3.0-blue"/> 
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
