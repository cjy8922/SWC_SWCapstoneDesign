import os
import glob
import random
import numpy as np
import cv2


random.seed(7777)
os.makedirs("./data/ASM/train/label", exist_ok=True)  # 원본 이미지 저장
os.makedirs("./data/ASM/train/input", exist_ok=True)  # 잡음 이미지 저장
os.makedirs("./data/ASM/valid/label", exist_ok=True)  # 원본 이미지 저장
os.makedirs("./data/ASM/valid/input", exist_ok=True)  # 잡음 이미지 저장

img_size = 256
img_margin = 0.2
num_image = 3
iteration = 5000


def gaussianNoise(image, loc=0, scale=0.5):
    gauss = np.random.normal(loc, scale, image.shape).astype('uint8')
    noise_img = cv2.add(image, gauss)
    return noise_img


def generator():
    train_file_list = sorted(glob.glob("./data/DIV2K/DIV2K_train_HR/*.png"))
    val_file_list = sorted(glob.glob("./data/DIV2K/DIV2K_valid_HR/*.png"))

    for mode, file_list in zip(["train", "valid"], [train_file_list, val_file_list]):
        for num in range(num_image):
            for idx, file_path in enumerate(file_list):
                file_name = file_path.split("/")[-1]

                # grayscale 이미지 불러와 margin 자르기 (이미지 변화가 큰 부분만 취하기 위함)
                recon_img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                height, width = recon_img.shape[:2]
                height_margin = int(height * img_margin)
                width_margin = int(width * img_margin)
                recon_img = recon_img[height_margin: height - (height_margin + 1), width_margin: width - (width_margin + 1), :]

                # 이미지 랜덤으로 256 * 256 크기로 자르기 (Original image: recon_img)
                height, width = recon_img.shape[:2]
                height_pixel = random.randint(0, height - (img_size + 1))
                width_pixel = random.randint(0, width - (img_size + 1))
                recon_img = recon_img[height_pixel: height_pixel + img_size, width_pixel: width_pixel + img_size, :]

                # Gaussian Noise 추가
                noise_img = gaussianNoise(recon_img, 0, 1.0)

                # 이미지 저장
                cv2.imwrite(f"./data/ASM/{mode}/label/{file_name[:-4]}_{num + 1}_label.png", recon_img)
                cv2.imwrite(f"./data/ASM/{mode}/input/{file_name[:-4]}_{num + 1}_input.png", noise_img)

                if (idx + 1) % 100 == 0:
                    print(f"{idx + 1}번째 이미지 {num + 1}개 생성 완료")

        print(f"    {mode} data generation complete!    ")


if __name__ == "__main__":
    generator()
