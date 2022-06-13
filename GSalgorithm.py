import os
import cv2
import utils
from tqdm import tqdm
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from ASM_propagation import propagation_ASM
from dataset import HologramDataset


# propagation setting
color_mode = 1
cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
feature_size = (8.0 * um, 8.0 * um)
prop_dist = (1.0 * cm, 1.0 * cm, 1.0 * cm)[color_mode]
wavelength = (638 * nm, 520 * nm, 450 * nm)[color_mode]


# gerchberg_saxton algorithm
def gerchberg_saxton(target_amp, init_phase,
                     num_iters, feature_size, wavelength, prop_dist, precomputed_H_f=None, precomputed_H_b=None):
    # initial phase
    real, imag = utils.polar_to_rect(torch.ones_like(init_phase), init_phase)
    slm_field = torch.complex(real, imag)

    # run the GS algorithm
    psnr_list = []
    tqdm_range = tqdm(range(num_iters))
    start = time.time()
    for _ in tqdm_range:
        # SLM plane to image plane
        recon_field = propagation_ASM(slm_field, feature_size, wavelength, prop_dist, precomputed_H=precomputed_H_f)

        # calculate PSNR
        recon_field_abs = torch.abs(recon_field)
        recon_field_abs = recon_field_abs / torch.amax(recon_field_abs)
        psnr = utils.calcPSNR(recon_field_abs, target_amp)
        psnr_list.append(psnr)
        tqdm_range.set_postfix({"PSNR": psnr.item()})

        # replace amplitude at the image plane
        recon_real, recon_imag = utils.polar_to_rect(target_amp, recon_field.angle())
        recon_field = torch.complex(recon_real, recon_imag)

        # image plane to SLM plane
        slm_field = propagation_ASM(recon_field, feature_size, wavelength, -prop_dist, precomputed_H=precomputed_H_b)

        # amplitude constraint at the SLM plane
        slm_real, slm_imag = utils.polar_to_rect(torch.ones_like(target_amp), slm_field.angle())
        slm_field = torch.complex(slm_real, slm_imag)
    end = time.time()

    # return phases
    return torch.angle(slm_field), np.array(psnr_list), end - start


def train(color_mode=1, num_iters=1000):
    print(f"-----------------------GS-{num_iters} START!-----------------------")
    os.makedirs(f"./experiments/GS-{num_iters}/phase/", exist_ok=True)
    os.makedirs(f"./experiments/GS-{num_iters}/recon/", exist_ok=True)
    dataset = HologramDataset(color_mode=color_mode, mode="valid")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    total_time_list = []
    total_psnr_list = []
    precomputed_H_f = None
    precomputed_H_b = None

    for idx, data in enumerate(dataloader):
        target_amp = data["label"]
        init_phase = (-0.5 + 1.0 * torch.rand(1, 1, *target_amp.size()))

        if precomputed_H_f is None:
            precomputed_H_f = propagation_ASM(torch.empty(*target_amp.shape, dtype=torch.complex64),
                                              feature_size, wavelength, prop_dist, return_H=True)
        if precomputed_H_b is None:
            precomputed_H_b = propagation_ASM(torch.empty(*target_amp.shape, dtype=torch.complex64),
                                              feature_size, wavelength, prop_dist, return_H=True)

        # GS-algorithm
        final_phase, psnr_list, generating_time = gerchberg_saxton(target_amp, init_phase,
                                                                   num_iters, feature_size, wavelength, prop_dist,
                                                                   precomputed_H_f=precomputed_H_f, precomputed_H_b=precomputed_H_b)
        total_time_list.append(generating_time)
        total_psnr_list.append(psnr_list)

        # Save Phase-only Hologram
        final_phase_8bit_inverted = utils.phasemap_8bit(final_phase)
        cv2.imwrite(f"./experiments/GS-{num_iters}/phase/{data['filename'][0][:-4]}.png", final_phase_8bit_inverted)

        # Reconstruction
        final_phase = torch.Tensor(((final_phase_8bit_inverted / 255) * 2 * torch.pi) - torch.pi)
        output_amp = propagation_ASM(torch.exp(1j * final_phase), feature_size, wavelength, prop_dist, precomputed_H=precomputed_H_f)
        output_amp = torch.abs(output_amp) / torch.amax(torch.abs(output_amp))
        output_amp_8bit = (output_amp * 255).squeeze().numpy().astype(np.uint8)
        cv2.imwrite(f"./experiments/GS-{num_iters}/recon/{data['filename'][0][:-4]}_recon.png", output_amp_8bit)

        if (idx + 1) % 100 == 0:
            print(f"{idx + 1}번째 이미지 위상 홀로그램 생성 완료")

    np.save(f"./experiments/GS-{num_iters}/time.npy", np.array(total_time_list))
    np.save(f"./experiments/GS-{num_iters}/psnr.npy", np.array(total_psnr_list))
    print(f"-----------------------GS-{num_iters} COMPLETE!-----------------------")


if __name__ == "__main__":
    print(np.mean(np.load("./experiments/HGNTranspose/psnr.npy")))
