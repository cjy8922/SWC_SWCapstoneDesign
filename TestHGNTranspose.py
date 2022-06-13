import os
import argparse
from glob import glob
from tqdm import tqdm
from time import time
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
import utils
from ASM_propagation import propagation_ASM
from dataset import HologramDataset
from HGNTranspose import HologramGenerator

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
parser = argparse.ArgumentParser()

# basic parameters
parser.add_argument("--name", type=str, default="HGNTranspose_down2", help="name of the experiment.")
parser.add_argument("--gpu_ids", type=str, default="2", help="gpu ids. use -1 for CPU")

# training parameters
parser.add_argument("--color_mode", type=int, default=1, help="choose color channel (blue:0, green:1, red:2")
parser.add_argument("--n_epochs", type=int, default=200, help="the number of epochs with the initial learning rate")
parser.add_argument("--batch_size", type=int, default=1, help="the number of batch size")
parser.add_argument("--lr", type=float, default=0.001, help="initial learning rate for adam")
parser.add_argument("--gamma", type=float, default=0.1, help="multiplicative factor of learning rate decay")
parser.add_argument("--lr_decay_iters", type=int, default=100, help="multiply by a gamma every lr_decay_iters epoch")
opt = parser.parse_args()

# set seed
random_seed = 7777
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# propagation settings
cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
feature_size = (8.0 * um, 8.0 * um)
prop_dist = (1.0 * cm, 1.0 * cm, 1.0 * cm)[opt.color_mode]
wavelength = (638 * nm, 520 * nm, 450 * nm)[opt.color_mode]


def test():
    # Set training option
    device = torch.device(f"cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")
    os.makedirs(f"./experiments/{opt.name}/phase", exist_ok=True)
    os.makedirs(f"./experiments/{opt.name}/recon", exist_ok=True)

    # Define Network
    print("Model: Hologram Generator")
    holoGenerator = HologramGenerator(rgb_channel=1, hidden_channel=32, sampling_block=2, residual_block=5).to(device)
    model_checkpoint = torch.load(glob(f"./experiments/{opt.name}/model/*.pth")[-1])
    holoGenerator.load_state_dict(model_checkpoint["model_state_dict"])

    # Define Dataloader
    print("Load Dataset (Mode: phase)")
    val_dataset = HologramDataset(color_mode=opt.color_mode, mode="valid")
    val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False)

    print(f"Start Validation: {opt.name}")
    # Validation
    holoGenerator.eval()
    val_psnr = 0
    tqdm_val_dataloader = tqdm(enumerate(val_dataloader), desc="Validation")

    time_list = []
    psnr_list = []
    for i, data in tqdm_val_dataloader:
        input = data["label"].float().to(device)
        label = data["label"].float().to(device)

        # Input Phase-only Hologram (DPAC)
        start = time()
        pred_H = propagation_ASM(input, feature_size, wavelength, -prop_dist, return_H=True)
        input_complex_hologram = propagation_ASM(input, feature_size, wavelength, -prop_dist, precomputed_H=pred_H)
        input_phase = utils.double_phase_coding(input_complex_hologram)

        # Predict Phase
        output_phase = holoGenerator(input_phase)
        time_list.append(time() - start)
        final_phase_8bit = utils.phasemap_8bit(output_phase, inverted=False)
        cv2.imwrite(f"./experiments/{opt.name}/phase/{data['filename'][0][:-4]}_phase.png", final_phase_8bit)

        # Predict Recon
        final_phase_8bit = torch.Tensor(final_phase_8bit).float().to(device)
        quantized_phase = torch.exp(1j * ((final_phase_8bit / 255) * 2 * torch.pi - torch.pi))
        pred_H2 = propagation_ASM(quantized_phase, feature_size, wavelength, prop_dist, return_H=True)
        output_recon = propagation_ASM(quantized_phase, feature_size, wavelength, prop_dist, precomputed_H=pred_H2)
        output_recon = (torch.abs(output_recon) / torch.amax(torch.abs(output_recon)))

        # Calculate PSNR
        psnr_list.append(utils.calcPSNR(output_recon, label).item())
        val_psnr += utils.calcPSNR(output_recon, label).item()

        tqdm_val_dataloader.set_postfix({
            "Batch": f"[{i + 1}/{len(val_dataloader)}]",
            "PSNR": f"{val_psnr / (i + 1):.6f}",
        })

        # Save Recon Image
        final_recon_8bit = (output_recon * 255).cpu().detach().squeeze().numpy().astype(np.uint8)
        cv2.imwrite(f"./experiments/{opt.name}/recon/{data['filename'][0][:-4]}_recon.png", final_recon_8bit)
    np.save(f"./experiments/{opt.name}/time.npy", np.array(time_list))
    np.save(f"./experiments/{opt.name}/psnr.npy", np.array(psnr_list))


if __name__ == "__main__":
    test()
    print(np.mean(np.load("./experiments/HGNTranspose_down2/time.npy")))
    print(np.mean(np.load("./experiments/HGNTranspose_down2/psnr.npy")))
