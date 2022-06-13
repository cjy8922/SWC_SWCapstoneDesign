import os
import argparse
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
import utils
from ASM_propagation import propagation_ASM
from dataset import HologramDataset
from HGNTranspose import HologramGenerator

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
parser = argparse.ArgumentParser()

# basic parameters
parser.add_argument("--name", type=str, default="HGNTranspose_down2_noDPAC", help="name of the experiment.")
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


def train():
    # Set training option
    device = torch.device(f"cuda") if torch.cuda.is_available() else torch.device("cpu")
    os.makedirs(f"./experiments/{opt.name}/model", exist_ok=True)
    os.makedirs(f"./experiments/{opt.name}/log", exist_ok=True)
    os.makedirs(f"./experiments/{opt.name}/output", exist_ok=True)
    writer = SummaryWriter(f"./experiments/{opt.name}/log/")

    # Define Network
    print("Model: Hologram Generator")
    holoGenerator = HologramGenerator(rgb_channel=1, hidden_channel=32, sampling_block=2, residual_block=5).to(device)

    # Define Loss Function
    print("Loss: L1 loss (MAE loss)")
    holoCriterion = nn.L1Loss().to(device)

    # Define Optimizer
    print("Optimizer: Adam Optimizer")
    holoOptimizer = optim.Adam(holoGenerator.parameters(), lr=opt.lr)

    # Define Learning Rate Scheduler
    holoScheduler = optim.lr_scheduler.StepLR(holoOptimizer, step_size=opt.lr_decay_iters, gamma=opt.gamma)

    # Define Dataloader
    print("Load Dataset (Mode: phase)")
    train_dataset = HologramDataset(color_mode=opt.color_mode, mode="train")
    val_dataset = HologramDataset(color_mode=opt.color_mode, mode="valid")
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False)

    print(f"Start Training: {opt.name}")
    for epoch in range(opt.n_epochs):
        # Training
        holoGenerator.train()
        train_loss, train_psnr = 0, 0
        tqdm_train_dataloader = tqdm(enumerate(train_dataloader), desc="Training")

        for i, data in tqdm_train_dataloader:
            input = data["label"].float().to(device)
            label = data["label"].float().to(device)

            # Input Phase-only Hologram (DPAC)
            pred_H = propagation_ASM(input, feature_size, wavelength, -prop_dist, return_H=True)
            input_complex_hologram = propagation_ASM(input, feature_size, wavelength, -prop_dist, precomputed_H=pred_H)
            input_phase = torch.angle(input_complex_hologram)
            # input_phase = utils.double_phase_coding(input_complex_hologram)

            # Training Network
            holoOptimizer.zero_grad()
            output_phase = holoGenerator(input_phase)  # predict phase-only hologram
            pred_H2 = propagation_ASM(output_phase, feature_size, wavelength, prop_dist, return_H=True)  # calculate ASM kernel
            pred_H2.requires_grad = False
            output_recon = propagation_ASM(torch.exp(1j * output_phase), feature_size, wavelength, prop_dist, precomputed_H=pred_H2)  # numerical reconstruction
            output_recon = torch.abs(output_recon) / torch.amax(torch.abs(output_recon))
            train_recon_loss = holoCriterion(output_recon, label)
            train_recon_loss.backward()
            holoOptimizer.step()
            train_loss += train_recon_loss.item()

            # Calculate PSNR
            train_psnr += utils.calcPSNR(output_recon, label)

            # Logging
            tqdm_train_dataloader.set_postfix({
                "Epoch": f"[{epoch + 1}/{opt.n_epochs}]",
                "Batch": f"[{i + 1}/{len(train_dataloader)}]",
                "Recon Loss": f"{train_loss / (i + 1):.6f}",
                "PSNR": f"{train_psnr / (i + 1):.6f}",
                "lr": holoOptimizer.param_groups[0]["lr"],
            })
        holoScheduler.step()

        # Validation
        holoGenerator.eval()
        valid_loss, valid_psnr = 0, 0
        tqdm_val_dataloader = tqdm(enumerate(val_dataloader), desc="Validation")

        with torch.no_grad():
            for i, data in tqdm_val_dataloader:
                input = data["label"].float().to(device)
                label = data["label"].float().to(device)

                # Input Phase-only Hologram (DPAC)
                pred_H = propagation_ASM(input, feature_size, wavelength, -prop_dist, return_H=True)
                input_complex_hologram = propagation_ASM(input, feature_size, wavelength, -prop_dist, precomputed_H=pred_H)
                input_phase = utils.double_phase_coding(input_complex_hologram)

                # Predict Phase and Recon
                output_phase = holoGenerator(input_phase)
                pred_H2 = propagation_ASM(output_phase, feature_size, wavelength, prop_dist, return_H=True)
                output_recon = propagation_ASM(torch.exp(1j * output_phase), feature_size, wavelength, prop_dist, precomputed_H=pred_H2)
                output_recon = torch.abs(output_recon) / torch.amax(torch.abs(output_recon))
                valid_recon_loss = holoCriterion(output_recon, label)
                valid_loss += valid_recon_loss.item()

                # Target Phase Only
                pred_H = propagation_ASM(label, feature_size, wavelength, -prop_dist, return_H=True)
                target_complex_hologram = propagation_ASM(label, feature_size, wavelength, -prop_dist, precomputed_H=pred_H)
                target_phase = torch.angle(target_complex_hologram)

                # Calculate PSNR
                valid_psnr += utils.calcPSNR(output_recon, label)

                # Display and Save image (Real, Image, Phase, Recon)
                if (i + 1) % 10 == 0:
                    phase = torch.cat((output_phase, target_phase), -2)
                    recon = torch.cat((output_recon, label), -2)
                    writer.add_images("Phase", phase, epoch + 1)
                    writer.add_images("Recon", recon, epoch + 1)
                    save_image(torch.cat((phase, recon), -2), f"./experiments/{opt.name}/output/{epoch + 1}.png")

                tqdm_val_dataloader.set_postfix({
                    "Epoch": f"[{epoch + 1}/{opt.n_epochs}]",
                    "Batch": f"[{i + 1}/{len(val_dataloader)}]",
                    "Recon Loss": f"{valid_loss / (i + 1):.6f}",
                    "PSNR": f"{valid_psnr / (i + 1):.6f}",
                })

        # Logging
        writer.add_scalars("Recon Loss", {
            "Training Loss": train_loss / len(train_dataloader),
            "Validation Loss": valid_loss / len(val_dataloader),
        }, epoch + 1)

        writer.add_scalars("PSNR", {
            "Training PSNR": train_psnr / len(train_dataloader),
            "Validation PSNR": valid_psnr / len(val_dataloader),
        }, epoch + 1)

        # Save model
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": holoGenerator.state_dict(),
                "optimizer_state_dict": holoOptimizer.state_dict(),
            }, f"./experiments/{opt.name}/model/model_phase_{epoch + 1}.pth")
            print(f"───────────────────────S A V E   M O D E L {epoch + 1}───────────────────────")


if __name__ == "__main__":
    train()
