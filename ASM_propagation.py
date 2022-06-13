import math
import torch
import numpy as np
import torch.fft
import utils


def propagation_ASM(u_in, feature_size, wavelength, z, return_H=False, precomputed_H=None, dtype=torch.float32):
    if precomputed_H is None:
        field_resolution = u_in.size()  # resolution of input field, should be: (num_images, num_channels, height, width, 2)
        num_y, num_x = field_resolution[-2], field_resolution[-1]  # number of pixels
        dy, dx = feature_size  # sampling inteval size
        y, x = (dy * float(num_y), dx * float(num_x))  # size of the field

        # frequency coordinates sampling
        fy = np.linspace(-1 / (2 * dy) + 0.5 / (2 * y), 1 / (2 * dy) - 0.5 / (2 * y), num_y)
        fx = np.linspace(-1 / (2 * dx) + 0.5 / (2 * x), 1 / (2 * dx) - 0.5 / (2 * x), num_x)
        FX, FY = np.meshgrid(fx, fy)

        # transfer function in numpy (omit distance)
        HH = 2 * math.pi * np.sqrt(1 / wavelength**2 - (FX**2 + FY**2))
        HH = torch.tensor(HH, dtype=dtype).to(u_in.device)
        HH = torch.reshape(HH, (1, 1, *HH.size()))
        H_exp = torch.mul(HH, z).to(u_in.device)   # multiply by distance

        # band-limited ASM - Matsushima et al. (2009)
        fy_max = 1 / np.sqrt((2 * z * (1 / y))**2 + 1) / wavelength
        fx_max = 1 / np.sqrt((2 * z * (1 / x))**2 + 1) / wavelength
        H_filter = torch.tensor(((np.abs(FX) < fx_max) & (np.abs(FY) < fy_max)).astype(np.uint8), dtype=dtype).to(u_in.device)

        # get real/img components
        H_real, H_imag = utils.polar_to_rect(H_filter.to(u_in.device), H_exp)
        H = torch.stack((H_real, H_imag), -1)
        H = torch.fft.ifftshift(H)
        H = torch.view_as_complex(H)
    else:
        H = precomputed_H

    if return_H:
        return H
    else:
        U1 = torch.fft.fftshift(torch.fft.fftn(torch.fft.fftshift(u_in)))
        U2 = H * U1
        u_out = torch.fft.fftshift(torch.fft.ifftn(torch.fft.fftshift(U2)))
        return u_out


if __name__ == "__main__":
    from PIL import Image
    import glob
    import matplotlib.pyplot as plt
    from torchvision.transforms import ToTensor, ToPILImage
    path = sorted(glob.glob("./data/ASM/train/label/*.png"))
    toTensor = ToTensor()
    toImage = ToPILImage()

    cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
    feature_size = (6.4 * um, 6.4 * um)
    prop_dist = (1 * cm, 1 * cm, 1 * cm)[0]
    wavelength = (638 * nm, 520 * nm, 450 * nm)[0]

    index = 22
    img1 = toTensor(Image.open(path[index]).convert("L"))
    img2 = toTensor(Image.open(path[1]).convert("L"))
    img = torch.stack((img1, img2), 0)
    print(torch.amax(img1[0]), torch.amin(img1[0]))

    out1 = propagation_ASM(img, feature_size, wavelength, -prop_dist, return_H=True)
    out1 = propagation_ASM(img, feature_size, wavelength, -prop_dist, precomputed_H=out1)
    print(torch.amax(torch.abs(out1[0])), torch.amin(torch.abs(out1[0])))
    print(torch.amax(torch.angle(out1[0])), torch.amin(torch.angle(out1[0])))
    out2 = propagation_ASM(out1, feature_size, wavelength, prop_dist, return_H=True)
    out2 = propagation_ASM(out1, feature_size, wavelength, prop_dist, precomputed_H=out2)
    print(torch.amax(torch.abs(out2[0])), torch.amin(torch.abs(out2[0])))

    mse = torch.mean((torch.abs(out2[0]) - img1) ** 2)
    psnr = 10 * np.log10(1 / mse)
    print(psnr)

    out_angle = toImage(torch.angle(out1[0]))
    out_abs = toImage(torch.abs(out2[0]))
    original = Image.open(path[index]).convert("L")

    plt.subplot(131)
    plt.imshow(original, cmap="gray")
    plt.title("Original")
    plt.subplot(132)
    plt.imshow(out_angle, cmap="gray")
    plt.title("Phase")
    plt.subplot(133)
    plt.imshow(out_abs, cmap="gray")
    plt.title("Recon")
    plt.tight_layout()
    plt.show()

    mse = np.mean((np.array(out_abs) - np.array(original)) ** 2)
    psnr = 10 * np.log10(1 / mse)
