import numpy as np
import torch


def rect_to_polar(real, imag):
    """Converts the rectangular complex representation to polar"""
    mag = torch.pow(real**2 + imag**2, 0.5)
    ang = torch.atan2(imag, real)
    return mag, ang


def polar_to_rect(mag, ang):
    """Converts the polar complex representation to rectangular"""
    real = mag * torch.cos(ang)
    imag = mag * torch.sin(ang)
    return real, imag


def phasemap_8bit(phasemap, inverted=True):
    """convert a phasemap tensor into a numpy 8bit phasemap that can be directly displayed"""
    output_phase = ((phasemap + np.pi) % (2 * np.pi)) / (2 * np.pi)
    if inverted:
        phase_out_8bit = ((1 - output_phase) * 255).round().cpu().detach().squeeze().numpy().astype(np.uint8)  # quantized to 8 bits
    else:
        phase_out_8bit = ((output_phase) * 255).round().cpu().detach().squeeze().numpy().astype(np.uint8)  # quantized to 8 bits
    return phase_out_8bit


def double_phase_coding(complex):
    amplitude = torch.abs(complex)
    angle = torch.angle(complex)
    amplitude = amplitude / torch.amax(amplitude)  # normalize (0~1)

    # double-phase
    phase1 = angle - torch.acos(amplitude)
    phase2 = angle + torch.acos(amplitude)
    phase_out = phase1
    phase_out[..., ::2, 1::2] = phase2[..., ::2, 1::2]
    phase_out[..., 1::2, ::2] = phase2[..., 1::2, ::2]

    phase_out -= torch.mean(phase_out)
    return (phase_out + torch.pi) % (2 * torch.pi) - torch.pi


def calcPSNR(output, target):
    """calculate PSNR (Peak signal: 1)"""
    mse = torch.mean((output - target) ** 2)
    psnr = 20 * torch.log10(1 / torch.sqrt(mse))
    return psnr
