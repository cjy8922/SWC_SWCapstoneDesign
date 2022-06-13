import torch
import torch.nn as nn
from torchsummary import summary


class HologramGenerator(nn.Module):
    def __init__(self, rgb_channel=1, hidden_channel=32, sampling_block=2, residual_block=5):
        super(HologramGenerator, self).__init__()
        self.rc = rgb_channel
        self.hc = hidden_channel
        self.sb = sampling_block
        self.rb = residual_block

        self.conv1 = nn.Conv2d(self.rc, self.hc, kernel_size=(5, 5), stride=(1, 1), padding=2)
        self.relu1 = nn.LeakyReLU(0.02)

        def _make_layers(sb, rb):
            layers = []
            for i in range(sb):
                layers.append(DownSampleBlock(self.hc, self.hc * 2))
                self.hc = self.hc * 2
            for i in range(rb):
                layers.append(ResidualBlock(self.hc))
            for i in range(sb):
                layers.append(UpSampleBlock(self.hc, self.hc // 2))
                self.hc = self.hc // 2
            return layers

        self.layers = nn.Sequential(*_make_layers(self.sb, self.rb))
        self.ob = OutputBranch(self.hc)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.layers(x)
        x = self.ob(x)
        return x


class DownSampleBlock(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(DownSampleBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm2d(output_channel),
            nn.LeakyReLU(0.02),
        )

    def forward(self, x):
        return self.layers(x)


class UpSampleBlock(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(UpSampleBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(input_channel, output_channel, kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(output_channel),
            nn.LeakyReLU(0.02),
        )

    def forward(self, x):
        return self.layers(x)


class ResidualBlock(nn.Module):
    def __init__(self, input_channel):
        super(ResidualBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channel, input_channel, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(input_channel),
            nn.LeakyReLU(0.02),
            nn.Conv2d(input_channel, input_channel, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(input_channel),
            nn.LeakyReLU(0.02),
        )
        
    def forward(self, x):
        return x + self.layers(x)


class OutputBranch(nn.Module):
    def __init__(self, output_channel):
        super(OutputBranch, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(output_channel, 1, kernel_size=(5, 5), stride=(1, 1), padding=2),
        )
        self.hardt = nn.Hardtanh(-torch.pi, torch.pi)

    def forward(self, x):
        return self.hardt(self.layers(x))


if __name__ == "__main__":
    block = HologramGenerator(rgb_channel=1, hidden_channel=32, sampling_block=2, residual_block=5).cuda()
    summary(block, (1, 256, 256))
