import glob
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class HologramDataset(Dataset):
    def __init__(self, color_mode=1, mode="train"):
        self.color_mode = color_mode
        self.label = sorted(glob.glob(f"./data/ASM/{mode}/label/*.png"))
        self.toTensor = transforms.ToTensor()

    def __getitem__(self, index):
        label_img = cv2.split(cv2.imread(self.label[index], cv2.IMREAD_COLOR))[self.color_mode] / 255
        label_img = torch.from_numpy(label_img)
        label_img = torch.reshape(label_img, (1, *label_img.size()))

        result = {
            "input": label_img,
            "label": label_img,
            "filename": self.label[index].split("/")[-1],
        }
        return result

    def __len__(self):
        return len(self.label)
