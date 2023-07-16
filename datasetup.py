import os
from glob import glob
from PIL import Image
from torch.utils.data import Dataset

class cycleGanDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.transform = transform

        root = os.path.join(root, ("train" if train else "test"))
        self.imgs_A = sorted(glob(os.path.join(root + "A", "*.*")))
        self.imgs_B = sorted(glob(os.path.join(root + "B", "*.*")))

        self.len_A = len(self.imgs_A)
        self.len_B = len(self.imgs_B)
        self.len = max(self.len_A, self.len_B)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        img_A = Image.open(self.imgs_A[index % self.len_A]).convert("RGB")
        img_B = Image.open(self.imgs_B[index % self.len_B]).convert("RGB")

        aug_A = self.transform(img_A)
        aug_B = self.transform(img_B)

        return aug_A, aug_B
