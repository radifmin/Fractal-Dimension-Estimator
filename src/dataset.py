import torch
from torch.utils.data import Dataset
from torchvision import transforms
from math import log
import random
from fractal_image_generator import generate_fractal_image

class FractalDataset(Dataset):
    def __init__(self, num_samples, D_range=(1.0, 2.0), fractal_depth=7):
        self.num_samples = num_samples
        self.D_range = D_range
        self.size = 2 ** fractal_depth
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Randomly select a fractal dimension from the range
        D = random.uniform(*self.D_range)
        # 2^depth = size
        depth = log(self.size, 2)
        depth = int(depth)
        image = generate_fractal_image(D=D, depth=depth)
        image = image.squeeze()
        image = self.transform(image)
        return image, torch.tensor(D, dtype=torch.float32)
