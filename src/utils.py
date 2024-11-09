import torch
from model import FractalDimensionCNN
from torchvision import transforms
from PIL import Image
Image.MAX_IMAGE_PIXELS = None


def load_model_checkpoint(checkpoint_path):
    model = FractalDimensionCNN()
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    fractal_depth = checkpoint['fractal_depth']
    model.eval()
    return model, fractal_depth

def preprocess_image(image_path, fractal_depth):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((2**fractal_depth, 2**fractal_depth)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = Image.open(image_path)
    return transform(image).unsqueeze(0)