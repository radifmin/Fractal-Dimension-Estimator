import torch
from utils import load_model_checkpoint, preprocess_image

def predict_fractal_dimension(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
    return output.item()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Predict fractal dimension of an image')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('model_path', type=str, help='Path to the checkpoint of the model')
    args = parser.parse_args()

    model, fractal_depth = load_model_checkpoint(args.model_path)
    image_tensor = preprocess_image(args.image_path, fractal_depth)

    fd = predict_fractal_dimension(model, image_tensor)
    print(f'Predicted fractal dimension: {fd}')
