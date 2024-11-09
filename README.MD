# Fractal dimension estimator

The project is a tool for evaluating the fractal dimension of images using a trained neural network. It also includes functionality for generating fractal images with different dimensions (via the external [FractalImageGenerator](https://github.com/radifmin/FractalImageGenerator) repository).
## Description
<!-- A dataset of fractal images is generated during model training using [FractalImageGenerator](https://github.com/radifmin/FractalImageGenerator) repository. -->
The main components of the project:

* **Fractal image generation:** creating images with a given fractal dimension for training and testing and testing the model.
* **Neural Network:** convolutional neural network (CNN) for predicting fractal dimension of an image.

## How to Run

1. **Clone the repository:**

    ```bash
    git clone https://github.com/radifmin/Fractal-Dimension-Estimator.git
    cd
    ```

2. **Install the requirements:**

    ```bash
    pip install -r requirements.tzt
    ```

3. **Install an external repository for generating fractals:**

    ```bash
    pip install git+https://github.com/radifmin/FractalImageGenerator.git
    ```

## Usage

To train model from scratch, run the following command:

```bash
python src/train.py --num_samples_train 128 --num_samples_val 16 --fractal_depth 7 --epochs 50 --learning_rate 0.001 --batch_size 32 --patience 5
```

Args:

* **```--num_samples_train```**: number of samples for training.
* **```--num_samples_val```**: number of samples for validaiton.
* **```--fractal_depth```**: depth of the fractals, which determines the size of the image ($2^{depth}$).
* **```--epochs```**: number of epochs.
* **```--learning_rate```**: learning rate.
* **```--batch_size```**: batch size.
* **```--patience```**: the number of epochs for early stopping in the absence of imporovement.

## Prediction of fractal dimension

To predict the fractal dimension of an image, run the following command:

```bash
python src/predict.py images/1.png src/models/FD_estimator_checkpoint.pth
```

Args:

* **```image_path```**: path to the image.
* **```model_path```**: path to the checkpoint of model.

## License

This project is distributed under the MIT license. See the [LICENSE](LICENSE) for more information.
