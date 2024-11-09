import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import FractalDimensionCNN
from dataset import FractalDataset
import argparse
from pathlib import Path

base_path = Path(__file__).parent / 'models'
def train_model(num_samples_train, num_samples_val, fractal_depth, epochs, learning_rate, batch_size, patience):
    model = FractalDimensionCNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset = FractalDataset(num_samples=num_samples_train, D_range=(1.0, 2.0), fractal_depth=fractal_depth)
    val_dataset   = FractalDataset(num_samples=num_samples_val,  D_range=(1.0, 2.0), fractal_depth=fractal_depth)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=True)


    base_path.mkdir(parents=True, exist_ok=True) # create directory for saving the best model
    
    no_improvement_epochs = 0
    best_val_loss = float('inf')
    for epoch in range(epochs):
        # training step
        model.train()
        for images, dimensions in train_dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, dimensions.view(-1, 1))
            loss.backward()
            optimizer.step()
        # validation step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, dimensions in val_dataloader:
                outputs = model(images)
                loss = criterion(outputs, dimensions.view(-1,1))
                val_loss += loss

        val_loss /= len(val_dataloader)
        # save the best model
        if val_loss < best_val_loss:
            best_val_loss = loss
            no_improvement_epochs = 0
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'fractal_depth': fractal_depth,
                'loss': val_loss
            }
            torch.save(checkpoint, base_path/'FD_estimator_checkpoint.pth')
        else:
            no_improvement_epochs += 1

        print(f'Epoch [{epoch+1}/{epochs}], validation loss: {val_loss:.4f}')
        if no_improvement_epochs >= patience:
            print(f'Early stopping on epoch {epoch+1}')
            break

    print("Training complete")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the Fractal Dimension Estimator CNN')
    parser.add_argument('--num_samples_train', type=int, default=128, help='Number of train images to create')
    parser.add_argument('--num_samples_val', type=int, default=16, help='Number of validation images to create')
    parser.add_argument('--fractal_depth', type=int, default=7, help='Depth of the fractal image. Created image will have 2^depth by 2^depth size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for data loader')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping if validation loss does not improve in specified number of epochs')

    args = parser.parse_args()

    train_model(args.num_samples_train, args.num_samples_val, args. fractal_depth, args.epochs, args.learning_rate, args.batch_size, args.patience)