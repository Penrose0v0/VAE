import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import cv2
import numpy as np
import os
import argparse
import time

from network import VAE, VAELoss
from dataset import VAEDataset
from utils import setup_seed, draw_figure, log, convert_seconds

setup_seed(3407)

def train(epoch_num, count=100):
    model.train()
    running_loss, total_loss, total = 0.0, 0.0, 1
    print(f"< Epoch {epoch_num + 1} >")
    for batch_idx, data in enumerate(train_loader):
        # Get data
        inputs = data
        inputs = inputs.to(device)

        optimizer.zero_grad()

        # Forward + Backward + Update
        outputs, loss = model(inputs)
        loss.backward()
        optimizer.step()

        # Calculate loss
        running_loss += loss.item()
        total_loss += loss.item()
        if batch_idx % count == count - 1:
            print('Batch %d\t loss: %.6f' % (batch_idx + 1, running_loss / count))
            running_loss = 0.0
        total += 1

    print('Average loss: %.6f' % (total_loss / total))
    return total_loss / total

def val(epoch_num):
    model.eval()
    h, c, sample = 0, 0, torch.empty(0).to(device)
    with torch.no_grad():
        while True:
            y, h, c = model.sample(h, c)
            sample = torch.cat((sample, y))
            if sample.shape[0] == 28:
                break

    # Save images
    image = sample.permute(1, 0).cpu().detach().numpy()
    image = (image * 0.3081 + 0.1307) * 255.
    image_save = np.where(image < 0, 0, image).astype('uint8')
    cv2.imwrite(os.path.join("outputs", f"epoch {epoch_num + 1}.jpg"), image_save)


if __name__ == "__main__":
    fmt = "----- {:^25} -----"
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='')
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--dataset-path', type=str, default='D:\coding\dataset\mnist\image')
    args = parser.parse_args()

    # Set hyper-parameters
    epochs = args.epoch
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    dataset_path = args.dataset_path
    model_path = args.model_path

    # Set device
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Create neural network
    print(fmt.format("Create neural network"))
    model = VAE()

    # Load pretrained model or create a new model
    if model_path != '':
        print(f"Loading pretrained model: {model_path}")
        model.load_state_dict(torch.load(model_path))
    else:
        print("Creating new model")
    model.to(device)
    print()

    # Define criterion
    criterion = VAELoss()
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

    # Load dataset
    print(fmt.format("Load dataset"))
    train_set = VAEDataset(image_folder=dataset_path)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    print()

    # Start training
    print(fmt.format("Start training") + '\n')
    min_loss = -1
    best_epoch = 0
    epoch_list, loss_list = [], []
    very_start = time.time()
    for epoch in range(epochs):
        start = time.time()

        # Train one epoch
        current_loss = train(epoch)
        val(epoch)

        # Save model
        torch.save(model.state_dict(), f"./weights/current.pth")
        if current_loss < min_loss or min_loss == -1:
            torch.save(model.state_dict(), f"./weights/best.pth")
            print("Update the best model")
            min_loss = current_loss
            best_epoch = epoch + 1

        # Draw figure and log
        epoch_list.append(epoch + 1)
        loss_list.append(current_loss)
        draw_figure(epoch_list, loss_list, "Loss", "./outputs/loss.png")
        log(loss_list, f"log/{very_start}.txt")

        # Elapsed time
        end = time.time()
        use_time = int(end - start)
        print(f"Elapsed time: {use_time // 60}m {use_time % 60}s\n")

    very_end = time.time()
    total_time = int(very_end - very_start)

    print(f"Training finished! Total elapsed time: {convert_seconds(total_time)}, "
          f"Best Epoch: {best_epoch}, Min Loss: {min_loss:.4f}")
