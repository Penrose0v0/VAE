import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import cv2
import os
import time

from network import VAE, VAELoss
from dataset import VAEDataset
from utils import unnormalize_image

seed_value = 3407
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)


# Hyper parameter
batch_size = 2  # 128
lr = 0.00005
epochs = 1000
device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Network
model = VAE().to(device)
# model.load_state_dict(torch.load("./weights/0311.pth"))
criterion = VAELoss()
optimizer = optim.Adam(model.parameters(), lr)

# Prepare dataset
train_set = VAEDataset()
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)


def train(epoch, count=50):
    model.train()
    running_loss = 0.0
    print(f"< Epoch {epoch + 1} >")
    for batch_idx, data in enumerate(train_loader):
        inputs = data
        inputs = inputs.to(device)

        optimizer.zero_grad()

        outputs, mean, logvar = model(inputs)
        loss = criterion(inputs, outputs, mean, logvar)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % count == count - 1:
            print('Batch %d\t loss: %.6f' % (batch_idx + 1, running_loss / count))
            running_loss = 0.0
            # image = outputs[0].permute(1, 2, 0).cpu().detach().numpy()
            # image = cv2.resize(image, (280, 280))
            # cv2.imshow("generation", image)
            # key = cv2.waitKey(0)
            # cv2.destroyAllWindows()

for i in range(epochs):
    start = time.time()

    # Train one epoch
    train(i)

    end = time.time()
    use_time = int(end - start)
    print(f"Elapsed time: {use_time // 60}m {use_time % 60}s\n")

    # Generate image
    if i % 1 == 0:
        sample = model.sample()
        image = sample[0].permute(1, 2, 0).cpu().detach().numpy()
        image_save = unnormalize_image(image).astype('uint8')

        # Save images
        image_save = cv2.cvtColor(image_save, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join("outputs", f"epoch {i+1}.jpg"), image_save)

        # cv2.imshow("generation", image_save)
        # key = cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # if key == ord('q'):
        #     break

    # Save model
    torch.save(model.state_dict(), f"./weights/current.pth")


test = model.sample()
print(test.shape)
