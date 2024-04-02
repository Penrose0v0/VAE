import torch
import cv2
import numpy as np
import os

from network import VAE
from utils import setup_seed

setup_seed(3407)

fmt = "----- {:^25} -----"
model_path = "./weights/best.pth"

def generate():
    model.eval()
    h, c, sample = 0, 0, torch.empty(0).to(device)
    with torch.no_grad():
        while True:
            y, h, c = model.sample(h, c)
            sample = torch.cat((sample, y))
            print(sample.shape[0])
            if sample.shape[0] < 28:
                continue

            # Post-process
            image = sample.permute(1, 0).cpu().detach().numpy()
            image = cv2.resize(image, (sample.shape[0] * 10, 280))
            cv2.imshow("generated", image)
            key = cv2.waitKey(0)
            if key == ord('q'):
                cv2.destroyAllWindows()
                break

    # Save images
    image = (image * 0.3081 + 0.1307) * 255.
    image = np.where(image < 0, 0, image).astype('uint8')
    cv2.imshow("generated", image)
    cv2.imwrite(os.path.join("outputs", f"generated image.jpg"), image)


# Set device
device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Create neural network
print(fmt.format("Create neural network"))
model = VAE()

print(f"Loading pretrained model: {model_path}")
model.load_state_dict(torch.load(model_path))
model.to(device)
print()

# Start generating
print(fmt.format("Start generating") + '\n')
while True:
    generate()
    
