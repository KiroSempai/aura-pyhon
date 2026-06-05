import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AuraCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 32)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def process_image(self, image_np):
        if image_np is None:
            return None
        if len(image_np.shape) == 2:
            image_np = np.stack([image_np] * 3, axis=2)
        if image_np.shape[2] == 1:
            image_np = np.repeat(image_np, 3, axis=2)

        tensor = torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0

        _, _, h, w = tensor.shape
        if h != 256 or w != 256:
            tensor = F.interpolate(tensor, size=(256, 256), mode="bilinear", align_corners=False)

        with torch.no_grad():
            features = self.forward(tensor)

        return features.squeeze(0).numpy()
