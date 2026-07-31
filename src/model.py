import torch
import torch.nn as nn

class SceneCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(SceneCNN, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 150 -> 75

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 75 -> 37

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 37 -> 18
        )

        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 18 * 18, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.fc_block(x)
        return x

if __name__ == "__main__":
    model = SceneCNN()
    print(model)
    dummy_input = torch.randn(4, 3, 150, 150)  # batch of 4 images
    output = model(dummy_input)
    print("Output shape:", output.shape)