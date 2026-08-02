import torch
import torch.nn as nn

class SceneCNN(nn.Module):
    def __init__(self, num_classes=6, dropout=0.0, filters=(16, 32, 64)):
        super(SceneCNN, self).__init__()

        f1, f2, f3 = filters

        self.conv_block = nn.Sequential(
            nn.Conv2d(3, f1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(f1, f2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(f2, f3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(f3 * 18 * 18, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.fc_block(x)
        return x

if __name__ == "__main__":
    model = SceneCNN()
    print(model)