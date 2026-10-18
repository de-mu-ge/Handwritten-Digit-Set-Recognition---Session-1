import torch.nn as nn
# input : (B, 28, 28)
# output: 10 (0, 9)

def forward(x):

    return x


class Model2828(nn.Module):
    def __init__(self):
        super(Model2828, self).__init__()

        self.layer1 = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 16, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 8, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(8),
            nn.Dropout2d(0.25),
            nn.Conv2d(8, 4, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(4, 2, kernel_size=2, stride=1, padding=0),
            nn.Dropout2d(0.25),
            nn.Conv2d(2, 1, kernel_size=2, stride=1, padding=0),

        )

        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )

        self.layer3 = nn.Sequential(
            nn.Linear(16, 10),
        )


        pass

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)

        # print(x.shape)

        x = self.layer3(x)

        return x


if __name__ == "__main__":

    import torch
    model = Model2828()

    x = torch.randn(1,1,28,28).float()

    y = model(x)
