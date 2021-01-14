import torch.nn as nn
import torch.nn.functional as F


class NeuronalNetwork(nn.Module):

## Modelo Convolucional sacado del tutorial
#
#     def __init__(self):
#         super().__init__()
#         #canales de entrada, canales de salida, convolucion cuadrada
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120) #16 canales * imagen de 5x5, 120 salidas
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
#
# ##


    def __init__(self):
        super().__init__()
        # 32x32x3xbatch_size
        self.fc1 = nn.Sequential(
            nn.Linear(3072, 1000),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1000, 200),
            nn.ReLU()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(200, 30),
            nn.ReLU()
        )

        self.fc4 = nn.Linear(30, 10)

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)
        return self.fc4(self.fc3(self.fc2(self.fc1(x))))


network = NeuronalNetwork()
