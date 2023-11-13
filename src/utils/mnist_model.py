import torch.nn as nn
import torch


# Classifier Model for the MNIST Datassets
class MNISTClassifier(nn.Module):
    def __init__(self):
        """
        Initializes the model
        """
        super(MNISTClassifier, self).__init__()
        self.EmbeddingLearner = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=(1, 1)),
            nn.ReLU(True),
            nn.Dropout2d(0.3),
            nn.Conv2d(16, 32, 3, padding=(1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
            nn.Conv2d(32, 32, 3, padding=(1, 1)),
            nn.ReLU(True),
            nn.Dropout2d(0.3),
            nn.Conv2d(32, 32, 3, padding=(1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d(2),
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(7 * 7 * 32, 128), nn.ReLU(True)
        )
        self.fc2 = nn.Sequential(nn.Linear(128, 10))
        self.emb_size = 128

    def forward(self, input):
        """
        Forward pass of the model
        :param input: Input to the model
        :return: Output of the model
        """
        x = self.EmbeddingLearner(input)

        flat_x = torch.flatten(x, 1)
        emb = self.fc1(flat_x)
        out = self.fc2(emb)
        return out, emb
