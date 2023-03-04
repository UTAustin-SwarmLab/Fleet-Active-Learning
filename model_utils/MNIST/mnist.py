import torch.nn as nn
import torch
from torch.utils.data import Dataset
from PIL import Image

# Classifier Model for the MNIST Datassets
class SmallMNISTClassifier(nn.Module):

    def __init__(self):
        """
        Initializes the model
        """
        super(SmallMNISTClassifier, self).__init__()
        self.EmbeddingLearner = nn.Sequential(
            nn.Conv2d(1,16,3,padding=(1,1)),
            nn.ReLU(True),
            nn.Dropout2d(0.3),
            nn.Conv2d(16,32,3,padding=(1,1)),
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
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(7*7*32, 128),
            nn.ReLU(True),
            nn.Linear(128,10)
        )
        
    def forward(self, input):
        """
        Forward pass of the model
        :param input: Input to the model
        :return: Output of the model
        """
        x = self.EmbeddingLearner(input)
        flat_x = torch.flatten(x,1)
        out = self.fc(flat_x)
        return out

class LargeMNISTClassifier(nn.Module):

    def __init__(self):
        """
        Initializes the model
        """
        super(LargeMNISTClassifier, self).__init__()

        self.EmbeddingLearner = nn.Sequential(
            nn.Conv2d(1,16,3,padding=(1,1)),
            nn.ReLU(True),
            nn.Dropout2d(0.3),
            nn.Conv2d(16,32,3,padding=(1,1)),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
            nn.Conv2d(32, 32, 3, padding=(1, 1)),
            nn.ReLU(True),
            nn.Dropout2d(0.3),
            nn.Conv2d(32, 32, 3, padding=(1, 1)),
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
        
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(3*3*32, 128),
            nn.ReLU(True),
            nn.Linear(128,10)
        )
    
    def forward(self, input):
        """
        Forward pass of the model
        :param input: Input to the model
        :return: Output of the model
        """
        x = self.EmbeddingLearner(input)
        flat_x = torch.flatten(x,1)
        out = self.fc(flat_x)
        return out

