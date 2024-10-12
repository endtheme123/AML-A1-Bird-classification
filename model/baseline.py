import torch.nn as nn
import torchvision.models as models

class BirdClassifier(nn.Module):
    def __init__(self, num_classes=200):
        super(BirdClassifier, self).__init__()
        # Use a pre-trained ResNet and fine-tune it
        self.model = models.resnet50(pretrained=True)
        num_features = self.model.fc.in_features
        # Replace the last fully connected layer
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return nn.Softmax(dim = 1)(self.model(x))
