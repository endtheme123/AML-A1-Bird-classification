import torch.nn as nn
import torchvision.models as models

class BirdClassifier(nn.Module):
    def __init__(self, num_classes=200, freeze = False):
        super(BirdClassifier, self).__init__()
        # Use a pre-trained ResNet and fine-tune it
        self.model = models.resnet50(pretrained=True)
        
        num_features = self.model.fc.in_features
        # Replace the last fully connected layer
        self.model.fc = nn.Sequential(
            nn.Linear(in_features=num_features, out_features=2048,bias=True),
            nn.SiLU(),
            nn.Dropout(),
            nn.Linear(in_features=2048, out_features=num_classes,bias=True)
        )

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.fc.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.model(x)



class EfficientBirdClassifier(nn.Module):
    def __init__(self, num_classes=200, freeze = False):
        super(EfficientBirdClassifier, self).__init__()
        # Use a pre-trained ResNet and fine-tune it
        self.model = models.efficientnet_b4(pretrained=True)
        
        classifier = nn.Sequential(
            nn.Linear(in_features=self.model.classifier[1].in_features, out_features=2048,bias=True),
            nn.Dropout(),
            nn.Linear(in_features=2048, out_features=200,bias=True)
        )
        self.model.classifier  = classifier

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.classifier.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.model(x)
