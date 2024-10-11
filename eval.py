import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class Evaluator:
    def __init__(self, model, val_loader, device='cuda'):
        self.model = model.to(device)
        self.val_loader = val_loader
        self.device = device

    def evaluate(self):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)
        self.plot_confusion_matrix(cm)

    def plot_confusion_matrix(self, cm):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.show()
