from data.CUBDataset import CUBTrainDataset,CUBTestDataset
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def get_dataloader(data_path, train_transform, test_transform, batch_size, shuffle, compact):
    
    train_dataset = CUBTrainDataset(root=data_path, transform=train_transform, compact = compact)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_dataset = CUBTestDataset(root=data_path, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    return train_loader, test_loader


def calculate_classwise_accuracy(model, dataloader, num_classes, device='cuda'):
    model.to(device)
    # Switch model to evaluation mode
    model.eval()

    # Initialize counters for correct predictions and total samples per class
    class_correct = [0 for _ in range(num_classes)]
    class_total = [0 for _ in range(num_classes)]

    with torch.no_grad():  # Disable gradient calculation
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass: Get model outputs
            outputs = model(inputs)

            # Get predictions by taking the index of the max logit for each sample
            _, preds = torch.max(outputs, 1)

            # Update counts for each class
            for label, pred in zip(labels, preds):
                if label == pred:
                    class_correct[label] += 1
                class_total[label] += 1

    # Calculate accuracy for each class
    class_accuracies = [100.0 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0 for i in range(num_classes)]

    # Calculate mean class accuracy
    mean_class_accuracy = sum(class_accuracies) / num_classes

    return class_accuracies, mean_class_accuracy


def plot_confusion_matrix_sklearn(model, dataloader, class_names, device='cuda'):
    # Set model to evaluation mode
    model.eval()

    # Initialize lists to hold true and predicted labels
    true_labels = []
    pred_labels = []

    # Disable gradient computation (no need for backpropagation in evaluation)
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Get model predictions
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Store the true labels and predicted labels
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())

    # Generate the confusion matrix using scikit-learn
    conf_matrix = confusion_matrix(true_labels, pred_labels)

    # Plot confusion matrix using sklearn's built-in display function
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
    
    # Plot the confusion matrix with a title and color map
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()


