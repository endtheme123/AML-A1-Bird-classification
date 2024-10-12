import torch.optim as optim
import torch
import torch.nn as nn
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
class Trainer:
    def __init__(self, model, train_loader, test_loader, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        # self.val_loader = val_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.device = device
        self.output_dir="./checkpoints"
        self.result_dir="./results"
        self.plot_dir = "./plots"
        if(not os.path.exists(self.output_dir)):
            os.mkdir(self.output_dir)
        if(not os.path.exists(self.plot_dir)):
            os.mkdir(self.plot_dir)
        if(not os.path.exists(self.result_dir)):
            os.mkdir(self.result_dir)
        self.writer = SummaryWriter(self.result_dir)
    def train(self, epochs=10):
        training_loss_list = []
        training_acc_list= []
        testing_loss_list= []
        testing_acc_list= []
        for epoch in tqdm(range(epochs)):
            self.model.train()
            running_loss = 0.0
            gt = []
            pred = []

            
            count = 0
            print()
            for (inputs, labels) in self.train_loader:
                
                # print(labels)
                # labels = torch.tensor(labels)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                # print(outputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                # print("predicted: ",predicted)
                # print("labels: ",labels)
                gt.extend(labels)
                pred.extend(predicted)
                count = count+1
                # total += labels.size(0)
                # correct += (predicted == labels).sum().item()
            training_loss = running_loss / count
            training_loss_list.append(training_loss)
            train_accuracy = 100*sum([1 for g, p in zip(gt, pred) if g == p]) / len(gt)
            training_acc_list.append(train_accuracy)

            val_loss, val_acc = self.validate()

            testing_loss_list.append(val_loss)
            testing_acc_list.append(val_acc)



            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss/len(self.train_loader)}, Accuracy: {train_accuracy}%")
            if(epoch+1) % 5 == 0 or epoch == 0:
                self.writer.add_scalar(
                    f"training_loss",
                    training_loss,
                    epochs,
                )
                self.writer.add_scalar(
                    f"training_acc",
                    train_accuracy,
                    epochs,
                )
                self.writer.add_scalar(
                    f"testing_loss",
                    val_loss,
                    epochs,
                )
                self.writer.add_scalar(
                    f"testing_acc",
                    val_acc,
                    epochs,
                )
                
                torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"Epoch_{epoch + 1}.pth"))
            # self.validate()
        print(testing_loss_list)
        print(training_loss_list)
        self.plot_graph(testing_loss_list, training_loss_list, os.path.join(self.plot_dir, 'Loss.png'), "loss")
        self.plot_graph(testing_acc_list, training_acc_list, os.path.join(self.plot_dir, 'Acc.png'), "Accuracy")

    def validate(self):
        self.model.eval()
        valid_loss = 0.0
        count = 0
        gt = []
        pred = []
        with torch.no_grad():
            for (data, target) in self.test_loader:
                data = data.type(torch.FloatTensor).to(self.device)
                target = target.to(self.device)
                output = self.model(data)
                preds = torch.argmax(output, axis=1).tolist()
                labels = target.tolist()
                gt.extend(labels)
                pred.extend(preds)
                loss = self.criterion(output, target)
                count = count+1
                # update-average-validation-loss
                valid_loss += loss.item()
            val_loss = valid_loss/count
            val_accuracy = 100*sum([1 for g, p in zip(gt, pred) if g == p]) / len(gt)
        #     correct = 0
        #     total = 0
        # with torch.no_grad():
        #     for inputs, labels in self.val_loader:
        #         inputs, labels = inputs.to(self.device), labels.to(self.device)
        #         outputs = self.model(inputs)
        #         _, predicted = torch.max(outputs, 1)
        #         total += labels.size(0)
        #         correct += (predicted == labels).sum().item()

        # val_accuracy = 100 * correct / total
        print(f'Validation Accuracy: {val_accuracy}%')
        return val_loss, val_accuracy


    def plot_graph(self, plot_list_val, plot_list_train, fig_path, graph_for):
        from matplotlib import pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.plot(plot_list_train, label=f'train_{graph_for}')
        plt.plot(plot_list_val,label=f'val_{graph_for}')
        plt.title('Train vs Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(fig_path)
