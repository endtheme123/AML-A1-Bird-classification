import torch.optim as optim
import torch.nn.functional as F
import torch
import torch.nn as nn
import os
from tqdm import tqdm
from grokfast import *
from torch.utils.tensorboard import SummaryWriter
class SelfDistillTrainer:
    def __init__(self, model, train_loader, test_loader, experiment_name, tempurature = 1.5, device='cuda', grokking = False):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        # self.val_loader = val_loader
        self.tempurature = tempurature
        self.grokking = grokking 
        self.criterion = nn.CrossEntropyLoss()
        self.kld = torch.nn.KLDivLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.device = device
        self.experiment_name = experiment_name
        self.output_dir=os.path.join("./checkpoints", experiment_name)
        self.result_dir=os.path.join("./results" , experiment_name)
        self.plot_dir = os.path.join("./plots" , experiment_name)
        if(not os.path.exists("./plots")):
            os.mkdir("./plots")
        if(not os.path.exists("./checkpoints")):
            os.mkdir("./checkpoints")
        if(not os.path.exists("./results")):
            os.mkdir("./results")
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
        grads = None
        for epoch in tqdm(range(epochs)):
            self.model.train()
            running_loss = 0.0
            running_corrects = 0

            print()
            for (inputs, labels) in self.train_loader:
                
                # print(labels)
                # labels = torch.tensor(labels)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    # randomly sample a sub network
                    self.model.adjust_sur_prop(0.7)
                    # calculate based on equation 3
                    student_outputs = self.model(inputs)
                    soft_student_outputs = F.log_softmax(student_outputs/self.tempurature, dim=1)
                    # calculate based on equation 4
                    self.model.adjust_sur_prop(1.0)
                    outputs = self.model(inputs)
                    soft_ouputs = F.softmax(outputs/self.tempurature, dim=1)
                    loss_kld = self.kld(soft_student_outputs, soft_ouputs)
                    # print(outputs)
                    teacher_loss_ce = self.criterion(outputs, labels)
                    student_loss_ce = self.criterion(student_outputs, labels)

                    loss = teacher_loss_ce + student_loss_ce + 0.1*(self.tempurature**2)*loss_kld

                    running_loss += loss.item()
                loss.backward()
                if(grokking):
                    grads = gradfilter_ema(self.model, grads=grads, alpha=0.98, lamb=2.0)
                self.optimizer.step()

                
                _, predicted = torch.max(outputs, 1)

                running_corrects += sum(1 for a, b in zip(predicted, labels) if a == b)
                print("output: ",outputs)
                print("predicted: ",predicted)
                print("labels: ",labels)
                # total += labels.size(0)
                # correct += (predicted == labels).sum().item()
            training_loss = running_loss/len(self.train_loader.dataset)
            training_loss_list.append(training_loss)
            
            train_accuracy = running_corrects/len(self.train_loader.dataset)
            training_acc_list.append(train_accuracy)

            val_loss, val_acc = self.validate()

            testing_loss_list.append(val_loss)
            testing_acc_list.append(val_acc)



            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss/len(self.train_loader)}, Accuracy: {train_accuracy}%")
            if(epoch+1) % 5 == 0 or epoch == 0:
                self.writer.add_scalar(
                    f"training_loss/num{epoch}",
                    training_loss,
                    epochs,
                )
                self.writer.add_scalar(
                    f"training_acc/num{epoch}",
                    train_accuracy,
                    epochs,
                )
                self.writer.add_scalar(
                    f"testing_loss/num{epoch}",
                    val_loss,
                    epochs,
                )
                self.writer.add_scalar(
                    f"testing_acc/num{epoch}",
                    val_acc,
                    epochs,
                )
                
                torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"Epoch_{epoch + 1}.pth"))
            # self.validate()
        # print(testing_loss_list)
        # print(training_loss_list)
        self.plot_graph(testing_loss_list, training_loss_list, os.path.join(self.plot_dir, 'Loss.png'), "loss")
        self.plot_graph(testing_acc_list, training_acc_list, os.path.join(self.plot_dir, 'Acc.png'), "Accuracy")

    def validate(self):
        self.model.eval()
        valid_loss = 0.0
        valid_corrects = 0
        pred = []
        with torch.no_grad():
            for (data, target) in self.test_loader:
                data = data.type(torch.FloatTensor).to(self.device)
                target = target.to(self.device)
                output = self.model(data)
                # prob = nn.Softmax(dim = 1)(output)
                preds = torch.argmax(output, axis=1).tolist()
                labels = target.tolist()
                loss = self.criterion(output, target)
                # count = count+1
                # update-average-validation-loss
                valid_loss += loss.item()
                valid_corrects += sum(1 for a, b in zip(preds, labels) if a == b)
            val_loss = valid_loss/len(self.test_loader.dataset)
            val_accuracy = valid_corrects/len(self.test_loader.dataset)
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