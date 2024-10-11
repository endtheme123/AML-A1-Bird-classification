import torch.optim as optim
import torch
import torch.nn as nn
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
class Trainer:
    def __init__(self, model, train_loader, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        # self.val_loader = val_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.device = device
        self.output_dir="./checkpoints"
        self.result_dir="/results"
        if(not os.path.exists(self.output_dir)):
            os.mkdir(self.output_dir)
        if(not os.path.exists(self.result_dir)):
            os.mkdir(self.result_dir)
        self.writer = SummaryWriter(self.result_dir)
    def train(self, epochs=10):
        for epoch in tqdm(range(epochs)):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for (inputs, labels) in self.train_loader:
                print(labels)
                labels = torch.tensor(labels)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_accuracy = 100 * correct / total
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss/len(self.train_loader)}, Accuracy: {train_accuracy}%")
            if(epoch+1) % 5 == 0 or epoch == 0:
                self.writer.add_scalar(
                    f"bpp/num{epoch}",
                    bpp,
                    self.step // self.save_and_sample_every,
                )
                self.writer.add_scalar(
                    f"psnr/num{i}",
                    batch_psnr(compressed.clamp(0.0, 1.0).to('cpu'), batch[0]),
                    self.step // self.save_and_sample_every,
                )
                self.writer.add_images(
                    f"compressed/num{i}",
                    compressed.clamp(0.0, 1.0),
                    self.step // self.save_and_sample_every,
                )
                self.writer.add_images(
                    f"original/num{i}",
                    batch[0],
                    self.step // self.save_and_sample_every,
                )
                torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"Epoch_{epoch + 1}.pth"))
            # self.validate()

    # def validate(self):
    #     self.model.eval()
    #     correct = 0
    #     total = 0
    #     with torch.no_grad():
    #         for inputs, labels in self.val_loader:
    #             inputs, labels = inputs.to(self.device), labels.to(self.device)
    #             outputs = self.model(inputs)
    #             _, predicted = torch.max(outputs, 1)
    #             total += labels.size(0)
    #             correct += (predicted == labels).sum().item()

    #     val_accuracy = 100 * correct / total
    #     print(f'Validation Accuracy: {val_accuracy}%')
