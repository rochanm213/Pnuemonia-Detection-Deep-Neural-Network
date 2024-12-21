import os
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.utils.data import random_split
import torch.optim as optim
from sklearn.metrics import f1_score, recall_score, roc_curve, auc
from torch.amp import autocast, GradScaler

import torch 
import torch.nn as nn


################################ DATA PROCESSING ###############################

# Define the base directory
base_dir = "C:\\Users\\akash\\Downloads\\Documents\\School\\McMaster\\Fourth Year\\COMPSCI 4AL3\\Final Project\\ChestXRay2017\\chest_xray"

# Define the transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.Grayscale(),         # Convert to grayscale 
    transforms.ToTensor(),          # Convert image to tensor (0-1 range)
    transforms.Normalize([0.5], [0.5])  # Normalize for grayscale (adjust if dataset stats are available)
    
    # transforms.Resize((224, 224)),  # Resize to 224x224
    # transforms.ToTensor(),          # Convert image to tensor (0-1 range)
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Normalize with ImageNet mean/std
])

def sort_images(source_folder, setName):
    # Determine the base directory of the source folder
    base_directory = os.path.join(source_folder, setName)
    source_folder = os.path.join(base_directory, "PNEUMONIA")

    # Update folder paths to be in the same location as the source folder
    folder1 = os.path.join(base_directory, "VIRUS")
    folder2 = os.path.join(base_directory, "BACTERIA")

    # Ensure destination folders exist
    os.makedirs(folder1, exist_ok=True)
    os.makedirs(folder2, exist_ok=True)

    # check if already sorted
    if not os.path.isdir(source_folder):
        print("Images already sorted")
        return

    # Iterate through files in the source folder
    for filename in os.listdir(source_folder):
        source_path = os.path.join(source_folder, filename)
        # Check if it's a file
        if os.path.isfile(source_path):
            # Check if the filename contains any of the keywords
            if any(keyword.lower() in filename.lower() for keyword in ["virus"]):
                destination_path = os.path.join(folder1, filename)
            else:
                destination_path = os.path.join(folder2, filename)

            # Move file using os.rename
            os.rename(source_path, destination_path)
    os.rmdir(source_folder)

   

# Load datasets
train_dataset = datasets.ImageFolder(os.path.join(base_dir, "train"), transform=transform)
test_dataset = datasets.ImageFolder(os.path.join(base_dir, "test"), transform=transform)

# Perform the random split
train_dataset, val_dataset = random_split(train_dataset, [4160, 1072])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# print(len(train_loader), len(val_loader), len(test_loader))

# Classes
# class_names = train_dataset.classes
# print(f"Classes: {class_names}")


# def show_images(dataloader, class_names):
#     # Get a batch of images and labels
#     images, labels = next(iter(dataloader))

#     # Create a grid of images
#     grid = make_grid(images, nrow=8, normalize=True)  # Adjust nrow for layout
#     np_grid = grid.numpy().transpose((1, 2, 0))  # Convert to HWC format for Matplotlib

#     # Plot the grid
#     plt.figure(figsize=(12, 6))
#     plt.imshow(np.clip(np_grid, 0, 1))  # Clip values to valid range
#     plt.axis('off')
#     plt.title("Sample Images")
#     plt.show()

#     # Display corresponding labels
#     print("Labels:", [class_names[label] for label in labels])

# # Visualize a batch from the train_loader
# show_images(train_loader, class_names)


###################################### MODEL ##########################################


# Before starting with the network, we need to build a ResidualBlock that we can re-use through out the network. The block contains a skip connection that allows us to bypass one or more layers in the network

# So instead of learning the full mapping F(x), it learns the residual mapping, which is the difference between the desired output of the block and the input. 

# This simplfies the overall learning of the model as we only need to adjust the residual to bring the output closer to the target

# This architecture is designed like this to solve two primary issues: vanishing gradients and degradation problem. 


class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, identity_downsample=None, stride=1
    ):
        super().__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:  # Ensure the identity downsample is checked correctly
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = self.create_layer(
            block, layers[0], out_channels=64, stride=1
        )
        self.layer2 = self.create_layer(
            block, layers[1], out_channels=128, stride=2
        )
        self.layer3 = self.create_layer(
            block, layers[2], out_channels=256, stride=2
        )
        self.layer4 = self.create_layer(
            block, layers[3], out_channels=512, stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def create_layer(self, block, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []

        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead
        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * 4),
            )

        layers.append(
            block(self.in_channels, out_channels, identity_downsample, stride)
        )

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = out_channels * 4

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)


def ResNet50(img_ch, num_classes):
    return ResNet(ResidualBlock, [3, 4, 6, 3], img_ch, num_classes)

def ResNet101(img_ch, num_classes):
    return ResNet(ResidualBlock, [3, 4, 23, 3], img_ch, num_classes)

def ResNet152(img_ch, num_classes):
    return ResNet(ResidualBlock, [3, 8, 36, 3], img_ch, num_classes)



################################ TRAINING #############################################

# For the person marking the training portion of the model, I recommend using a machine with a GPU as the training process does take a while 


# Early Stopper Class

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    # Notice that this implementation is a little different as it compares its previous validation losses rather than look at the difference between training and validation. This is because for this model the training has no impact on the validation and thus there is no real reason to look at it in this case. 

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class TrainModel():

    def __init__(self, model, learning_rate, batch_size, epochs, l2_lambda, early_stopping=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_lambda) # Weight decay is used for L2 Regularization
        self.criterion = nn.CrossEntropyLoss().to('cuda')  
        self.batch_size= batch_size
        self.epochs = epochs
        self.train_losses = []
        self.val_losses = []
        self.early_stopping = early_stopping
        self.early_stopper = EarlyStopper(patience=1, min_delta=0.01)
        self.overfitting_epoch = 0
        self.scaler = GradScaler()

            
    def train_one_epoch(self):
        self.model.train()  # Set model to training mode
        running_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Reset the gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backpropagation and optimization
            loss.backward()
            self.optimizer.step()

            # Compute accuracy
            _, predicted = outputs.max(1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            # Accumulate loss
            running_loss += loss.item()
            # Log progress
            if batch_idx % 10 == 0:  # Adjust logging frequency
                print(f"Batch {batch_idx}: Loss = {loss.item():.4f}")

        # Calculate average loss and accuracy for the epoch
        avg_loss_epoch = running_loss / len(train_loader)
        avg_acc_epoch = (total_correct / total_samples) * 100

        self.train_losses.append(avg_loss_epoch)
        print(f"Training Loss: {avg_loss_epoch:.3f}, Accuracy: {avg_acc_epoch:.1f}%\n")


    def validate_one_epoch(self):
        self.model.eval()  # Set model to evaluation mode
        running_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                # Forward pass
                with autocast(device_type='cuda'):  # Automatically uses 16-bit precision
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                # Compute accuracy
                _, predicted = outputs.max(1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

                # Accumulate loss
                running_loss += loss.item()

        # Calculate average loss and accuracy for the epoch
        print(len(val_loader))
        avg_loss_epoch = running_loss / len(val_loader)
        avg_acc_epoch = (total_correct / total_samples) * 100

        self.val_losses.append(avg_loss_epoch)
        print(f"Validation Loss: {avg_loss_epoch:.3f}, Accuracy: {avg_acc_epoch:.1f}%\n")
        print('******************************************************')

        return avg_loss_epoch

    
    def TrainAndValidate(self):
        print("Training and Validation Started")
        for epoch in range(self.epochs):
            print("Epoch", epoch + 1)
            self.train_one_epoch()
            validation_loss = self.validate_one_epoch()
            print(len(self.train_losses), len(self.val_losses))
            if self.early_stopping and self.early_stopper.early_stop(validation_loss):
                self.overfitting_epoch = epoch + 1  # Store the epoch (1-indexed)
                print("Early Stop")
                break

        print("Finished training")
        return [self.train_losses, self.val_losses]


    def test(self,):
        self.model.eval()  # Set the model to evaluation mode

        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        all_labels = []
        all_preds = []
        all_probs = []

        for batch_idx, data in enumerate(test_loader):
            images, labels = data[0].to(self.device), data[1].to(self.device)

            with torch.no_grad():  # Disable gradient computation for evaluation
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)  # Compute loss for the batch
                running_loss += loss.item()

                # Calculate the number of correct predictions
                predicted_labels = torch.argmax(outputs, dim=1)
                correct_predictions += (predicted_labels == labels).sum().item()

                # Keep track of the total number of samples
                total_samples += labels.size(0)

                # Store labels, predictions,s and probabilities for metrics
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted_labels.cpu().numpy())
                all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())  # Probabilities for all classes

        # Calculate overall accuracy and average loss
        avg_loss = running_loss / len(test_loader)
        accuracy = (correct_predictions / total_samples) * 100

        # Calculate F1-score and recall
        f1 = f1_score(all_labels, all_preds, average='weighted')  # Use 'weighted' for multi-class
        recall = recall_score(all_labels, all_preds, average='weighted')  # Use 'weighted' for multi-class

        # Compute ROC curve and AUC for each class
        all_labels_one_hot = torch.nn.functional.one_hot(torch.tensor(all_labels), num_classes=3).numpy()  # One-hot encoding
        fpr = {}
        tpr = {}
        roc_auc = {}

        for i in range(3):  # Loop through each class
            fpr[i], tpr[i], _ = roc_curve(all_labels_one_hot[:, i], [prob[i] for prob in all_probs])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot ROC curve for each class
        plt.figure()
        colors = ['darkorange', 'blue', 'green']
        for i in range(3):
            plt.plot(fpr[i], tpr[i], color=colors[i], lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid()
        plt.show()

        print(f"Test Loss: {avg_loss:.3f}")
        print(f"Test Accuracy: {accuracy:.2f}%")
        print(f"Test F1-Score: {f1:.2f}")
        print(f"Test Recall: {recall:.2f}")



################################## EXPERIMENTS ##########################################


# Experiment 1: Tuning model for optimal learning rate. The optimal learning that was determined by the experiment was 1e-4 for Multiclassification which is was different for the Binary Model with RGB channels that had a leearning rate of 1e-3. 

def experiment1():
    epochs = 5

    model1 = TrainModel(ResNet50(img_ch=1, num_classes=3), learning_rate=1e-3, batch_size=32, epochs=epochs, early_stopping=False)
    model2 = TrainModel(ResNet50(img_ch=1, num_classes=3), learning_rate=1e-4, batch_size=32, epochs=epochs, early_stopping=False)
    model3 = TrainModel(ResNet50(img_ch=1, num_classes=3), learning_rate=1e-5, batch_size=32, epochs=epochs, early_stopping=False)

    train_losses_1, val_losses_1 = model1.TrainAndValidate()
    train_losses_2, val_losses_2 = model2.TrainAndValidate()
    train_losses_3, val_losses_3 = model3.TrainAndValidate()
    
    # Plotting Training Loss and Validation loss for each epoch
    plt.figure(figsize=(10, 5))

    plt.plot(range(1, epochs + 1), train_losses_1, label="Training Loss, LR - 0.001", linestyle='-', color='blue', marker='o')
    plt.plot(range(1, epochs + 1), val_losses_1, label="Validation Loss, LR - 0.001", linestyle='--', color='blue', marker='o')

    plt.plot(range(1, epochs + 1), train_losses_2, label="Training Loss, LR - 0.0001", linestyle='-', color='green', marker='o')
    plt.plot(range(1, epochs + 1), val_losses_2, label="Validation Loss, LR - 0.0001", linestyle='--', color='green', marker='o')

    plt.plot(range(1, epochs + 1), train_losses_3, label="Training Loss, LR - 0.00001", linestyle='-', color='red', marker='o')
    plt.plot(range(1, epochs + 1), val_losses_3, label="Validation Loss, LR - 0.00001", linestyle='--', color='red', marker='o')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss per Epoch")
    plt.legend()
    plt.grid()
    plt.show()

    model1.test()
    model2.test()
    model3.test()

# Experiment 2 : Trying different Model Architectures. We noticed that one average the deeper models 101 and 152 performed better however they took much longer and are also more prone to overfitting

def experiment2():
    print("Experiment 2 Started")
    epochs = 5

    model1 = TrainModel(ResNet50(img_ch=1, num_classes=3), learning_rate=1e-4, batch_size=32, epochs=epochs, early_stopping=False)
    model2 = TrainModel(ResNet101(img_ch=1, num_classes=3), learning_rate=1e-4, batch_size=32, epochs=epochs, early_stopping=False)
    model3 = TrainModel(ResNet152(img_ch=1, num_classes=3), learning_rate=1e-4, batch_size=32, epochs=epochs, early_stopping=False)

    train_losses_1, val_losses_1 = model1.TrainAndValidate()
    train_losses_2, val_losses_2 = model2.TrainAndValidate()
    train_losses_3, val_losses_3 = model3.TrainAndValidate()
    
    # Plotting Training Loss and Validation loss for each epoch
    plt.figure(figsize=(10, 5))

    plt.plot(range(1, epochs + 1), train_losses_1, label="Training Loss, ResNet50", linestyle='-', color='blue', marker='o')
    plt.plot(range(1, epochs + 1), val_losses_1, label="Validation Loss, ResNet50", linestyle='--', color='blue', marker='o')

    plt.plot(range(1, epochs + 1), train_losses_2, label="Training Loss, ResNet101", linestyle='-', color='green', marker='o')
    plt.plot(range(1, epochs + 1), val_losses_2, label="Validation Loss, ResNet101", linestyle='--', color='green', marker='o')

    plt.plot(range(1, epochs + 1), train_losses_3, label="Training Loss, ResNet152", linestyle='-', color='red', marker='o')
    plt.plot(range(1, epochs + 1), val_losses_3, label="Validation Loss, ResNet152        ", linestyle='--', color='red', marker='o')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss per Epoch")
    plt.legend()
    plt.grid()
    plt.show()



    model1.test()
    model2.test()
    model3.test()

# Experiment 3: Regulurization with the deeper models, in general ResNet152 perfomed better

def experiment3():
    print("Experiment 3 Started")
    epochs = 5

    model1 = TrainModel(ResNet101(img_ch=1, num_classes=3), learning_rate=1e-4, batch_size=32, l2_lambda=0.01, epochs=epochs, early_stopping=False)
    model2 = TrainModel(ResNet152(img_ch=1, num_classes=3), learning_rate=1e-4, batch_size=32, l2_lambda=0.01, epochs=epochs, early_stopping=False)


    train_losses_1, val_losses_1 = model1.TrainAndValidate()
    train_losses_2, val_losses_2 = model2.TrainAndValidate()
    
    # Plotting Training Loss and Validation loss for each epoch
    plt.figure(figsize=(10, 5))

    plt.plot(range(1, epochs + 1), train_losses_1, label="Training Loss, ResNet101 with Reg", linestyle='-', color='green', marker='o')
    plt.plot(range(1, epochs + 1), val_losses_1, label="Validation Loss, ResNet101 with Reg", linestyle='--', color='green', marker='o')

    plt.plot(range(1, epochs + 1), train_losses_2, label="Training Loss, ResNet152 with Reg", linestyle='-', color='red', marker='o')
    plt.plot(range(1, epochs + 1), val_losses_2, label="Validation Loss, ResNet152 with Reg", linestyle='--', color='red', marker='o')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss per Epoch")
    plt.legend()
    plt.grid()
    plt.show()


    model1.test()
    model2.test()

# Final Experiment: Determining optimal Regularization value w/ Early Stopping. For this experiment we did each model seperately because training this model takes a lot of time

def experiment4():
    print("Experiment 4 Started")
    epochs = 5

    # model1 = TrainModel(ResNet152(img_ch=1, num_classes=3), learning_rate=1e-4, batch_size=32, l2_lambda=0.01, epochs=epochs, early_stopping=True)
    # model2 = TrainModel(ResNet152(img_ch=1, num_classes=3), learning_rate=1e-4, batch_size=32, l2_lambda=0.1e-3, epochs=epochs, early_stopping=True)
    model3 = TrainModel(ResNet152(img_ch=1, num_classes=3), learning_rate=1e-4, batch_size=32, l2_lambda=0.1e-4, epochs=epochs, early_stopping=True)


    # train_losses_1, val_losses_1 = model1.TrainAndValidate()
    # train_losses_2, val_losses_2 = model2.TrainAndValidate()
    train_losses_3, val_losses_3 = model3.TrainAndValidate() 
    
    # Plotting Training Loss and Validation loss for each epoch
    plt.figure(figsize=(10, 5))

    # plt.plot(range(1, len(train_losses_1) + 1), train_losses_1, label="Training Loss, ResNet 152 with Reg = 0.01", linestyle='-', color='green', marker='o')
    # plt.plot(range(1, len(val_losses_1) + 1), val_losses_1, label="Validation Loss, ResNet 152 with Reg = 0.01", linestyle='--', color='green', marker='o')

    # plt.plot(range(1, len(train_losses_2) + 1), train_losses_2, label="Training Loss, ResNet152 with Reg = 1e-3", linestyle='-', color='red', marker='o')
    # plt.plot(range(1, len(val_losses_2) + 1), val_losses_2, label="Validation Loss, ResNet152 with Reg = 1e-3", linestyle='--', color='red', marker='o')

    plt.plot(range(1, len(train_losses_3) + 1), train_losses_3, label="Training Loss, ResNet152 with Reg = 1e-4", linestyle='-', color='red', marker='o')
    plt.plot(range(1, len(val_losses_3) + 1), val_losses_3, label="Validation Loss, ResNet152 with Reg = 1e-4", linestyle='--', color='red', marker='o')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss per Epoch")
    plt.legend()
    plt.grid()
    plt.show()


    # model1.test()
    # model2.test()
    # model3.test() 

# Saving the optimal configuration into a pickle file
def final_model():
    epochs = 5
    final_model = TrainModel(ResNet152(img_ch=1, num_classes=3), learning_rate=1e-4, batch_size=32, l2_lambda=0.1e-4, epochs=epochs, early_stopping=True)
    final_model.TrainAndValidate() 
    torch.save(final_model.model.state_dict(), "final_trained_resnet152.pth")


if __name__ == "__main__":

     # Define source and destination folders
    source_folder = base_dir
    # Sort images
    sort_images(source_folder, "val")
    sort_images(source_folder, "train")
    sort_images(source_folder, "test")

    #Experiments 
    # experiment1() 
    # experiment2()
    # experiment3()
    # experiment4()
    
    final_model()







        

