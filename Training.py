import os
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.utils.data import random_split
import torch.optim as optim
from sklearn.metrics import f1_score, recall_score

import torch 
import torch.nn as nn


#### DATA PROCESSING ####

# Define the base directory
base_dir = "C:\\Users\\akash\\Downloads\\Documents\\School\\McMaster\\Fourth Year\\COMPSCI 4AL3\\Final Project\\ChestXRay2017\\chest_xray"

# Define the transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.ToTensor(),          # Convert image to tensor (0-1 range)
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize with ImageNet mean/std
])

# Load datasets
train_dataset = datasets.ImageFolder(os.path.join(base_dir, "train"), transform=transform)
test_dataset = datasets.ImageFolder(os.path.join(base_dir, "test"), transform=transform)

# Perform the random split
train_dataset, val_dataset = random_split(train_dataset, [5000, 232])

print(len(train_dataset), len(val_dataset), len(test_dataset))

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


################################### MODEL #######################################


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


def ResNet50(img_channel, num_classes):
    return ResNet(ResidualBlock, [3, 4, 6, 3], img_channel, num_classes)


# Initialize the model

# For the person marking the training portion of the model, I recommend using a machine with a GPU as the training process does take a while  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet50(img_channel=3, num_classes=2).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

batch_size = 32
epochs = 5

train_losses = []
val_losses = []

################################## TRAINING AND VALIDAATION #######################################################

def train_one_epoch():
    model.train()  # Set model to training mode
    running_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Reset the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

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

    train_losses.append(avg_loss_epoch)
    print(f"Training Loss: {avg_loss_epoch:.3f}, Accuracy: {avg_acc_epoch:.1f}%\n")


def validate_one_epoch():
    model.train(False)  # Set model to evaluation mode
    running_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

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

    val_losses.append(avg_loss_epoch)
    print(f"Validation Loss: {avg_loss_epoch:.3f}, Accuracy: {avg_acc_epoch:.1f}%\n")
    print('******************************************************')



def TrainAndValidate():
    print("Training and Validation Started")
    for epoch in range(epochs):
        print("Epoch", epoch + 1)
        train_one_epoch()
        validate_one_epoch()

    print("Finished training")

    # Plotting Training Loss and Validation loss for each epoch
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label="Training Loss")
    plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss per Epoch")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # Code for training and validation
    TrainAndValidate()
    torch.save(model.state_dict(), "trained_resnet50.pth")
    print("Model saved as trained_resnet50.pth")






        

