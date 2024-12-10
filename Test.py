import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import f1_score, recall_score
from Training import ResNet50, test_loader  # Import your model architecture

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model architecture
model = ResNet50(img_channel=3, num_classes=2).to(device)

# Load the saved weights into the model
model.load_state_dict(torch.load("trained_resnet50.pth", map_location=device, weights_only=True))

# Define the criterion (same as during training)
criterion = nn.CrossEntropyLoss()

# Assuming you have already defined `test_loader` with the test data
def test():
    model.eval()  # Set the model to evaluation mode

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    all_labels = []
    all_preds = []

    for batch_idx, data in enumerate(test_loader):
        images, labels = data[0].to(device), data[1].to(device)

        with torch.no_grad():  # Disable gradient computation for evaluation
            outputs = model(images)
            loss = criterion(outputs, labels)  # Compute loss for the batch
            running_loss += loss.item()

            # Calculate the number of correct predictions
            predicted_labels = torch.argmax(outputs, dim=1)
            correct_predictions += (predicted_labels == labels).sum().item()

            # Keep track of the total number of samples
            total_samples += labels.size(0)

            # Store labels and predictions for calculating F1-score and recall
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted_labels.cpu().numpy())

    # Calculate overall accuracy and average loss
    avg_loss = running_loss / len(test_loader)
    accuracy = (correct_predictions / total_samples) * 100

    # Calculate F1-score and recall
    f1 = f1_score(all_labels, all_preds, average='weighted')  # Use 'weighted' for multi-class
    recall = recall_score(all_labels, all_preds, average='weighted')  # Use 'weighted' for multi-class

    print(f"Test Loss: {avg_loss:.3f}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Test F1-Score: {f1:.2f}")
    print(f"Test Recall: {recall:.2f}")

# Run the test
test()
