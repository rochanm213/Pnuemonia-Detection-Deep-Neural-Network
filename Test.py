import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import f1_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt
from Training import ResNet152, test_loader  # Import your model architecture

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model architecture
model = ResNet152(img_ch=1, num_classes=3).to(device)

# Load the saved weights into the model
model.load_state_dict(torch.load("final_trained_resnet152.pth", map_location=device, weights_only=True))

# Define the criterion (same as during training)
criterion = nn.CrossEntropyLoss()


def test():
    model.eval()  # Set the model to evaluation mode

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    all_labels = []
    all_preds = []
    all_probs = []

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

            # Store labels, predictions, and probabilities for metrics
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

# Run the test
test()
