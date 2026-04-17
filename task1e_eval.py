# Name: Ahilesh Vadivel
# Date: 3rd April 2026
# Project 5 - Recognition using Deep Networks
# Task 1E: Load saved model and evaluate on first 10 test examples

#Imports
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


#Network Definition (must match task1_mnist.py exactly)
class MyNetwork(nn.Module):
    """
    Same CNN architecture as task1_mnist.py.
    Must be defined here so we can load the saved state dict.
    """
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1   = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.conv2   = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.dropout = nn.Dropout(p=0.5)
        self.pool    = nn.MaxPool2d(kernel_size=2)
        self.fc1     = nn.Linear(320, 50)
        self.fc2     = nn.Linear(50, 10)

    # Computes a forward pass through the network
    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.dropout(self.conv2(x))))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x


#Load Saved Model
def load_model(model_path="mnist_model.pth"):
    """
    Instantiates MyNetwork, loads saved weights, and sets to eval mode.
    eval() disables dropout so predictions are deterministic.
    """
    model = MyNetwork()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Model loaded from '{model_path}'")
    return model


#Load Test Set
def load_test_data(batch_size=10):
    """
    Loads the MNIST test set with shuffle=False so the first batch
    always contains the same first 10 examples.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', train=False,
                                   download=True, transform=transform),
        batch_size=batch_size,
        shuffle=False           # keep order deterministic
    )
    return test_loader


#Print Predictions
def print_predictions(model, images, labels, n=10):
    """
    Runs the first n images through the model and prints:
      - The 10 output values (log-probabilities) to 2 decimal places
      - The predicted digit (index of max output)
      - The correct label
    Returns the list of predictions for use in plotting.
    """
    with torch.no_grad():
        outputs = model(images[:n])         # shape: (n, 10)

    predictions = []
    print(f"\n{'─'*65}")
    print(f"{'Idx':>4}  {'Output values (log-probs)':^44}  {'Pred':>4}  {'Label':>5}")
    print(f"{'─'*65}")

    for i in range(n):
        vals  = outputs[i]                          # 10 log-probs
        pred  = vals.argmax().item()                # index of max
        label = labels[i].item()                    # ground truth
        match = "✓" if pred == label else "✗"

        # Format each of the 10 output values to 2 decimal places
        vals_str = "  ".join(f"{v.item():5.2f}" for v in vals)
        print(f"[{i}]  {vals_str}   →  {pred}   (label: {label}) {match}")
        predictions.append(pred)

    print(f"{'─'*65}\n")
    return predictions


#Plot 3×3 Grid
def plot_predictions_grid(images, predictions, labels, n=9):
    """
    Plots the first 9 test digits in a 3×3 grid.
    Each subplot title shows the model's prediction.
    Saves the figure as 'test_predictions.png'.
    """
    fig, axes = plt.subplots(3, 3, figsize=(7, 7))
    fig.suptitle("Model Predictions on First 9 Test Digits", fontsize=13)

    for i, ax in enumerate(axes.flat):
        img  = images[i].squeeze().numpy()
        pred = predictions[i]

        ax.imshow(img, cmap='gray')
        ax.set_title(f"Prediction: {pred}", fontsize=11)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("test_predictions.png", dpi=150)
    plt.show()
    print("Plot saved as 'test_predictions.png'")


#Main
def main(argv):
    """
    Entry point for Task 1E.
    Loads the saved model, runs it on the first 10 test examples,
    prints output values + predictions, and plots a 3x3 grid of results.
    """
    # Load model and data
    model       = load_model("mnist_model.pth")
    test_loader = load_test_data(batch_size=10)

    # Grab the first batch (exactly 10 images)
    images, labels = next(iter(test_loader))

    # Print predictions for all 10 examples
    predictions = print_predictions(model, images, labels, n=10)

    # Plot the first 9 in a 3x3 grid
    plot_predictions_grid(images, predictions, labels, n=9)


if __name__ == "__main__":
    main(sys.argv)