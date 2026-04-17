# Name: Ahilesh Vadivel
# Date: 4rd April 2026
# Project 5 - Recognition using Deep Networks
# Task 1F: Run trained model on custom handwritten digit images

#Imports
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt


#Network Definition (must match task1_mnist.py exactly)
class MyNetwork(nn.Module):
    """
    Same CNN architecture used during training.
    Must match exactly so the saved weights load correctly.
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
    """Loads the trained model weights and sets it to evaluation mode."""
    model = MyNetwork()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Model loaded from '{model_path}'")
    return model


#Load & Preprocess a Single Image
def preprocess_image(image_path):
    """
    Loads a single handwritten digit image and preprocesses it to match
    the MNIST format:
      1. Convert to greyscale
      2. Resize to 28×28
      3. Convert to tensor  (pixel values → [0.0, 1.0])
      4. Invert intensities  (your photo: dark digit on white bg
                              MNIST:      white digit on black bg)
      5. Normalize with MNIST mean (0.1307) and std (0.3081)

    Returns a (1, 1, 28, 28) tensor ready for the network.
    """
    transform = transforms.Compose([
        transforms.Grayscale(),                         # ensure single channel
        transforms.Resize((28, 28)),                    # match MNIST dimensions
        transforms.ToTensor(),                          # → [0.0, 1.0]
        transforms.Lambda(lambda x: 1.0 - x),          # invert: dark→light
        transforms.Normalize((0.1307,), (0.3081,))     # match MNIST stats
    ])

    img = Image.open(image_path)
    return transform(img).unsqueeze(0)   # add batch dimension → (1,1,28,28)


#Load All Digit Images
def load_custom_digits(image_folder="custom_digits"):
    """
    Loads all digit images from the given folder.
    Expects files named 0.png, 1.png, ..., 9.png  (or .jpg).
    Returns a list of (tensor, true_label, filename) tuples.
    """
    digits = []

    for label in range(10):
        # Try both .png and .jpg extensions
        for ext in [".png", ".jpg", ".jpeg"]:
            path = os.path.join(image_folder, f"{label}{ext}")
            if os.path.exists(path):
                tensor = preprocess_image(path)
                digits.append((tensor, label, f"{label}{ext}"))
                break
        else:
            print(f"  Warning: no image found for digit {label} in '{image_folder}'")

    print(f"\nLoaded {len(digits)} custom digit images from '{image_folder}/'")
    return digits


#Run Network on Custom Digits
def predict_custom_digits(model, digits):
    """
    Runs each image through the network.
    Prints the output values, prediction, and true label for each digit.
    Returns a list of (image_tensor, true_label, prediction) tuples.
    """
    results = []

    print(f"\n{'─'*68}")
    print(f"{'File':<10}  {'Output values (log-probs)':^44}  {'Pred':>4}  {'Label':>5}")
    print(f"{'─'*68}")

    with torch.no_grad():
        for tensor, label, fname in digits:
            output = model(tensor)                  # forward pass
            pred   = output.argmax(dim=1).item()    # predicted class
            match  = "✓" if pred == label else "✗"

            vals_str = "  ".join(f"{v.item():5.2f}" for v in output[0])
            print(f"{fname:<10}  {vals_str}   →  {pred}   (label: {label}) {match}")

            results.append((tensor.squeeze(), label, pred))

    print(f"{'─'*68}\n")
    return results


#Plot Results Grid
def plot_custom_results(results):
    """
    Plots all custom digit images in a 2×5 grid.
    Each title shows the predicted label; green = correct, red = wrong.
    Saves the figure as 'custom_digit_results.png'.
    """
    n    = len(results)
    cols = 5
    rows = (n + cols - 1) // cols          # ceiling division

    fig, axes = plt.subplots(rows, cols, figsize=(11, 5 * rows // 2))
    fig.suptitle("Network Predictions on Custom Handwritten Digits", fontsize=13)

    for i, ax in enumerate(axes.flat):
        if i < n:
            img, label, pred = results[i]

            # Undo normalization for display: x * std + mean → [0,1]
            display = img * 0.3081 + 0.1307
            ax.imshow(display.numpy(), cmap='gray')

            # Green title if correct, red if wrong
            color = 'green' if pred == label else 'red'
            ax.set_title(f"Pred: {pred}  (true: {label})",
                         fontsize=10, color=color)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("custom_digit_results.png", dpi=150)
    plt.show()
    print("Plot saved as 'custom_digit_results.png'")


#Main
def main(argv):
    """
    Entry point for Task 1F.
    Loads the trained model, reads custom handwritten digit images,
    preprocesses them to match MNIST format, runs them through the
    network, and displays the results.
    """
    model       = load_model("mnist_model.pth")
    digits      = load_custom_digits("custom_digits")

    if not digits:
        print("No images found. Please add digit images to the 'custom_digits/' folder.")
        return

    results     = predict_custom_digits(model, digits)
    plot_custom_results(results)


if __name__ == "__main__":
    main(sys.argv)