# Name: Ahilesh Vadivel
# Date: 4rd April 2026
# Project 5 - Recognition using Deep Networks
# Task 2: Examine the network — Task 2A: Analyze first layer filters

#Imports
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2


#Network Definition (must match task1_mnist.py exactly)
class MyNetwork(nn.Module):
    """
    Same CNN architecture as task1_mnist.py.
    Must be defined here to load the saved weights correctly.
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
    print(f"Model loaded from '{model_path}'\n")
    return model


#Print Model Structure
def print_model_structure(model):
    """
    Prints the full model structure showing each layer and its name.
    The layer names (conv1, conv2, fc1, fc2 etc.) are used to
    access weights directly via model.<layer_name>.weight
    """
    print("── Model Structure ──────────────────────────────────")
    print(model)
    print()


#Analyze First Layer Filters
def analyze_first_layer(model):
    """
    Retrieves the weights of conv1 (the first convolutional layer).
    Shape: [10, 1, 5, 5] — 10 filters, 1 input channel, 5x5 each.
    Prints the shape and the raw weight values for each filter.
    Returns the weights tensor for visualization.
    """
    # Access weights using the layer name shown in the model printout
    weights = model.conv1.weight

    print("── conv1 Filter Weights ─────────────────────────────")
    print(f"Shape: {list(weights.shape)}  "
          f"→  [num_filters, in_channels, height, width]\n")

    # Detach from computation graph before printing/numpy conversion
    with torch.no_grad():
        for i in range(weights.shape[0]):
            filter_i = weights[i, 0]        # shape: (5, 5)
            print(f"Filter {i}:")
            print(filter_i.detach().numpy().round(4))
            print()

    return weights


#Visualize Filters
def visualize_filters(weights):
    """
    Plots all 10 conv1 filters in a 3×4 grid (last 2 cells left blank).
    Each filter is a 5×5 matrix of learned weight values displayed as
    a heatmap. Saves the figure as 'conv1_filters.png'.
    """
    fig = plt.figure(figsize=(9, 7))
    fig.suptitle("conv1 Filter Weights (10 filters, 5×5)", fontsize=13)

    with torch.no_grad():
        for i in range(10):
            ax = fig.add_subplot(3, 4, i + 1)   # 3 rows, 4 cols, 1-indexed
            filter_img = weights[i, 0].detach().numpy()

            ax.imshow(filter_img, cmap='viridis')
            ax.set_title(f"Filter {i}", fontsize=10)
            ax.set_xticks([])                    # cleaner plot — no tick marks
            ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("conv1_filters.png", dpi=150)
    plt.show()
    print("Filter plot saved as 'conv1_filters.png'")


#Load First Training Image
def load_first_training_image():
    """
    Loads the MNIST training set and returns the first image as a
    numpy array (28x28, float32) for use with OpenCV's filter2D.
    Also returns the true label of that image.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = torchvision.datasets.MNIST(
        './data', train=True, download=True, transform=transform
    )
    image, label = train_set[0]             # first training example
    # Squeeze channel dim → (28,28) numpy float32 for OpenCV
    img_np = image.squeeze().numpy().astype(np.float32)
    print(f"First training image — label: {label}, shape: {img_np.shape}\n")
    return img_np, label


#Apply Filters with OpenCV
def apply_filters(model, img_np):
    """
    Applies each of the 10 conv1 filters to the image using cv2.filter2D.
    Returns a list of (filter_weights_2d, filtered_image) tuples.
    cv2.filter2D performs a 2D cross-correlation (same as Conv2d with 1 channel).
    """
    results = []

    with torch.no_grad():
        for i in range(10):
            # Extract ith filter as (5,5) numpy array
            kernel = model.conv1.weight[i, 0].detach().numpy()

            # Apply filter — ddepth=-1 keeps same depth as source image
            filtered = cv2.filter2D(img_np, ddepth=-1, kernel=kernel)
            results.append((kernel, filtered))
            print(f"Filter {i} applied — "
                  f"output range: [{filtered.min():.3f}, {filtered.max():.3f}]")

    return results


#Visualize Filter Effects
def visualize_filter_effects(results, label):
    """
    Plots a 4x5 grid showing each filter's weights alongside its filtered
    output when applied to the first training image.
    Columns alternate: [filter weights | filtered image] for all 10 filters.
    Saves the figure as 'filter_effects.png'.
    """
    fig = plt.figure(figsize=(10, 9))
    fig.suptitle(f"conv1 Filter Effects on First Training Image (label: {label})",
                 fontsize=12)

    for i, (kernel, filtered) in enumerate(results):
        # Each filter occupies two columns: weights (left) and output (right)
        # Row = i // 5,  pair position within row = i % 5
        # In a 4-row x 5-col grid, filter i sits at:
        #   weights  → subplot index (row*10) + (col*2) + 1
        #   filtered → subplot index (row*10) + (col*2) + 2
        row = i // 5
        col = i % 5
        idx_weights  = row * 10 + col * 2 + 1
        idx_filtered = row * 10 + col * 2 + 2

        # Filter weights heatmap
        ax1 = fig.add_subplot(4, 5, idx_weights)
        ax1.imshow(kernel, cmap='gray')
        ax1.set_title(f"F{i} weights", fontsize=8)
        ax1.set_xticks([])
        ax1.set_yticks([])

        # Filtered output
        ax2 = fig.add_subplot(4, 5, idx_filtered)
        ax2.imshow(filtered, cmap='gray')
        ax2.set_title(f"F{i} output", fontsize=8)
        ax2.set_xticks([])
        ax2.set_yticks([])

    plt.tight_layout()
    plt.savefig("filter_effects.png", dpi=150)
    plt.show()
    print("\nFilter effects plot saved as 'filter_effects.png'")


def main(argv):
    """
    Entry point for Task 2.
    Task 2A: loads model, prints structure, visualizes conv1 filter weights.
    Task 2B: applies each filter to the first training image using cv2.filter2D
             and plots the filter weights alongside the filtered outputs.
    """
    model = load_model("mnist_model.pth")

    # Task 2A — print structure and visualize filters
    print_model_structure(model)
    weights = analyze_first_layer(model)
    visualize_filters(weights)

    # Task 2B — apply filters to first training image and visualize effects
    img_np, label = load_first_training_image()
    results       = apply_filters(model, img_np)
    visualize_filter_effects(results, label)


if __name__ == "__main__":
    main(sys.argv)