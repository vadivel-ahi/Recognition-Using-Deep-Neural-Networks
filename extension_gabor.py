# Name: Ahilesh Vadivel
# Date: 6th April 2026
# Project 5 - Recognition using Deep Networks
# Extension: Replace conv1 with a fixed Gabor filter bank and retrain the rest

#Imports
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


#Gabor Filter Generation
def make_gabor_kernel(size, sigma, theta, lambd, gamma, psi=0):
    """
    Creates a single 2D Gabor filter kernel.

    A Gabor filter is a sinusoidal wave modulated by a Gaussian envelope.
    It responds strongly to edges and textures at a specific:
      - orientation (theta)  — angle of the edge it detects
      - frequency  (lambd)   — spatial frequency (stripe width)
      - aspect     (gamma)   — ellipticity of the Gaussian envelope
      - bandwidth  (sigma)   — spread of the Gaussian

    Parameters
    ----------
    size  : int   — kernel size (e.g. 5 for a 5x5 filter)
    sigma : float — standard deviation of the Gaussian envelope
    theta : float — orientation in radians
    lambd : float — wavelength of the sinusoidal factor
    gamma : float — spatial aspect ratio
    psi   : float — phase offset (default 0)

    Returns a (size x size) numpy array.
    """
    half = size // 2
    y, x = np.mgrid[-half:half+1, -half:half+1]

    # Rotate coordinates by theta
    x_rot =  x * np.cos(theta) + y * np.sin(theta)
    y_rot = -x * np.sin(theta) + y * np.cos(theta)

    # Gaussian envelope × sinusoidal carrier
    gaussian = np.exp(-(x_rot**2 + gamma**2 * y_rot**2) / (2 * sigma**2))
    sinusoid = np.cos(2 * np.pi * x_rot / lambd + psi)

    kernel = gaussian * sinusoid
    # Normalize so weights sum to zero (zero-mean filter — important for edges)
    kernel -= kernel.mean()
    return kernel.astype(np.float32)


def build_gabor_bank(n_filters=10, kernel_size=5):
    """
    Builds a bank of n_filters Gabor kernels with varied orientations
    and frequencies to cover diverse edge-detection directions.

    We use:
      - 5 orientations: 0°, 36°, 72°, 108°, 144°  (evenly spaced over 180°)
      - 2 frequencies:  low (λ=4.0) and high (λ=2.5)
      → 10 filters total, matching conv1's original output channels

    Returns a tensor of shape (n_filters, 1, kernel_size, kernel_size).
    """
    filters = []
    orientations = np.linspace(0, np.pi, n_filters // 2, endpoint=False)
    lambdas      = [4.0, 2.5]   # low and high spatial frequency

    for lambd in lambdas:
        for theta in orientations:
            k = make_gabor_kernel(
                size=kernel_size,
                sigma=2.0,
                theta=theta,
                lambd=lambd,
                gamma=0.5
            )
            filters.append(k)

    # Stack into tensor: (n_filters, 1, H, W)
    bank = np.stack(filters, axis=0)[:, np.newaxis, :, :]
    return torch.tensor(bank)


#Gabor Network 
class GaborNet(nn.Module):
    """
    MNIST CNN with the first convolutional layer replaced by a fixed
    Gabor filter bank.

    The first layer (conv1) is initialized with hand-crafted Gabor filters
    and its weights are FROZEN — it will never be updated by backprop.
    All other layers (conv2 onward) are trained normally.

    This tests the hypothesis:
      "Hand-crafted edge detectors can substitute for learned filters
       in the first layer, since early CNN layers typically learn
       Gabor-like filters anyway."
    """
    def __init__(self, n_filters=10, kernel_size=5):
        super(GaborNet, self).__init__()

        #Layer 1: Fixed Gabor filter bank 
        self.conv1 = nn.Conv2d(1, n_filters, kernel_size=kernel_size,
                               bias=False)   # no bias — Gabor filters are zero-mean

        # Initialize conv1 weights with Gabor kernels
        gabor_bank = build_gabor_bank(n_filters, kernel_size)
        with torch.no_grad():
            self.conv1.weight.copy_(gabor_bank)

        # Freeze conv1 — requires_grad=False means backprop won't touch it
        for param in self.conv1.parameters():
            param.requires_grad = False

        #Remaining layers: trainable 
        self.conv2   = nn.Conv2d(n_filters, 20, kernel_size=5)
        self.dropout = nn.Dropout(p=0.5)
        self.pool    = nn.MaxPool2d(2)
        self.fc1     = nn.Linear(320, 50)
        self.fc2     = nn.Linear(50, 10)

    # Computes a forward pass — identical structure to MyNetwork
    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.dropout(self.conv2(x))))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x


#Data Loading
def load_mnist(train_batch_size=64, test_batch_size=1000):
    """Loads the MNIST digit dataset with standard normalization."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', train=True,
                                   download=True, transform=transform),
        batch_size=train_batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', train=False,
                                   download=True, transform=transform),
        batch_size=test_batch_size, shuffle=False
    )
    return train_loader, test_loader


#Training 
def train_epoch(model, train_loader, optimizer, epoch,
                train_losses, train_counter):
    """Runs one epoch of training, recording loss every 10 batches."""
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss   = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            train_losses.append(loss.item())
            train_counter.append(
                batch_idx * len(data) + (epoch-1) * len(train_loader.dataset)
            )
            print(f"  Epoch {epoch}  "
                  f"[{batch_idx * len(data):>6}/{len(train_loader.dataset)}]  "
                  f"Loss: {loss.item():.4f}")


#Evaluation
def evaluate(model, loader, losses, label="Test"):
    """Evaluates accuracy and average loss on the given loader."""
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in loader:
            output      = model(data)
            total_loss += F.nll_loss(output, target, reduction='sum').item()
            correct    += output.argmax(dim=1).eq(target).sum().item()
    avg_loss = total_loss / len(loader.dataset)
    accuracy = 100.0 * correct / len(loader.dataset)
    losses.append(avg_loss)
    print(f"  {label} — Avg loss: {avg_loss:.4f}  "
          f"Accuracy: {correct}/{len(loader.dataset)} ({accuracy:.1f}%)")
    return accuracy


#Visualize Gabor Bank
def visualize_gabor_bank(model):
    """
    Plots the 10 Gabor filters used as conv1 weights.
    Saves figure as 'gabor_filter_bank.png'.
    """
    fig = plt.figure(figsize=(10, 4))
    fig.suptitle("Gabor Filter Bank — conv1 Weights (Fixed)", fontsize=13)

    with torch.no_grad():
        for i in range(10):
            ax = fig.add_subplot(2, 5, i+1)
            kernel = model.conv1.weight[i, 0].numpy()
            ax.imshow(kernel, cmap='gray')
            ax.set_title(f"Filter {i}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("gabor_filter_bank.png", dpi=150)
    plt.show()
    print("Gabor bank saved as 'gabor_filter_bank.png'")


#Compare Filter Outputs
def visualize_gabor_outputs(model, test_loader):
    """
    Applies the Gabor conv1 filters to the first test image and plots
    the filter outputs — mirrors Task 2B for direct comparison.
    Saves figure as 'gabor_filter_outputs.png'.
    """
    import cv2
    images, labels = next(iter(test_loader))
    img_np = images[0].squeeze().numpy().astype(np.float32)

    fig = plt.figure(figsize=(10, 9))
    fig.suptitle(f"Gabor Filter Outputs on Test Image (label: {labels[0].item()})",
                 fontsize=12)

    with torch.no_grad():
        for i in range(10):
            kernel   = model.conv1.weight[i, 0].numpy()
            filtered = cv2.filter2D(img_np, ddepth=-1, kernel=kernel)

            row = i // 5
            col = i % 5
            ax1 = fig.add_subplot(4, 5, row*10 + col*2 + 1)
            ax1.imshow(kernel,   cmap='gray')
            ax1.set_title(f"F{i} weights", fontsize=8)
            ax1.set_xticks([]); ax1.set_yticks([])

            ax2 = fig.add_subplot(4, 5, row*10 + col*2 + 2)
            ax2.imshow(filtered, cmap='gray')
            ax2.set_title(f"F{i} output",  fontsize=8)
            ax2.set_xticks([]); ax2.set_yticks([])

    plt.tight_layout()
    plt.savefig("gabor_filter_outputs.png", dpi=150)
    plt.show()
    print("Gabor outputs saved as 'gabor_filter_outputs.png'")


#Plot Training Curves
def plot_comparison(gabor_train_counter, gabor_train_losses,
                    gabor_test_counter,  gabor_test_losses,
                    orig_accuracies,     gabor_accuracies):
    """
    Two-panel comparison plot:
      Left  — training loss curve for Gabor model
      Right — per-epoch test accuracy: Gabor vs original CNN
    Saves figure as 'gabor_comparison.png'.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Gabor Filter Bank vs Learned Filters — MNIST", fontsize=13)

    # Training loss
    ax1.plot(gabor_train_counter, gabor_train_losses,
             color='blue', linewidth=0.8, label='Train loss (Gabor)')
    ax1.scatter(gabor_test_counter, gabor_test_losses,
                color='red', zorder=5, s=40, label='Test loss (Gabor)')
    ax1.set_xlabel("Training examples seen")
    ax1.set_ylabel("NLL Loss")
    ax1.set_title("Gabor Model — Training Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy comparison
    epochs = range(1, len(gabor_accuracies) + 1)
    ax2.plot(epochs, orig_accuracies,  'g-o', linewidth=1.5,
             label='Original CNN (learned filters)')
    ax2.plot(epochs, gabor_accuracies, 'b-s', linewidth=1.5,
             label='Gabor CNN (fixed filters)')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Test accuracy (%)")
    ax2.set_title("Test Accuracy per Epoch")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("gabor_comparison.png", dpi=150)
    plt.show()
    print("Comparison plot saved as 'gabor_comparison.png'")


#Main
def main(argv):
    """
    Extension entry point.

    Hypothesis:
      Because standard CNNs tend to learn Gabor-like filters in their
      first layer anyway, replacing conv1 with a fixed hand-crafted
      Gabor bank should achieve similar accuracy to a fully-trained CNN,
      while also converging faster (fewer parameters to learn).

    Steps:
      1. Build GaborNet and visualize the filter bank
      2. Train GaborNet for 5 epochs, evaluating after each epoch
      3. Compare against the original CNN accuracy (from task 1)
      4. Visualize how the Gabor filters respond to a test image
    """
    N_EPOCHS = 5
    torch.manual_seed(42)

    train_loader, test_loader = load_mnist()

    #Build and inspect Gabor model 
    model = GaborNet(n_filters=10, kernel_size=5)
    print("── GaborNet Structure ───────────────────────────────")
    print(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"\nTrainable parameters : {trainable:,}")
    print(f"Frozen parameters    : {frozen:,}  ← Gabor conv1 weights\n")

    # Visualize the Gabor filter bank before training
    visualize_gabor_bank(model)

    #Train
    # Only non-frozen params passed to optimizer
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.01, momentum=0.5
    )

    train_losses  = []
    train_counter = []
    test_losses   = []
    gabor_accs    = []
    test_counter  = [len(train_loader.dataset) * e for e in range(1, N_EPOCHS+1)]

    # Original CNN per-epoch accuracies from task 1 (fill in your values)
    orig_accs = [94.5, 96.8, 97.4, 97.9, 98.1]

    print(f"Training GaborNet for {N_EPOCHS} epochs...\n")
    for epoch in range(1, N_EPOCHS + 1):
        print(f"── Epoch {epoch}/{N_EPOCHS} ──────────────────────")
        train_epoch(model, train_loader, optimizer, epoch,
                    train_losses, train_counter)
        acc = evaluate(model, test_loader, test_losses)
        gabor_accs.append(acc)

    #Compare and visualize
    plot_comparison(train_counter, train_losses,
                    test_counter,  test_losses,
                    orig_accs,     gabor_accs)
    visualize_gabor_outputs(model, test_loader)

    #Final summary 
    print("\n── Final Accuracy Comparison ────────────────────────")
    print(f"  Original CNN (fully trained) : {orig_accs[-1]:.1f}%")
    print(f"  GaborNet (fixed conv1)       : {gabor_accs[-1]:.1f}%")
    diff = gabor_accs[-1] - orig_accs[-1]
    print(f"  Difference                   : {diff:+.1f}%")


if __name__ == "__main__":
    main(sys.argv)