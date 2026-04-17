# Name: Ahilesh Vadivel
# Date: 3rd April 2026
# Project 5 - Recognition using Deep Networks
# Tasks 1A, 1B, 1C: Load MNIST, define CNN, train and evaluate

#Imports
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


#Network Definition
class MyNetwork(nn.Module):
    """
    CNN for MNIST digit recognition.
    Architecture:
        Conv(10, 5x5) → MaxPool(2x2) + ReLU
        Conv(20, 5x5) → Dropout(0.5) → MaxPool(2x2) + ReLU
        Flatten → Linear(320→50) + ReLU
        Linear(50→10) + LogSoftmax
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
        # Block 1: Conv → Pool → ReLU
        x = F.relu(self.pool(self.conv1(x)))
        # Block 2: Conv → Dropout → Pool → ReLU
        x = F.relu(self.pool(self.dropout(self.conv2(x))))
        # Flatten: (batch, 20, 4, 4) → (batch, 320)
        x = x.view(-1, 320)
        # FC layers
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x


#Data Loaders
def load_data(train_batch_size=64, test_batch_size=1000):
    """
    Returns DataLoaders for the MNIST training and test sets.
    Training set is shuffled; test set is not (keeps order deterministic).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', train=True,  download=True, transform=transform),
        batch_size=train_batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform),
        batch_size=test_batch_size, shuffle=False
    )
    return train_loader, test_loader


#Display First Six Test Digits
def plot_first_six_test_digits(test_loader):
    """
    Plots the first six MNIST test digits in a 2x3 grid and saves the figure.
    """
    images, labels = next(iter(test_loader))
    fig, axes = plt.subplots(2, 3, figsize=(7, 5))
    fig.suptitle("First 6 MNIST Test Digits", fontsize=14)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].squeeze().numpy(), cmap='gray')
        ax.set_title(f"Label: {labels[i].item()}", fontsize=11)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("first_six_test_digits.png", dpi=150)
    plt.show()


#Training
def train_network(model, train_loader, optimizer, epoch,
                  train_losses, train_counter):
    """
    Runs one epoch of training.
    Records the negative log-likelihood loss after every 10 batches.
    train_losses / train_counter are lists that accumulate across epochs.
    """
    model.train()                       # sets dropout/batchnorm to training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()           # clear gradients from last step
        output = model(data)            # forward pass
        loss   = F.nll_loss(output, target)   # negative log-likelihood loss
        loss.backward()                 # backpropagation
        optimizer.step()                # update weights

        # Record loss every 10 batches
        if batch_idx % 10 == 0:
            train_losses.append(loss.item())
            train_counter.append(
                batch_idx * len(data) + (epoch - 1) * len(train_loader.dataset)
            )
            print(f"  Epoch {epoch}  "
                  f"[{batch_idx * len(data):>6}/{len(train_loader.dataset)}]  "
                  f"Loss: {loss.item():.4f}")


#Evaluation
def evaluate_network(model, loader, losses, label="Test"):
    """
    Evaluates the model on the given loader.
    Appends the average loss to the provided losses list.
    Returns accuracy (%) over the full dataset.
    """
    model.eval()                        # disables dropout during evaluation
    total_loss = 0
    correct    = 0
    with torch.no_grad():               # no gradient computation needed
        for data, target in loader:
            output      = model(data)
            total_loss += F.nll_loss(output, target, reduction='sum').item()
            pred        = output.argmax(dim=1, keepdim=True)
            correct    += pred.eq(target.view_as(pred)).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = 100.0 * correct / len(loader.dataset)
    losses.append(avg_loss)
    print(f"  {label} set — Avg loss: {avg_loss:.4f}  "
          f"Accuracy: {correct}/{len(loader.dataset)} ({accuracy:.1f}%)")
    return accuracy


#Plot Training Curve
def plot_training_curve(train_counter, train_losses,
                        test_counter, test_losses):
    """
    Plots training loss (blue line) vs test loss (red dots) against
    the number of training examples seen, matching the style in the spec.
    Saves the figure as 'training_curve.png'.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(train_counter, train_losses,
            color='blue', linewidth=0.8, label='Train loss')
    ax.scatter(test_counter, test_losses,
               color='red', zorder=5, s=40, label='Test loss')

    ax.set_xlabel("Number of training examples seen")
    ax.set_ylabel("Negative log likelihood loss")
    ax.set_title("Training vs Test Loss — MNIST CNN")
    ax.legend()
    plt.tight_layout()
    plt.savefig("training_curve.png", dpi=150)
    plt.show()
    print("Plot saved as 'training_curve.png'")


#Save / Load Model
def save_model(model, optimizer,
               model_path="mnist_model.pth",
               optim_path="mnist_optimizer.pth"):
    """Saves the model and optimizer state dicts to disk for later tasks."""
    torch.save(model.state_dict(), model_path)
    torch.save(optimizer.state_dict(), optim_path)
    print(f"Model saved to '{model_path}'")


#Main
def main(argv):
    """
    Entry point.
    Loads data, trains the CNN for N_EPOCHS, evaluates after each epoch,
    plots the training curve, and saves the model to disk.
    """
    #Hyper-parameters
    N_EPOCHS          = 5
    TRAIN_BATCH_SIZE  = 64
    TEST_BATCH_SIZE   = 1000
    LEARNING_RATE     = 0.01
    MOMENTUM          = 0.5

    #Setup
    torch.manual_seed(42)               # reproducibility
    train_loader, test_loader = load_data(TRAIN_BATCH_SIZE, TEST_BATCH_SIZE)

    # Task 1A — display first six test digits
    plot_first_six_test_digits(test_loader)

    # Task 1B — build network
    model     = MyNetwork()
    optimizer = optim.SGD(model.parameters(),
                          lr=LEARNING_RATE, momentum=MOMENTUM)
    print(model)

    #Accumulators for plotting
    train_losses  = []
    train_counter = []
    test_losses   = []
    # x-axis positions for test points: end of each epoch
    test_counter  = [len(train_loader.dataset) * e for e in range(1, N_EPOCHS + 1)]

    #Task 1C — train + evaluate for N epochs
    print(f"\nTraining for {N_EPOCHS} epochs...\n")
    for epoch in range(1, N_EPOCHS + 1):
        print(f"── Epoch {epoch}/{N_EPOCHS} ──────────────────────")
        train_network(model, train_loader, optimizer, epoch,
                      train_losses, train_counter)
        evaluate_network(model, test_loader, test_losses, label="Test")

    #Plot training curve
    plot_training_curve(train_counter, train_losses,
                        test_counter, test_losses)

    #Save model for later tasks
    save_model(model, optimizer)

    return


if __name__ == "__main__":
    main(sys.argv)