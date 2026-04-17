# Name: Ahilesh Vadivel
# Date: 4rd April 2026
# Project 5 - Recognition using Deep Networks
# Task 3: Transfer Learning — MNIST network → Greek letter recognition

#Imports 
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# Import MyNetwork directly from task1 (reuse without rewriting)
from task1_mnist import MyNetwork


#Greek Transform 
class GreekTransform:
    """
    Converts the 133x133 color greek letter images to match MNIST format:
      1. RGB → Grayscale (single channel)
      2. Affine scale: 36/128 shrinks image to approx 28x28 content area
      3. Center crop to exactly 28x28
      4. Invert intensities: black-on-white → white-on-black (like MNIST)
    """
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36/128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)


#Load Greek Data 
def load_greek_data(training_set_path, batch_size=5):
    """
    Loads the greek letter dataset using ImageFolder.
    ImageFolder expects subfolders named after each class:
        greek_train/
            alpha/   (9 images)
            beta/    (9 images)
            gamma/   (9 images)
    Returns a DataLoader with shuffling enabled.
    """
    greek_train = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            training_set_path,
            transform=transforms.Compose([
                transforms.ToTensor(),
                GreekTransform(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        ),
        batch_size=batch_size,
        shuffle=True
    )
    print(f"Greek dataset loaded from '{training_set_path}'")
    print(f"  Classes : {greek_train.dataset.classes}")
    print(f"  Samples : {len(greek_train.dataset)}\n")
    return greek_train


#Build Transfer Model
def build_transfer_model(model_path="mnist_model.pth"):
    """
    Loads the pre-trained MNIST network and adapts it for greek letter
    classification via transfer learning:
      1. Load saved MNIST weights
      2. Freeze ALL parameters (no gradient updates for existing layers)
      3. Replace fc2 (the last layer) with a new Linear(50 → 3)
         — this new layer is NOT frozen, so only it gets trained
    """
    # Step 1: load pre-trained MNIST model
    model = MyNetwork()
    model.load_state_dict(torch.load(model_path))
    print("Pre-trained MNIST weights loaded.\n")
    print("── Original Model ───────────────────────────────────")
    print(model)

    # Step 2: freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Step 3: replace last layer — fc2 now maps 50 → 3 (alpha, beta, gamma)
    # This new layer is not frozen (requires_grad=True by default)
    model.fc2 = nn.Linear(50, 3)

    print("\n── Modified Model (fc2 replaced: 50 → 3) ───────────")
    print(model)
    print()
    return model


#Train Transfer Model
def train_greek(model, greek_train, n_epochs=50):
    """
    Trains only the new fc2 layer on the greek letter dataset.
    All other layers are frozen — their weights don't change.
    Records the average loss per epoch for plotting.
    Returns the list of per-epoch losses.
    """
    # Only parameters with requires_grad=True (i.e. fc2) are updated
    optimizer   = optim.Adam(filter(lambda p: p.requires_grad,
                                    model.parameters()), lr=0.01)
    epoch_losses = []

    print(f"Training for {n_epochs} epochs...\n")
    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct    = 0
        total      = 0

        for data, target in greek_train:
            optimizer.zero_grad()
            output      = model(data)
            loss        = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pred        = output.argmax(dim=1)
            correct    += pred.eq(target).sum().item()
            total      += len(target)

        avg_loss = epoch_loss / len(greek_train)
        accuracy = 100.0 * correct / total
        epoch_losses.append(avg_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{n_epochs}  "
                  f"Loss: {avg_loss:.4f}  "
                  f"Accuracy: {correct}/{total} ({accuracy:.1f}%)")

    print("\nTraining complete.")
    return epoch_losses


#Plot Training Error
def plot_training_error(epoch_losses):
    """
    Plots the average training loss per epoch.
    Saves the figure as 'greek_training_loss.png'.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(epoch_losses) + 1), epoch_losses,
            color='blue', linewidth=1.5, marker='o', markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average NLL Loss")
    ax.set_title("Transfer Learning — Training Loss on Greek Letters")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("greek_training_loss.png", dpi=150)
    plt.show()
    print("Training loss plot saved as 'greek_training_loss.png'")


#Save Transfer Model
def save_model(model, path="greek_model.pth"):
    """Saves the transfer-learned model weights to disk."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to '{path}'")


#Test on Custom Greek Images
def test_custom_greek(model, image_folder="custom_greek",
                      class_names=["alpha", "beta", "gamma"]):
    """
    Loads custom greek letter images (your own handwritten examples),
    preprocesses them to match the greek training format, runs them
    through the network, and plots the results in a grid.

    Expects files named alpha_1.jpeg, beta_1.jpeg, gamma_1.jpeg, etc.
    in the custom_greek/ folder.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        GreekTransform(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    results = []
    for fname in sorted(os.listdir(image_folder)):
        if not fname.lower().endswith(('.jpeg', '.jpg', '.png')):
            continue

        # Infer true label from filename prefix (alpha/beta/gamma)
        true_label = -1
        for idx, name in enumerate(class_names):
            if fname.lower().startswith(name):
                true_label = idx
                break

        img_path = os.path.join(image_folder, fname)
        img      = Image.open(img_path).convert("RGB")
        tensor   = transform(img).unsqueeze(0)   # (1,1,28,28)

        model.eval()
        with torch.no_grad():
            output = model(tensor)
            pred   = output.argmax(dim=1).item()

        match = "✓" if pred == true_label else "✗"
        print(f"  {fname:<20} → Pred: {class_names[pred]:<6}  "
              f"True: {class_names[true_label] if true_label >= 0 else '?':<6} {match}")
        results.append((tensor.squeeze(), fname, pred,
                        true_label, class_names))

    if not results:
        print("No custom images found in 'custom_greek/'.")
        return

    # Plot results
    n    = len(results)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(8, 3 * rows))
    fig.suptitle("Custom Greek Letter Predictions", fontsize=13)

    for i, ax in enumerate(axes.flat if rows > 1 else [axes] if n == 1 else axes):
        if i < n:
            img, fname, pred, true_label, names = results[i]
            display = img * 0.3081 + 0.1307      # undo normalization for display
            ax.imshow(display.numpy(), cmap='gray')
            color = 'green' if pred == true_label else 'red'
            ax.set_title(f"Pred: {names[pred]}\n({fname})",
                         fontsize=9, color=color)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("custom_greek_results.png", dpi=150)
    plt.show()
    print("Custom results saved as 'custom_greek_results.png'")


#Main
def main(argv):
    """
    Entry point for Task 3 — Transfer Learning on Greek Letters.
    Steps:
      1. Load greek dataset via ImageFolder + GreekTransform
      2. Build transfer model (freeze MNIST weights, replace fc2 → 3 classes)
      3. Train on greek letters and plot training error
      4. Save the model
      5. Test on your own custom greek letter images
    """
    # Path to the folder containing alpha/, beta/, gamma/ subfolders
    training_set_path = "greek_train\\greek_train"

    # Steps 1-4: train transfer model
    greek_train  = load_greek_data(training_set_path, batch_size=5)
    model        = build_transfer_model("mnist_model.pth")
    epoch_losses = train_greek(model, greek_train, n_epochs=50)
    plot_training_error(epoch_losses)
    save_model(model, "greek_model.pth")

    # Step 5: test on your own images
    print("\n── Testing on custom greek images ───────────────────")
    test_custom_greek(model, image_folder="custom_greek")


if __name__ == "__main__":
    main(sys.argv)