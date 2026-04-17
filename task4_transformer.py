# Name: Ahilesh Vadivel
# Date: 5th April 2026
# Project 5 - Recognition using Deep Networks
# Task 4: Re-implement the network using Transformer layers

#Imports
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

#Transformer Network
class NetTransformer(nn.Module):
    """
    Vision Transformer (ViT-style) for MNIST digit recognition.

    Pipeline:
      1. Patch embedding  — divide 28x28 image into overlapping patches,
                            project each patch to an embedding vector (token)
      2. Transformer encoder — stack of multi-head self-attention layers
      3. Token aggregation   — average all token outputs → single vector
      4. Classification head — Linear+ReLU+Dropout → Linear+LogSoftmax

    Default settings:
      patch_size=14, embed_dim=64, num_heads=4, num_layers=4,
      hidden_dim=128, dropout=0.1, n_classes=10
    """

    def __init__(self,
                 patch_size=14,      # size of each square image patch
                 embed_dim=64,       # dimension of each token embedding
                 num_heads=4,        # number of attention heads
                 num_layers=4,       # number of transformer encoder layers
                 hidden_dim=128,     # hidden size in classification head
                 dropout=0.1,        # dropout rate
                 n_classes=10):      # number of output classes
        super(NetTransformer, self).__init__()

        self.patch_size = patch_size

        #Step 1: Patch Embedding
        # Each patch is (patch_size * patch_size * 1) pixels (grayscale)
        # A Linear layer projects it to embed_dim → one token per patch
        patch_dim = patch_size * patch_size * 1   # 1 = grayscale channels
        self.patch_embedding = nn.Linear(patch_dim, embed_dim)

        #Step 2: Transformer Encoder
        # TransformerEncoderLayer: multi-head self-attention + feedforward block
        # batch_first=True → input shape is (batch, seq_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,   # standard 4× expansion
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        #Steps 3 & 4: Classification Head 
        # After averaging all tokens we get a single (batch, embed_dim) vector
        # FC1: embed_dim → hidden_dim  with ReLU + Dropout
        # FC2: hidden_dim → n_classes  with LogSoftmax
        self.fc1     = nn.Linear(embed_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2     = nn.Linear(hidden_dim, n_classes)


    def make_patches(self, x):
        """
        Divides input images into overlapping patches using unfold.

        x shape: (batch, 1, 28, 28)

        unfold(dim, size, step):
          - dim 2 (height): extract windows of size patch_size, step 1
          - dim 3 (width):  extract windows of size patch_size, step 1

        Returns patches of shape: (batch, num_patches, patch_size*patch_size)
        where num_patches = (28 - patch_size + 1)^2
        """
        p = self.patch_size
        # Slide patch window across both spatial dims with stride 1 (overlapping)
        patches = x.unfold(2, p, 1).unfold(3, p, 1)
        # patches shape: (batch, 1, n_h, n_w, p, p)
        # n_h = n_w = 28 - p + 1
        b, c, n_h, n_w, _, _ = patches.shape
        # Flatten spatial positions and patch pixels
        patches = patches.contiguous().view(b, n_h * n_w, c * p * p)
        return patches   # (batch, num_patches, patch_dim)


    def forward(self, x):
        """
        Full forward pass through the transformer network.

        x: (batch, 1, 28, 28)
        """
        # Step 1 — create tokens from patches
        patches = self.make_patches(x)               # (B, num_patches, patch_dim)
        tokens  = self.patch_embedding(patches)      # (B, num_patches, embed_dim)

        # Step 2 — run through transformer encoder
        encoded = self.transformer_encoder(tokens)   # (B, num_patches, embed_dim)

        # Step 3 — aggregate: average all token embeddings → single vector
        aggregated = encoded.mean(dim=1)             # (B, embed_dim)

        # Step 4 — classification head
        out = F.relu(self.fc1(aggregated))           # (B, hidden_dim)
        out = self.dropout(out)
        out = F.log_softmax(self.fc2(out), dim=1)    # (B, n_classes)
        return out


#Data Loaders
def load_data(train_batch_size=64, test_batch_size=1000):
    """Loads MNIST train and test sets with standard normalization."""
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
    """
    Runs one epoch of training and records loss every 10 batches.
    """
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
                batch_idx * len(data) + (epoch - 1) * len(train_loader.dataset)
            )
            print(f"  Epoch {epoch}  "
                  f"[{batch_idx * len(data):>6}/{len(train_loader.dataset)}]  "
                  f"Loss: {loss.item():.4f}")


#Evaluation
def evaluate(model, loader, losses, label="Test"):
    """
    Evaluates model accuracy and average loss on the given loader.
    """
    model.eval()
    total_loss = 0
    correct    = 0
    with torch.no_grad():
        for data, target in loader:
            output      = model(data)
            total_loss += F.nll_loss(output, target, reduction='sum').item()
            pred        = output.argmax(dim=1, keepdim=True)
            correct    += pred.eq(target.view_as(pred)).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = 100.0 * correct / len(loader.dataset)
    losses.append(avg_loss)
    print(f"  {label} — Avg loss: {avg_loss:.4f}  "
          f"Accuracy: {correct}/{len(loader.dataset)} ({accuracy:.1f}%)")
    return accuracy


#Plot Training Curve 
def plot_training_curve(train_counter, train_losses,
                        test_counter, test_losses):
    """
    Plots training loss (blue line) and test loss (red dots).
    Saves figure as 'transformer_training_curve.png'.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train_counter, train_losses,
            color='blue', linewidth=0.8, label='Train loss')
    ax.scatter(test_counter, test_losses,
               color='red', zorder=5, s=40, label='Test loss')
    ax.set_xlabel("Number of training examples seen")
    ax.set_ylabel("Negative log likelihood loss")
    ax.set_title("Transformer Model — Training vs Test Loss (MNIST)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("transformer_training_curve.png", dpi=150)
    plt.show()
    print("Plot saved as 'transformer_training_curve.png'")


#Main
def main(argv):
    """
    Entry point for Task 4.
    Builds and trains a Vision Transformer on MNIST with default settings,
    then plots and saves the training curve.

    Default NetTransformer settings:
      patch_size=14, embed_dim=64, num_heads=4,
      num_layers=4, hidden_dim=128, dropout=0.1
    """
    #Hyperparameters 
    N_EPOCHS         = 5
    TRAIN_BATCH_SIZE = 64
    LEARNING_RATE    = 0.001   # Adam works better than SGD for transformers

    torch.manual_seed(42)

    #Build model 
    model = NetTransformer(
        patch_size=14,
        embed_dim=64,
        num_heads=4,
        num_layers=4,
        hidden_dim=128,
        dropout=0.1,
        n_classes=10
    )
    print("── Transformer Model Structure ──────────────────────")
    print(model)

    # Count total trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {total_params:,}\n")

    #Data + Optimizer
    train_loader, test_loader = load_data(TRAIN_BATCH_SIZE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    #Training loop 
    train_losses  = []
    train_counter = []
    test_losses   = []
    test_counter  = [len(train_loader.dataset) * e for e in range(1, N_EPOCHS + 1)]

    print(f"Training transformer for {N_EPOCHS} epochs...\n")
    for epoch in range(1, N_EPOCHS + 1):
        print(f"── Epoch {epoch}/{N_EPOCHS} ──────────────────────")
        train_epoch(model, train_loader, optimizer, epoch,
                    train_losses, train_counter)
        evaluate(model, test_loader, test_losses, label="Test")

    #Plot + Save
    plot_training_curve(train_counter, train_losses,
                        test_counter, test_losses)
    torch.save(model.state_dict(), "transformer_model.pth")
    print("Model saved as 'transformer_model.pth'")


if __name__ == "__main__":
    main(sys.argv)