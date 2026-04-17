# Name: Ahilesh Vadivel
# Date: 5th April 2026
# Project 5 - Recognition using Deep Networks
# Task 5: Design your own experiment — Fashion MNIST CNN hyperparameter search

#Imports 
import sys
import csv
import time
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#Configurable CNN
class FlexNet(nn.Module):
    """
    A flexible CNN for Fashion MNIST whose key hyperparameters can all be
    set at construction time. This lets us swap configurations without
    rewriting the model.

    Parameters
    ----------
    conv1_filters : int   — number of filters in first conv layer
    conv2_filters : int   — number of filters in second conv layer
    fc_hidden     : int   — number of nodes in the hidden FC layer
    dropout_rate  : float — dropout probability (applied after conv2 and fc1)
    n_classes     : int   — number of output classes (10 for Fashion MNIST)
    """
    def __init__(self,
                 conv1_filters=10,
                 conv2_filters=20,
                 fc_hidden=50,
                 dropout_rate=0.5,
                 n_classes=10):
        super(FlexNet, self).__init__()

        self.conv1   = nn.Conv2d(1, conv1_filters, kernel_size=5)
        self.conv2   = nn.Conv2d(conv1_filters, conv2_filters, kernel_size=5)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.pool    = nn.MaxPool2d(2)

        # Compute flattened size after two conv+pool blocks on 28x28 input:
        # 28 → conv(5) → 24 → pool(2) → 12 → conv(5) → 8 → pool(2) → 4
        flat_size = conv2_filters * 4 * 4
        self.fc1  = nn.Linear(flat_size, fc_hidden)
        self.fc2  = nn.Linear(fc_hidden, n_classes)

    # Computes a forward pass through the network
    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.dropout(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.log_softmax(self.fc2(x), dim=1)
        return x


#Data Loading
def load_fashion_mnist(train_batch_size=64, test_batch_size=1000):
    """
    Loads the Fashion MNIST dataset.
    Fashion MNIST is a drop-in replacement for MNIST with 10 clothing
    categories — same size (28x28 grayscale) but more challenging.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))   # Fashion MNIST statistics
    ])
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.FashionMNIST('./data', train=True,
                                          download=True, transform=transform),
        batch_size=train_batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.FashionMNIST('./data', train=False,
                                          download=True, transform=transform),
        batch_size=test_batch_size, shuffle=False
    )
    return train_loader, test_loader


#Single Training Run
def run_experiment(config, n_epochs=5):
    """
    Trains and evaluates a FlexNet with the given config dict.
    Returns a result dict with accuracy, loss, and timing.

    config keys: conv1_filters, conv2_filters, fc_hidden,
                 dropout_rate, batch_size
    """
    train_loader, test_loader = load_fashion_mnist(
        train_batch_size=config['batch_size']
    )
    model     = FlexNet(
        conv1_filters=config['conv1_filters'],
        conv2_filters=config['conv2_filters'],
        fc_hidden=config['fc_hidden'],
        dropout_rate=config['dropout_rate']
    )
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    t_start = time.time()

    # Train for n_epochs
    for epoch in range(1, n_epochs + 1):
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            loss = F.nll_loss(model(data), target)
            loss.backward()
            optimizer.step()

    # Evaluate on test set
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            output      = model(data)
            total_loss += F.nll_loss(output, target, reduction='sum').item()
            correct    += output.argmax(dim=1).eq(target).sum().item()

    elapsed  = time.time() - t_start
    accuracy = 100.0 * correct / len(test_loader.dataset)
    avg_loss = total_loss / len(test_loader.dataset)

    result = {**config,
              'accuracy': round(accuracy, 3),
              'test_loss': round(avg_loss, 4),
              'train_time_s': round(elapsed, 1)}
    return result


#Experiment Plan
def build_experiment_plan():
    """
    Builds the full list of configurations to evaluate using a
    linear (round-robin) search strategy.

    Baseline config: conv1=10, conv2=20, fc=50, dropout=0.5, batch=64

    Phase 1 — vary conv filters     (baseline dropout/fc/batch fixed)
    Phase 2 — vary dropout rate     (best filters from P1, baseline fc/batch)
    Phase 3 — vary FC hidden nodes  (best filters+dropout, baseline batch)
    Phase 4 — vary batch size       (best filters+dropout+fc)
    Phase 5 — second pass: filters  (best dropout/fc/batch from P2-4)
    Phase 6 — second pass: dropout  (best from P1,P3,P4)
    Phase 7 — top grid search       (cross best values from each dimension)

    Total: approximately 60-70 configurations.
    """
    plans = []

    #Phase 1: Vary conv filters
    filter_options = [(8,16),(10,20),(16,32),(24,48),(32,64),(48,96)]
    for f1, f2 in filter_options:
        plans.append({'phase':1,'conv1_filters':f1,'conv2_filters':f2,
                      'fc_hidden':50,'dropout_rate':0.5,'batch_size':64})

    #Phase 2: Vary dropout rate
    dropout_options = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    for d in dropout_options:
        plans.append({'phase':2,'conv1_filters':10,'conv2_filters':20,
                      'fc_hidden':50,'dropout_rate':d,'batch_size':64})

    #Phase 3: Vary FC hidden nodes
    hidden_options = [32, 64, 128, 256, 512, 1024]
    for h in hidden_options:
        plans.append({'phase':3,'conv1_filters':10,'conv2_filters':20,
                      'fc_hidden':h,'dropout_rate':0.5,'batch_size':64})

    #Phase 4: Vary batch size
    batch_options = [16, 32, 64, 128, 256, 512]
    for b in batch_options:
        plans.append({'phase':4,'conv1_filters':10,'conv2_filters':20,
                      'fc_hidden':50,'dropout_rate':0.5,'batch_size':b})

    #Phase 5: Second pass — filters with updated baseline
    for f1, f2 in [(16,32),(24,48),(32,64)]:
        plans.append({'phase':5,'conv1_filters':f1,'conv2_filters':f2,
                      'fc_hidden':128,'dropout_rate':0.3,'batch_size':64})

    #Phase 6: Second pass — dropout with updated baseline 
    for d in [0.2, 0.3, 0.4]:
        plans.append({'phase':6,'conv1_filters':32,'conv2_filters':64,
                      'fc_hidden':128,'dropout_rate':d,'batch_size':64})

    #Phase 7: Top grid search — best values cross-combined
    top_filters  = [(16,32),(32,64)]
    top_dropouts = [0.2, 0.3]
    top_hidden   = [128, 256]
    top_batches  = [32, 64]
    for (f1,f2), d, h, b in itertools.product(top_filters, top_dropouts,
                                               top_hidden, top_batches):
        plans.append({'phase':7,'conv1_filters':f1,'conv2_filters':f2,
                      'fc_hidden':h,'dropout_rate':d,'batch_size':b})

    return plans


#Save Results 
def save_results(results, path="experiment_results.csv"):
    """Saves all experiment results to a CSV file for later analysis."""
    if not results:
        return
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to '{path}'")


#Plot Results 
def plot_results(results):
    """
    Generates four subplots — one per search dimension — showing how
    test accuracy varies as each hyperparameter changes.
    Phases 1-4 correspond directly to dimensions 1-4.
    Saves figure as 'experiment_results.png'.
    """
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Fashion MNIST — Hyperparameter Search Results", fontsize=14)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    #Helper: extract phase data and sort by x value
    def phase_data(phase, x_key, label_fn=None):
        rows = [r for r in results if r['phase'] == phase]
        rows.sort(key=lambda r: r[x_key])
        xs   = [label_fn(r) if label_fn else r[x_key] for r in rows]
        accs = [r['accuracy'] for r in rows]
        return xs, accs

    # Phase 1 — conv filters
    ax1 = fig.add_subplot(gs[0, 0])
    xs, accs = phase_data(1, 'conv1_filters',
                          lambda r: f"{r['conv1_filters']}/{r['conv2_filters']}")
    ax1.plot(xs, accs, 'bo-', linewidth=1.5, markersize=6)
    ax1.set_title("Dim 1 — Conv Filter Counts", fontsize=11)
    ax1.set_xlabel("conv1 / conv2 filters")
    ax1.set_ylabel("Test accuracy (%)")
    ax1.tick_params(axis='x', rotation=30)
    ax1.grid(True, alpha=0.3)

    # Phase 2 — dropout rate
    ax2 = fig.add_subplot(gs[0, 1])
    xs, accs = phase_data(2, 'dropout_rate')
    ax2.plot(xs, accs, 'rs-', linewidth=1.5, markersize=6)
    ax2.set_title("Dim 2 — Dropout Rate", fontsize=11)
    ax2.set_xlabel("Dropout rate")
    ax2.set_ylabel("Test accuracy (%)")
    ax2.grid(True, alpha=0.3)

    # Phase 3 — FC hidden nodes
    ax3 = fig.add_subplot(gs[1, 0])
    xs, accs = phase_data(3, 'fc_hidden')
    ax3.plot(xs, accs, 'g^-', linewidth=1.5, markersize=6)
    ax3.set_title("Dim 3 — FC Hidden Nodes", fontsize=11)
    ax3.set_xlabel("Hidden nodes")
    ax3.set_ylabel("Test accuracy (%)")
    ax3.grid(True, alpha=0.3)

    # Phase 4 — batch size
    ax4 = fig.add_subplot(gs[1, 1])
    xs, accs = phase_data(4, 'batch_size')
    ax4.plot(xs, accs, 'm*-', linewidth=1.5, markersize=8)
    ax4.set_title("Dim 4 — Batch Size", fontsize=11)
    ax4.set_xlabel("Batch size")
    ax4.set_ylabel("Test accuracy (%)")
    ax4.grid(True, alpha=0.3)

    plt.savefig("experiment_results.png", dpi=150)
    plt.show()
    print("Results plot saved as 'experiment_results.png'")


#Print Summary
def print_summary(results):
    """Prints the top 10 configurations sorted by test accuracy."""
    sorted_r = sorted(results, key=lambda r: r['accuracy'], reverse=True)
    print("\n── Top 10 Configurations ────────────────────────────────────────")
    print(f"{'#':<3} {'C1':>4} {'C2':>4} {'FC':>5} {'Drop':>6} "
          f"{'Batch':>6} {'Acc%':>7} {'Loss':>7} {'Time(s)':>8}")
    print("─" * 65)
    for i, r in enumerate(sorted_r[:10], 1):
        print(f"{i:<3} {r['conv1_filters']:>4} {r['conv2_filters']:>4} "
              f"{r['fc_hidden']:>5} {r['dropout_rate']:>6.2f} "
              f"{r['batch_size']:>6} {r['accuracy']:>7.3f} "
              f"{r['test_loss']:>7.4f} {r['train_time_s']:>8.1f}")
    print("─" * 65)
    best = sorted_r[0]
    print(f"\nBest config → conv1={best['conv1_filters']}, "
          f"conv2={best['conv2_filters']}, fc={best['fc_hidden']}, "
          f"dropout={best['dropout_rate']}, batch={best['batch_size']}")
    print(f"Best accuracy: {best['accuracy']:.3f}%")


#Main
def main(argv):
    """
    Entry point for Task 5.

    Hypotheses (pre-evaluation predictions):
      H1 — More conv filters will improve accuracy up to a point (~32/64),
           then plateau or slightly drop due to overfitting.
      H2 — Low dropout (<0.2) will overfit; high dropout (>0.6) will underfit.
           Optimal is expected around 0.3-0.4 for Fashion MNIST.
      H3 — Increasing FC hidden nodes will help up to ~256; beyond that
           the gains diminish and training slows significantly.
      H4 — Smaller batch sizes (32-64) will give slightly better accuracy
           but take longer to train; very large batches (512) will hurt accuracy.

    Strategy: linear (round-robin) search across 4 dimensions, then a
    focused grid search around the best values found.
    Total experiments: ~66 configurations, 5 epochs each.
    """
    torch.manual_seed(42)

    print("Building experiment plan...")
    plans = build_experiment_plan()
    print(f"Total configurations to evaluate: {len(plans)}\n")

    results     = []
    total       = len(plans)

    for i, config in enumerate(plans, 1):
        print(f"[{i:>3}/{total}] Phase {config['phase']} — "
              f"c1={config['conv1_filters']}, c2={config['conv2_filters']}, "
              f"fc={config['fc_hidden']}, drop={config['dropout_rate']}, "
              f"batch={config['batch_size']}", end="  →  ", flush=True)

        result = run_experiment(config, n_epochs=5)
        results.append(result)
        print(f"acc={result['accuracy']:.2f}%  "
              f"loss={result['test_loss']:.4f}  "
              f"time={result['train_time_s']:.1f}s")

        # Save incrementally so progress isn't lost if interrupted
        if i % 10 == 0:
            save_results(results)

    # Final save + analysis
    save_results(results)
    print_summary(results)
    plot_results(results)


if __name__ == "__main__":
    main(sys.argv)