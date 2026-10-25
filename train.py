"""
Deney 1 - MNIST uzerinde temel MLP (CNN yok, onceden egitilmis model yok).
Calistirma:  python train.py
"""
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ---------------- Hiperparametreler (raporda belirtilecek) ----------------
HP = {
    "hidden_sizes": [256, 128],   # 2 hidden layer
    "activation": "ReLU",
    "optimizer": "Adam",
    "learning_rate": 1e-3,
    "batch_size": 128,
    "epochs": 15,
    "val_size": 5000,             # 60000 -> 55000 train + 5000 validation
    "seed": 42,
}
MEAN, STD = 0.1307, 0.3081        # MNIST normalizasyon degerleri
# --------------------------------------------------------------------------

torch.manual_seed(HP["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Cihaz:", device)


class MLP(nn.Module):
    def __init__(self, hidden_sizes):
        super().__init__()
        layers, in_dim = [nn.Flatten()], 28 * 28
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 10))  # softmax CrossEntropyLoss icinde
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total_loss += criterion(out, y).item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def main():
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))])
    full_train = datasets.MNIST("data", train=True, download=True, transform=tf)
    test_set = datasets.MNIST("data", train=False, download=True, transform=tf)

    # Train / Validation ayrimi
    n_val = HP["val_size"]
    train_set, val_set = random_split(
        full_train, [len(full_train) - n_val, n_val],
        generator=torch.Generator().manual_seed(HP["seed"]),
    )
    print(f"Train: {len(train_set)} | Validation: {len(val_set)} | Test: {len(test_set)}")

    bs = HP["batch_size"]
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=bs)
    test_loader = DataLoader(test_set, batch_size=bs)

    model = MLP(HP["hidden_sizes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=HP["learning_rate"])

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    start = time.time()

    for epoch in range(1, HP["epochs"] + 1):
        model.train()
        run_loss, correct, n = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
        tr_loss, tr_acc = run_loss / n, correct / n
        va_loss, va_acc = evaluate(model, val_loader, criterion)
        for k, v in zip(hist, (tr_loss, va_loss, tr_acc, va_acc)):
            hist[k].append(v)
        print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} "
              f"| val loss {va_loss:.4f} acc {va_acc:.4f}")

    train_time = time.time() - start

    # Test verisi YALNIZCA burada, egitim bittikten sonra bir kez kullanilir
    test_loss, test_acc = evaluate(model, test_loader, criterion)

    results = {
        "hyperparameters": HP,
        "final_train_loss": hist["train_loss"][-1],
        "final_val_loss": hist["val_loss"][-1],
        "final_train_acc": hist["train_acc"][-1],
        "final_val_acc": hist["val_acc"][-1],
        "test_acc": test_acc,
        "test_loss": test_loss,
        "training_time_sec": train_time,
    }
    print(json.dumps(results, indent=2))
    with open("results_exp1.json", "w") as f:
        json.dump(results, f, indent=2)

    torch.save(model.state_dict(), "mlp_exp1.pt")

    # Grafikler
    ep = range(1, HP["epochs"] + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, hist["train_loss"], label="Train")
    ax[0].plot(ep, hist["val_loss"], label="Validation")
    ax[0].set_title("Loss"); ax[0].set_xlabel("Epoch"); ax[0].legend()
    ax[1].plot(ep, hist["train_acc"], label="Train")
    ax[1].plot(ep, hist["val_acc"], label="Validation")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("Epoch"); ax[1].legend()
    plt.tight_layout()
    plt.savefig("training_curves_exp1.png", dpi=150)
    print("Grafik kaydedildi: training_curves_exp1.png")


if __name__ == "__main__":
    main()
