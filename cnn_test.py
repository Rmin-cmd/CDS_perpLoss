import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# --- 1) Simple encoder (swap in your own) ---
class SimpleCNN(nn.Module):
    def __init__(self, z_dim=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,stride=1,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,stride=1,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*8*8, z_dim)
        )
    def forward(self, x):
        return self.features(x)  # [B, z_dim]

# --- 2) DP-means clustering ---
def dp_means_clustering(embs, lamda):
    """
    Sequential DP-means: start with first point as center, add new center
    whenever a point is more than sqrt(lamda) away from all existing centers.
    """
    centers = [embs[0]]
    for z in embs[1:]:
        d2 = torch.stack([torch.sum((z-μ)**2) for μ in centers])
        if torch.min(d2) > lamda:
            centers.append(z)
    return torch.stack(centers)  # [K_c, D]

# --- 3) Global IMP classifier module ---
class GlobalIMPClassifier(nn.Module):
    def __init__(self, encoder, num_classes, z_dim, lamda, learn_sigma=True):
        super().__init__()
        self.encoder     = encoder
        self.C           = num_classes
        self.z_dim       = z_dim
        self.lamda       = lamda
        # optional learnable σ
        init_logσ = torch.log(torch.tensor(1.0))
        self.log_sigma  = nn.Parameter(init_logσ) if learn_sigma else init_logσ
        # placeholder for prototypes & radii
        self.register_buffer('protos', torch.zeros(0))
        self.register_buffer('radii',  torch.zeros(0))
        self.cluster_labels = []

    @torch.no_grad()
    def cluster_all(self, train_loader, device):
        """Embed entire train set, then DP-means cluster per class."""
        all_embs, all_lbls = [], []
        self.encoder.eval()
        for x,y in train_loader:
            z = self.encoder(x.to(device))
            all_embs.append(z.cpu())
            all_lbls.append(y)
        all_embs  = torch.cat(all_embs, dim=0)
        all_lbls  = torch.cat(all_lbls, dim=0)
        protos, radii, labels = [], [], []
        σ = self.log_sigma.exp().item()
        for c in range(self.C):
            mask  = (all_lbls==c)
            embs_c= all_embs[mask]
            if embs_c.numel()==0:
                continue
            centers = dp_means_clustering(embs_c, self.lamda)  # [Kc,D]
            Kc = centers.size(0)
            protos.append(centers)
            radii.append(torch.full((Kc,), σ))
            labels.extend([c]*Kc)
        # consolidate
        self.protos = torch.cat(protos, dim=0).to(device)  # [K_total, D]
        self.radii  = torch.cat(radii,  dim=0).to(device)  # [K_total]
        self.cluster_labels = torch.tensor(labels, dtype=torch.long, device=device)

    def forward(self, x, y):
        """
        x: [B,3,32,32], y: [B]
        returns: loss scalar, accuracy
        """
        B = x.size(0)
        z = self.encoder(x)                    # [B, D]
        D = self.z_dim
        K = self.protos.size(0)
        # compute log‐likelihoods: [B, K]
        # log p ∝ -||z - μ||^2/(2σ^2) - D*log(σ√2π)
        diff = z.unsqueeze(1) - self.protos.unsqueeze(0)     # [B,K,D]
        d2   = torch.sum(diff*diff, dim=2)                   # [B,K]
        inv2σ2 = 1.0/(2*(self.log_sigma.exp()**2))
        log_norm = - D * torch.log(self.log_sigma.exp()*math.sqrt(2*math.pi))
        logps = - d2 * inv2σ2 + log_norm                     # [B,K]

        # reduce to class‐wise scores by taking max over clusters of each class
        class_scores = []
        for c in range(self.C):
            mask = (self.cluster_labels==c)
            if mask.any():
                class_scores.append(logps[:,mask].max(dim=1).values)
            else:
                # if no cluster for class c, give -inf
                class_scores.append(torch.full((B,), float('-inf'), device=x.device))
        logits = torch.stack(class_scores, dim=1)            # [B, C]

        loss = F.cross_entropy(logits, y)
        acc  = (logits.argmax(dim=1)==y).float().mean()
        return loss, acc

# --- 4) Training script ---
def train_global_imp():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    train_ds = datasets.CIFAR10('.\CIFAR', train=True,  download=True, transform=transform)
    val_ds   = datasets.CIFAR10('.\CIFAR', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=False, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False, num_workers=4)

    # model
    encoder = SimpleCNN(z_dim=64)
    model   = GlobalIMPClassifier(encoder, num_classes=10, z_dim=64, lamda=0.3).to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 1) cluster once before training
    model.cluster_all(train_loader, device)

    # 2) epoch loop (classify train & val)
    for epoch in tqdm(range(1,11)):
        model.train()
        running_loss, running_acc = 0.0, 0.0
        for x,y in DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4):
            x,y = x.to(device), y.to(device)
            loss, acc = model(x,y)
            opt.zero_grad(); loss.backward(); opt.step()
            running_loss += loss.item()
            running_acc  += acc.item()
        print(f"[Train] Epoch {epoch}  loss={running_loss/len(train_ds):.4f}  acc={running_acc/len(train_ds):.4f}")

        model.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(device), y.to(device)
                loss, acc = model(x,y)
                val_loss += loss.item(); val_acc += acc.item()
        print(f"[Val]   Epoch {epoch}  loss={val_loss/len(val_ds):.4f}  acc={val_acc/len(val_ds):.4f}")

if __name__ == "__main__":
    train_global_imp()
