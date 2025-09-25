# -*- coding: utf-8 -*-
# feature_selection.py
#
# Seleção de features a partir de embeddings MobileNetV2
# usando dataset de atributos (multi-label a partir do CSV).
#
# Gera: top_features.json e plot_feature_scores.png

import torch
import os

# limitar threads PyTorch
torch.set_num_threads(max(1, int(os.cpu_count() * 0.9)))

import numpy as np
np.set_printoptions(threshold=20)

import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, models

from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")
device = torch.device("cpu")


# ==========================
# Config
# ==========================
@dataclass
class Config:
    data_root: str = "../data"
    train_csv: str = str(Path(data_root) / "Market1501/annotations/phase1/train/train.csv")

    img_size: int = 224
    batch_size: int = 16
    num_workers: int = 0

    save_dir: str = "checkpoints"


# ==========================
# MobileNetHead
# ==========================
class MobileNetHead(nn.Module):
    def __init__(self, out_dim: int = 1280, pretrained: bool = True):
        super().__init__()
        try:
            base = models.mobilenet_v2(
                weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
            )
        except Exception:
            base = models.mobilenet_v2(pretrained=pretrained)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.bn = nn.BatchNorm1d(1280)

    def forward(self, x, return_embedding: bool = True):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.bn(x)
        return nn.functional.normalize(x, p=2, dim=1)


# ==========================
# Dataset
# ==========================
class AttributeDataset(Dataset):
    def __init__(self, csv_path: Path, img_size: int, data_root: Path):
        self.df = pd.read_csv(csv_path, index_col=0)
        self.img_paths = [Path(data_root) / idx for idx in self.df.index]
        self.labels = self.df.values.astype(np.float32)
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        y = self.labels[idx]
        return x, y


# ==========================
# Extração de embeddings
# ==========================
@torch.no_grad()
def extract_embeddings(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device)
        z = model(x, return_embedding=True)
        feats.append(z.cpu().numpy())
        labels.append(y.numpy())
    X = np.vstack(feats).astype(np.float32)
    Y = np.vstack(labels).astype(np.float32)  # multi-label
    return X, Y


# ==========================
# Ranqueamento de features por atributo
# ==========================
def rank_features_multi(X: np.ndarray, Y: np.ndarray, attr_names, top_k=30):
    results = {}
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    for j, attr in enumerate(attr_names):
        y = Y[:, j].astype(int)  # binário
        if len(np.unique(y)) < 2:
            continue  # ignora atributo sem variação

        f_vals, _ = f_classif(Xs, y)
        mi_vals = mutual_info_classif(Xs, y, random_state=42)

        def zscore(a): return (a - a.mean()) / (a.std() + 1e-9)
        combo = 0.5*zscore(f_vals) + 0.5*zscore(mi_vals)

        order = np.argsort(-combo)[:top_k]
        results[attr] = {
            "top_indices": order.tolist(),
            "top_scores": combo[order].tolist(),
        }

    return results


# ==========================
# Plot (exemplo do 1º atributo)
# ==========================
def plot_first_attr(ranking: dict, out_path: Path):
    if not ranking:
        return None
    first_attr = next(iter(ranking))
    data = ranking[first_attr]
    idx, sc = data["top_indices"], data["top_scores"]
    plt.figure(figsize=(10,5))
    x = np.arange(len(idx))
    plt.bar(x, sc)
    plt.xticks(x, [str(i) for i in idx], rotation=90)
    plt.title(f"Importância - {first_attr}")
    plt.tight_layout()
    plt.savefig(out_path)
    return str(out_path)


# ==========================
# Pipeline
# ==========================
def run_feature_selection(cfg: Config, device: torch.device):
    os.makedirs(cfg.save_dir, exist_ok=True)

    ds = AttributeDataset(cfg.train_csv, cfg.img_size, Path(cfg.data_root))
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = MobileNetHead(pretrained=True).to(device)

    X, Y = extract_embeddings(model, loader, device)
    ranking = rank_features_multi(X, Y, ds.df.columns, top_k=30)

    json_path = Path(cfg.save_dir) / "top_features.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ranking, f, ensure_ascii=False, indent=2)

    png_path = Path(cfg.save_dir) / "plot_feature_scores.png"
    plot_first_attr(ranking, png_path)

    return {"ranking": ranking, "json_file": str(json_path), "plot_file": str(png_path)}


# ==========================
# Main
# ==========================
if __name__ == "__main__":
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    result = run_feature_selection(cfg, device)

    print("[FS] Ranking salvo em:", result["json_file"])
    print("[FS] Gráfico salvo em:", result["plot_file"])
