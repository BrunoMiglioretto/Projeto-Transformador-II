# mobileNet_per_feature.py
# -*- coding: utf-8 -*-
# RODAR: python mobileNet_per_feature.py --data-root ../data

import os, random, sys, csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, UnidentifiedImageError

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==========================
# Config
# ==========================
@dataclass
class Config:
    data_root: str = "../data"
    train_csv: str = "Market1501/annotations/phase1/train/train.csv"
    # test_csv REMOVIDO (não usamos mais val.csv como teste)

    img_size: int = 224
    batch_size: int = 64
    epochs: int = 5
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    seed: int = 42

    # NOVO: fração do train.csv que será usada (desses, faremos 70/30 para treino/val)
    take_fraction: float = 0.60
    inner_val_ratio: float = 0.30  # 30% dos 60% para validação

    save_dir: str = "mobileNet_selectedFeatures"
    model_prefix: str = "mobilenet_v2_feature"


# ==========================
# Features selecionadas
# ==========================
SELECTED_ATTRS = [
    "LowerBody-Color-Grey",
    # "LowerBody-Color-Orange",
    # "LowerBody-Color-Pink","LowerBody-Color-Purple","LowerBody-Color-Red",
    # "LowerBody-Color-White","LowerBody-Color-Yellow","LowerBody-Color-Other",
    # "LowerBody-Type-Trousers&Shorts","LowerBody-Type-Skirt&Dress",

    # "Accessory-Backpack","Accessory-Bag",
    # "Accessory-Glasses-Normal","Accessory-Glasses-Sun","Accessory-Hat"
]


# ==========================
# Logger
# ==========================
class Logger:
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# ==========================
# Utils
# ==========================
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def safe_open_image(path: Path) -> Image.Image:
    try:
        return Image.open(path).convert("RGB")
    except FileNotFoundError:
        raise FileNotFoundError(f"Imagem não encontrada: {path}")
    except UnidentifiedImageError:
        raise ValueError(f"Arquivo não é imagem válida: {path}")

def choose_fraction_indices(n: int, fraction: float, seed: int):
    """Escolhe aleatoriamente floor(n * fraction) índices."""
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    k = max(1, int(np.floor(n * fraction)))
    return idx[:k].tolist()

def random_split_indices_local(indices: List[int], val_ratio: float, seed: int):
    """Divide uma lista de índices em (train, val) respeitando val_ratio."""
    rng = np.random.RandomState(seed)
    indices = np.array(indices)
    rng.shuffle(indices)
    k = max(1, int(np.floor(len(indices) * val_ratio)))
    val_idx = indices[:k].tolist()
    train_idx = indices[k:].tolist()
    return train_idx, val_idx


# ==========================
# Leitura CSV
# ==========================
class Schema:
    def __init__(self, mode: str, img_col: str, label_col: Optional[str], attr_cols: List[str]):
        self.mode = mode
        self.img_col = img_col
        self.label_col = label_col
        self.attr_cols = attr_cols

def read_csv_rows(csv_path: Path, selected_attrs=None) -> Tuple[Schema, List[Tuple[str, np.ndarray]]]:
    import csv as _csv
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        keys = reader.fieldnames
        img_col = "img_path" if "img_path" in keys else keys[0]
        label_col = "label" if "label" in keys else None

        if label_col:  # classificação única
            schema = Schema("single", img_col, label_col, [])
            rows = [(r[img_col], np.array([int(r[label_col])], dtype=np.int64)) for r in reader]
        elif selected_attrs:  # atributos multi-label
            attr_cols = [c for c in keys if c in selected_attrs]
            schema = Schema("multi", img_col, None, attr_cols)
            rows = []
            for r in reader:
                vals = [int(float(r[c])) if r[c] != "" else 0 for c in attr_cols]
                rows.append((r[img_col], np.array(vals, dtype=np.float32)))
        else:  # apenas imagens
            schema = Schema("images_only", img_col, None, [])
            rows = [(r[img_col], np.array([])) for r in reader]
    return schema, rows


# ==========================
# Dataset
# ==========================
class TaskDataset(Dataset):
    def __init__(self, root: Path, csv_rel: str, selected_attrs: List[str]):
        csv_path = root / csv_rel
        self.schema, self.samples = read_csv_rows(csv_path, selected_attrs)
        self.root = root
        self.transform = None
        if self.schema.mode == "single":
            labels = sorted({int(y[0]) for _, y in self.samples})
            self.label_to_idx = {y:i for i,y in enumerate(labels)}
            self.num_classes = len(self.label_to_idx)
        elif self.schema.mode == "multi":
            self.num_attrs = len(self.schema.attr_cols)

    def set_transform(self, t): self.transform = t
    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        rel, y = self.samples[i]
        img = safe_open_image(self.root / rel)
        if self.transform: img = self.transform(img)
        return img, y, rel


# ==========================
# Modelo
# ==========================
class MobileNetHead(nn.Module):
    def __init__(self, out_dim: int, pretrained: bool = True):
        super().__init__()
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        in_feats = 1280
        self.fc = nn.Linear(in_feats, out_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# ==========================
# Transforms
# ==========================
def make_transforms(size):
    return (
        transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(0.2,0.2,0.2,0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ]),
        transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
    )


# ==========================
# Métricas
# ==========================
def compute_metrics(y_true, y_pred):
    eps = 1e-9
    acc = (y_true == y_pred).mean()
    tp = np.logical_and(y_pred==1, y_true==1).sum()
    fp = np.logical_and(y_pred==1, y_true==0).sum()
    fn = np.logical_and(y_pred==0, y_true==1).sum()
    prec = tp / (tp+fp+eps)
    rec  = tp / (tp+fn+eps)
    f1   = 2*prec*rec / (prec+rec+eps)
    return acc, prec, rec, f1


# ==========================
# Treino por feature
# ==========================
def train_per_feature(cfg: Config, device: torch.device):
    root = Path(cfg.data_root)

    # Dataset completo do train.csv
    ds_full = TaskDataset(root, cfg.train_csv, SELECTED_ATTRS)

    # Seleciona 60% do train.csv
    n_total = len(ds_full)
    take_idx = choose_fraction_indices(n_total, cfg.take_fraction, cfg.seed)

    # Split 70/30 (treino/val) dentro dos 60%
    train_idx, val_idx = random_split_indices_local(take_idx, cfg.inner_val_ratio, cfg.seed)

    # Transforms
    tr_tf, te_tf = make_transforms(cfg.img_size)

    # Para garantir transforms independentes, criamos dois "views" do mesmo dataset
    train_ds = Subset(ds_full, train_idx)
    val_ds   = Subset(ds_full, val_idx)

    # Importante: usar instâncias separadas de TaskDataset para não sobrescrever transform
    train_base = TaskDataset(root, cfg.train_csv, SELECTED_ATTRS)
    val_base   = TaskDataset(root, cfg.train_csv, SELECTED_ATTRS)
    train_base.set_transform(tr_tf)
    val_base.set_transform(te_tf)
    train_ds.dataset = train_base
    val_ds.dataset   = val_base

    # Dataloaders
    train_ld = DataLoader(train_ds, cfg.batch_size, shuffle=True,  num_workers=cfg.num_workers)
    val_ld   = DataLoader(val_ds,   cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # dirs
    ckpt_dir = Path(cfg.save_dir) / "CHECKPOINTS"
    plot_dir = Path(cfg.save_dir) / "PLOTS-FEATURES"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # results.csv (mantido igual; métricas de teste ficam NaN)
    results_csv = Path(cfg.save_dir) / "results.csv"
    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Feature","BestValF1","TestAcc","TestPrec","TestRec","TestF1"])

    # loop por atributo
    for j, attr in enumerate(ds_full.schema.attr_cols):
        print(f"\n[INFO] Treinando modelo para atributo: {attr}")
        model = MobileNetHead(out_dim=1).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()

        best_val_f1 = -1.0
        history = {"train_loss": [], "val_loss": [], "acc": [], "prec": [], "rec": [], "f1": []}

        # ====================== TREINO ======================
        for ep in range(1, cfg.epochs+1):
            model.train(); running=0; n=0
            for x,y,_ in train_ld:
                x, target = x.to(device), y[:,j].unsqueeze(1).to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = loss_fn(logits, target)
                loss.backward()
                optimizer.step()
                running += loss.item()*x.size(0); n += x.size(0)
            train_loss = running/max(n,1)

            # validação (dos 60% selecionados)
            model.eval(); val_running=0; vn=0
            all_true=[]; all_pred=[]
            with torch.no_grad():
                for x,y,_ in val_ld:
                    x, target = x.to(device), y[:,j].unsqueeze(1).to(device)
                    logits = model(x)
                    loss = loss_fn(logits, target)
                    val_running += loss.item()*x.size(0); vn += x.size(0)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    preds = (probs >= 0.5).astype(int)
                    all_true.append(target.cpu().numpy())
                    all_pred.append(preds)
            val_loss = val_running/max(vn,1)
            y_true = np.vstack(all_true).ravel()
            y_pred = np.vstack(all_pred).ravel()
            acc, prec, rec, f1 = compute_metrics(y_true, y_pred)

            # histórico
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["acc"].append(acc)
            history["prec"].append(prec)
            history["rec"].append(rec)
            history["f1"].append(f1)

            print(f"[{attr} | Epoch {ep:03d}] "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"ACC={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}")

            if f1 > best_val_f1:
                best_val_f1 = f1
                save_path = ckpt_dir / f"{cfg.model_prefix}_{attr}.pth"
                torch.save({"model": model.state_dict(), "attr": attr}, save_path)
                print(f"  -> Melhor modelo salvo com F1={best_val_f1:.3f}: {save_path}")

        # ==== plot evolução ====
        plt.figure(figsize=(10,5))
        epochs = range(1, cfg.epochs+1)
        plt.plot(epochs, history["train_loss"], label="Train Loss")
        plt.plot(epochs, history["val_loss"], label="Val Loss")
        plt.plot(epochs, history["f1"], label="Val F1")
        plt.xlabel("Epoch"); plt.ylabel("Score / Loss")
        plt.title(f"Evolução - {attr}")
        plt.legend(); plt.tight_layout()
        plt.savefig(plot_dir / f"{attr}_curve.png")
        plt.close()

        # ====================== (Sem TESTE externo) ======================
        acc = prec = rec = f1 = float("nan")

        # salvar no CSV
        with open(results_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([attr, best_val_f1, acc, prec, rec, f1])


# ==========================
# Main
# ==========================
def main():
    import argparse
    parser=argparse.ArgumentParser("MobileNetV2 por feature com métricas")
    parser.add_argument("--data-root", type=str, default=Config.data_root)
    args=parser.parse_args()
    cfg=Config(data_root=args.data_root)

    # logger
    os.makedirs(cfg.save_dir, exist_ok=True)
    sys.stdout = Logger(Path(cfg.save_dir) / "train.log")

    set_seed(cfg.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    train_per_feature(cfg, device)

if __name__=="__main__":
    main()
