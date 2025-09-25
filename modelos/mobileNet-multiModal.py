# mobileNet.py
# -*- coding: utf-8 -*-

# RODAR COM CMD: python mobileNet.py --data-root data

import os, csv, time, random
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

# ==========================
# Config
# ==========================
@dataclass
class Config:
    data_root: str = "../data"  # raiz que contém Market1501/, PA100k/, etc.
    train_csv: str = "Market1501/annotations/phase1/train/train.csv"
    val1_csv: str   = "Market1501/annotations/phase1/val_task1/val.csv"
    val2_q_csv: str = "Market1501/annotations/phase1/val_task2/val_queries.csv"
    val2_g_csv: str = "Market1501/annotations/phase1/val_task2/val_imgs.csv"

    img_size: int = 224
    batch_size: int = 64
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    seed: int = 42
    amp: bool = True
    val_ratio: float = 0.30

    save_dir: str = "checkpoints"
    model_file: str = "mobilenet_v2_market1501.pth"

# ==========================
# Utils
# ==========================
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def _first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists(): return p
    return None

def resolve_csv(base: Path, rel: str, try_train_subdir: bool = True) -> Path:
    cand = [base / rel]
    if try_train_subdir:
        p = base / rel
        cand.append(p.parent / "train" / p.name)
    found = _first_existing(cand)
    if not found:
        raise FileNotFoundError("CSV não encontrado. tentei: " + " | ".join(map(str, cand)))
    return found

def safe_open_image(path: Path) -> Image.Image:
    try:
        return Image.open(path).convert("RGB")
    except FileNotFoundError:
        raise FileNotFoundError(f"Imagem não encontrada: {path}")
    except UnidentifiedImageError:
        raise ValueError(f"Arquivo não é imagem válida: {path}")

# ==========================
# Leitura de CSV (robusta)
# ==========================
class Schema:
    # mode ∈ {"single", "multi", "unlabeled"}
    def __init__(self, mode: str, img_col: str, label_col: Optional[str], attr_cols: List[str]):
        self.mode = mode
        self.img_col = img_col
        self.label_col = label_col
        self.attr_cols = attr_cols

def read_csv_rows(csv_path: Path) -> Tuple[Schema, List[Tuple[str, np.ndarray]]]:
    """
    Formatos suportados:
      - '# a1,a2,...' na 1ª linha → multi-label de atributos.
      - cabeçalho normal (img_path + label/person_id/...) → single/multi.
      - sem cabeçalho e sem vírgula na 1ª linha → UNLABELED (apenas caminhos).
    """
    with open(csv_path, 'r', encoding='utf-8') as f:
        raw_lines = [ln.rstrip("\n") for ln in f if ln.strip() != ""]
    if not raw_lines:
        raise ValueError(f"{csv_path} está vazio.")

    # 1) atributos com cabeçalho comentado
    if raw_lines[0].lstrip().startswith("#"):
        attr_names = [s.strip() for s in raw_lines[0].lstrip("#").strip().split(",")]
        schema = Schema("multi", "img_path", None, attr_names)
        rows: List[Tuple[str, np.ndarray]] = []
        for ln in raw_lines[1:]:
            if ln.lstrip().startswith("#"):  # comentários adicionais
                continue
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) < 1 + len(attr_names):
                raise ValueError(f"Linha com colunas insuficientes:\n{ln}")
            img_rel = parts[0]
            vals = parts[1:1+len(attr_names)]
            y = np.array([int(float(v)) if v != "" else 0 for v in vals], dtype=np.float32)
            rows.append((img_rel, y))
        return schema, rows

    # 2) sem vírgula → apenas caminhos (unlabeled)
    if "," not in raw_lines[0]:
        schema = Schema("unlabeled", "img_path", None, [])
        rows = [(ln.strip(), np.array([], dtype=np.float32)) for ln in raw_lines]
        return schema, rows

    # 3) cabeçalho normal
    import csv as _csv
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        try:
            first = next(iter(reader))
        except StopIteration:
            raise ValueError(f"{csv_path} está vazio.")
        keys = [k for k in first.keys() if isinstance(k, str)]
        kmap = {k.lower(): k for k in keys}
        img_candidates = ["img_path","image","path","filepath","img","file","relpath"]
        label_candidates = ["label","person_id","pid","id","class","target"]

        img_col = next((kmap[k] for k in img_candidates if k in kmap), keys[0])
        label_col = next((kmap[k] for k in label_candidates if k in kmap), None)

        f.seek(0); reader = _csv.DictReader(f)
        rows: List[Tuple[str, np.ndarray]] = []
        if label_col is not None:
            schema = Schema("single", img_col, label_col, [])
            for r in reader:
                if r.get(img_col) is None: continue
                rows.append((r[img_col].strip(), np.array([int(r[label_col])], dtype=np.int64)))
        else:
            attr_cols = [c for c in keys if c != img_col and not c.strip().startswith("#")]
            if not attr_cols:
                schema = Schema("unlabeled", img_col, None, [])
                for r in reader:
                    if r.get(img_col) is None: continue
                    rows.append((r[img_col].strip(), np.array([], dtype=np.float32)))
            else:
                schema = Schema("multi", img_col, None, attr_cols)
                for r in reader:
                    if r.get(img_col) is None: continue
                    vals = [int(float(r[c])) if (r[c] is not None and r[c] != '') else 0 for c in attr_cols]
                    rows.append((r[img_col].strip(), np.array(vals, dtype=np.float32)))
        return schema, rows

# ==========================
# Datasets
# ==========================
class TaskDataset(Dataset):
    def __init__(self, root: Path, csv_rel: str, want_train: bool):
        csv_path = resolve_csv(root, csv_rel, try_train_subdir=want_train)
        self.schema, self.samples = read_csv_rows(csv_path)
        self.root = root
        self.transform = None

        if self.schema.mode == "single":
            labels = sorted({int(y[0]) for _, y in self.samples})
            self.label_to_idx = {y:i for i,y in enumerate(labels)}
            self.idx_to_label = {i:y for y,i in self.label_to_idx.items()}
            self.num_classes = len(self.label_to_idx)
        elif self.schema.mode == "multi":
            self.num_attrs = len(self.samples[0][1])

    def set_transform(self, t): self.transform = t
    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        rel, y = self.samples[i]
        img = safe_open_image(self.root / rel)
        if self.transform: img = self.transform(img)
        if self.schema.mode == "single":
            return img, torch.tensor(self.label_to_idx[int(y[0])], dtype=torch.long)
        elif self.schema.mode == "multi":
            return img, torch.tensor(y, dtype=torch.float32), rel
        else:  # unlabeled
            return img, rel

# ==========================
# Modelo
# ==========================
class MobileNetHead(nn.Module):
    def __init__(self, out_dim: int, pretrained: bool = True):
        super().__init__()
        try:
            base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
        except Exception:
            base = models.mobilenet_v2(pretrained=pretrained)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        in_feats = 1280
        self.bn = nn.BatchNorm1d(in_feats)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(in_feats, out_dim)

    def forward(self, x, return_embedding=False):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.bn(x)
        if return_embedding:
            return nn.functional.normalize(x, p=2, dim=1)
        x = self.dropout(x)
        return self.fc(x)

# ==========================
# Transforms
# ==========================
def make_transforms(size):
    train_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return train_tf, test_tf

# ==========================
# Métricas
# ==========================
def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(-y_score)
    y_true = y_true[order]
    if y_true.sum() == 0: return 0.0
    cumsum = np.cumsum(y_true)
    precision = cumsum / (np.arange(len(y_true)) + 1)
    return float((precision * y_true).sum() / y_true.sum())

def f1_macro(y_true: np.ndarray, y_score: np.ndarray, thr=0.5) -> float:
    y_pred = (y_score >= thr).astype(np.int32)
    eps = 1e-9
    f1s = []
    for j in range(y_true.shape[1]):
        tp = np.logical_and(y_pred[:,j]==1, y_true[:,j]==1).sum()
        fp = np.logical_and(y_pred[:,j]==1, y_true[:,j]==0).sum()
        fn = np.logical_and(y_pred[:,j]==0, y_true[:,j]==1).sum()
        p = tp / (tp+fp+eps); r = tp / (tp+fn+eps)
        f1s.append(2*p*r/(p+r+eps))
    return float(np.mean(f1s))

# ==========================
# Split automático quando val.csv é unlabeled
# ==========================
def stratified_split_single(labels_idx: List[int], val_ratio: float, seed: int):
    rng = random.Random(seed)
    from collections import defaultdict
    by_cls = defaultdict(list)
    for i, y in enumerate(labels_idx):
        by_cls[y].append(i)
    train_idx, val_idx = [], []
    for y, ids in by_cls.items():
        rng.shuffle(ids)
        k = max(1, int(len(ids) * val_ratio))
        val_idx.extend(ids[:k])
        train_idx.extend(ids[k:])
    return train_idx, val_idx

def random_split_indices(n: int, val_ratio: float, seed: int):
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    k = max(1, int(n * val_ratio))
    return idx[k:].tolist(), idx[:k].tolist()

# ==========================
# Treino / Validação (Task 1)
# ==========================
def train_task1(cfg: Config, device: torch.device) -> Path:
    root = Path(cfg.data_root)
    train_ds_full = TaskDataset(root, cfg.train_csv, want_train=True)

    tr_tf, te_tf = make_transforms(cfg.img_size)

    # SEMPRE usa holdout 70/30 do train.csv
    print(f"[INFO] Usando holdout de {int(cfg.val_ratio*100)}% do train para validação (ignorando val.csv declarado).")

    if train_ds_full.schema.mode == "single":
        labels_idx = [train_ds_full.label_to_idx[int(y[0])] for _, y in train_ds_full.samples]
        tr_idx, va_idx = stratified_split_single(labels_idx, cfg.val_ratio, cfg.seed)
        out_dim = train_ds_full.num_classes
        loss_fn = nn.CrossEntropyLoss()
        schema_mode = "single"
    else:
        n = len(train_ds_full)
        tr_idx, va_idx = random_split_indices(n, cfg.val_ratio, cfg.seed)
        out_dim = train_ds_full.num_attrs
        loss_fn = nn.BCEWithLogitsLoss()
        schema_mode = "multi"

    # aplica transforms e “injeção” de __getitem__ para Subset herdar o transform
    from torch.utils.data import Subset
    train_ds_full.set_transform(tr_tf)
    train_ds = Subset(train_ds_full, tr_idx)
    train_ds.__class__.__getitem__ = lambda self, i, _orig=train_ds_full: _orig[self.indices[i]]  # type: ignore

    train_ds_full.set_transform(te_tf)
    val_ds   = Subset(train_ds_full, va_idx)
    val_ds.__class__.__getitem__   = lambda self, i, _orig=train_ds_full: _orig[self.indices[i]]  # type: ignore

    # modelo/otimizadores
    model = MobileNetHead(out_dim=out_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type=="cuda")

    train_ld = DataLoader(train_ds, cfg.batch_size, True,  num_workers=cfg.num_workers, pin_memory=(device.type=="cuda"))
    val_ld   = DataLoader(val_ds,   cfg.batch_size, False, num_workers=cfg.num_workers, pin_memory=(device.type=="cuda"))
    root = Path(cfg.data_root)
    train_ds_full = TaskDataset(root, cfg.train_csv, want_train=True)
    val_ds_declared = TaskDataset(root, cfg.val1_csv,   want_train=False)

    tr_tf, te_tf = make_transforms(cfg.img_size)

    # Se o val.csv for UNLABELED, criamos um holdout do train
    use_holdout = (val_ds_declared.schema.mode == "unlabeled")
    if use_holdout:
        print(f"[INFO] {cfg.val1_csv} não possui rótulos. Criando holdout {int(cfg.val_ratio*100)}% do train para validação.")
        if train_ds_full.schema.mode == "single":
            # constrói vetor de rótulos (índices já normalizados)
            labels_idx = [train_ds_full.label_to_idx[int(y[0])] for _, y in train_ds_full.samples]
            tr_idx, va_idx = stratified_split_single(labels_idx, cfg.val_ratio, cfg.seed)
        else:
            n = len(train_ds_full)
            tr_idx, va_idx = random_split_indices(n, cfg.val_ratio, cfg.seed)

        train_ds = Subset(train_ds_full, tr_idx)
        val_ds   = Subset(train_ds_full, va_idx)

        # precisamos aplicar transform manual em Subset:
        train_ds_full.set_transform(tr_tf)
        train_tf_applier = train_ds_full.transform
        def _getitem_subset_train(i, _orig=train_ds_full):
            return _orig[i]
        train_ds.__class__.__getitem__ = lambda self, i: _getitem_subset_train(self.indices[i])  # type: ignore

        train_ds_full.set_transform(te_tf)
        def _getitem_subset_val(i, _orig=train_ds_full):
            return _orig[i]
        val_ds.__class__.__getitem__ = lambda self, i: _getitem_subset_val(self.indices[i])  # type: ignore

        # configura dimensões de saída
        if train_ds_full.schema.mode == "single":
            out_dim = train_ds_full.num_classes
            loss_fn = nn.CrossEntropyLoss()
        else:
            out_dim = train_ds_full.num_attrs
            loss_fn = nn.BCEWithLogitsLoss()
        schema_mode = train_ds_full.schema.mode

    else:
        # usar train e o val declarado
        train_ds_full.set_transform(tr_tf)
        val_ds_declared.set_transform(te_tf)
        train_ds = train_ds_full
        val_ds   = val_ds_declared
        schema_mode = train_ds_full.schema.mode
        loss_fn = nn.CrossEntropyLoss() if schema_mode == "single" else nn.BCEWithLogitsLoss()
        out_dim = (train_ds_full.num_classes if schema_mode == "single" else train_ds_full.num_attrs)

    # Modelo e otimizadores
    model = MobileNetHead(out_dim=out_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type=="cuda")

    train_ld = DataLoader(train_ds, cfg.batch_size, True,  num_workers=cfg.num_workers, pin_memory=(device.type=="cuda"))
    val_ld   = DataLoader(val_ds,   cfg.batch_size, False, num_workers=cfg.num_workers, pin_memory=(device.type=="cuda"))

    os.makedirs(cfg.save_dir, exist_ok=True)
    save_path = Path(cfg.save_dir) / cfg.model_file

    def eval_single():
        def topk(logits, target, k):
            _, pred = logits.topk(k, 1, True, True)
            return pred.eq(target.view(-1,1)).any(1).float().mean().item()*100.0
        model.eval(); c1=c5=0; m=0
        with torch.no_grad():
            for x,y in val_ld:
                x,y=x.to(device),y.to(device)
                logits=model(x)
                b=x.size(0); c1+=topk(logits,y,1)*b/100.0; c5+=topk(logits,y,5)*b/100.0; m+=b
        return (c1/m)*100.0, (c5/m)*100.0

    def eval_multi():
        model.eval(); all_logits=[]; all_targets=[]
        with torch.no_grad():
            for x,y,*_rest in val_ld:
                x=x.to(device); logits=model(x)
                all_logits.append(logits.cpu().numpy()); all_targets.append(y.numpy())
        y_score = np.vstack(all_logits); y_true = np.vstack(all_targets)
        APs = [average_precision(y_true[:,j], y_score[:,j]) for j in range(y_true.shape[1])]
        mAP = 100.0*float(np.mean(APs))
        # sigmoid para F1
        proba = 1.0/(1.0+np.exp(-y_score))
        F1  = 100.0*f1_macro(y_true, proba, thr=0.5)
        return mAP, F1

    best_metric = -1.0
    for ep in range(1, cfg.epochs+1):
        model.train(); running=0; n=0; t0=time.time()
        for batch in train_ld:
            optimizer.zero_grad(set_to_none=True)
            if schema_mode == "single":
                x,y = batch
                x,y = x.to(device), y.to(device)
                with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                    logits = model(x); loss = loss_fn(logits,y)
            else:
                x,y,*_ = batch
                x,y = x.to(device), y.to(device)
                with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                    logits = model(x); loss = loss_fn(logits,y)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            running += loss.item()*x.size(0); n += x.size(0)

        # validação SEMPRE existe aqui (holdout ou val rotulado)
        if schema_mode == "single":
            top1, top5 = eval_single()
            msg_val=f"val@1={top1:.2f}% val@5={top5:.2f}%"
            current_metric = top1
        else:
            mAP, F1 = eval_multi()
            msg_val=f"val mAP={mAP:.2f}%  F1={F1:.2f}%"
            current_metric = mAP

        print(f"[{ep:03d}] loss={running/max(n,1):.4f} | {msg_val} | {time.time()-t0:.1f}s")

        if current_metric > best_metric:
            best_metric = current_metric
            payload = {"model": model.state_dict(), "schema": schema_mode}
            if schema_mode == "single": payload["num_classes"] = (train_ds_full.num_classes if hasattr(train_ds_full,'num_classes') else out_dim)
            else: payload["num_attrs"] = (train_ds_full.num_attrs if hasattr(train_ds_full,'num_attrs') else out_dim)
            torch.save(payload, save_path)
            print(f"  -> melhor até agora salvo: {save_path}")

    print("Treino concluído.")
    return save_path

# ==========================
# Task 2 (retrieval) — opcional
# ==========================
class CSVEmbDataset(Dataset):
    def __init__(self, root: Path, csv_rel: str):
        csv_path = resolve_csv(root, csv_rel, try_train_subdir=False)
        import csv as _csv
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = _csv.DictReader(f); first = next(iter(reader))
            keys = [k for k in first.keys() if isinstance(k,str)]
            kmap = {k.lower():k for k in keys}
            pid_col = next((kmap[k] for k in ["person_id","label","pid","id"] if k in kmap), None)
            if pid_col is None:
                raise ValueError(f"{csv_path} não tem coluna de person_id/label; não dá para avaliar Task 2.")
            img_col = next((kmap[k] for k in ["img_path","image","path","filepath","file","relpath"] if k in kmap), keys[0])
            f.seek(0); reader = _csv.DictReader(f)
            self.samples = [(r[img_col], int(r[pid_col])) for r in reader if r.get(img_col) is not None]
        self.root=root; self.tf=None
    def set_transform(self,t): self.tf=t
    def __len__(self): return len(self.samples)
    def __getitem__(self,i):
        rel,pid=self.samples[i]
        img=safe_open_image(self.root/rel)
        if self.tf: img=self.tf(img)
        return img, pid, rel

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a/(np.linalg.norm(a,axis=1,keepdims=True)+1e-12)
    b = b/(np.linalg.norm(b,axis=1,keepdims=True)+1e-12)
    return a@b.T

def retrieval_metrics(S,q_lbl,g_lbl,ks=(1,5,10)):
    order=np.argsort(-S,axis=1); topk={}
    for k in ks:
        match=sum(np.any(g_lbl[order[i,:k]]==q_lbl[i]) for i in range(S.shape[0]))
        topk[f"top{k}"]=100.0*match/max(S.shape[0],1)
    APs=[]
    for i in range(S.shape[0]):
        rank=order[i]; rel=(g_lbl[rank]==q_lbl[i]).astype(np.int32)
        if rel.sum()==0: APs.append(0.0); continue
        cum=np.cumsum(rel); prec=cum/(np.arange(len(rel))+1)
        APs.append(float((prec*rel).sum()/rel.sum()))
    return topk, 100.0*float(np.mean(APs))

def evaluate_task2(cfg: Config, ckpt_path: Path, device: torch.device):
    if not ckpt_path.exists():
        print(f"[AVISO] checkpoint não encontrado: {ckpt_path}. pulando Task 2."); return
    ckpt=torch.load(ckpt_path,map_location=device)
    if ckpt.get("schema")!="single":
        print("[INFO] Modelo treinado para atributos (multi-label). Task 2 requer IDs — pulando."); return

    model=MobileNetHead(out_dim=ckpt["num_classes"]); model.load_state_dict(ckpt["model"]); model.to(device)
    _,te=make_transforms(Config.img_size)
    root=Path(cfg.data_root)
    try:
        q_ds=CSVEmbDataset(root,cfg.val2_q_csv); g_ds=CSVEmbDataset(root,cfg.val2_g_csv)
    except Exception as e:
        print(f"[INFO] {e} Pulando Task 2."); return
    q_ds.set_transform(te); g_ds.set_transform(te)

    q_ld=DataLoader(q_ds,cfg.batch_size,False,num_workers=cfg.num_workers,pin_memory=(device.type=="cuda"))
    g_ld=DataLoader(g_ds,cfg.batch_size,False,num_workers=cfg.num_workers,pin_memory=(device.type=="cuda"))

    model.eval(); Q=[]; yq=[]; G=[]; yg=[]
    with torch.no_grad():
        for x,pid,_ in q_ld:
            Q.append(model(x.to(device),return_embedding=True).cpu().numpy()); yq+=list(map(int,pid))
        for x,pid,_ in g_ld:
            G.append(model(x.to(device),return_embedding=True).cpu().numpy()); yg+=list(map(int,pid))
    S=cosine_sim(np.vstack(Q), np.vstack(G))
    topk,mAP=retrieval_metrics(S,np.array(yq),np.array(yg))
    print("Task 2 - Retrieval")
    for k,v in topk.items(): print(f"  {k}: {v:.2f}%")
    print(f"  mAP: {mAP:.2f}%")

# ==========================
# Main
# ==========================
def main():
    import argparse
    parser=argparse.ArgumentParser("MobileNetV2 • Market1501/UPAR (IDs • Atributos)")
    parser.add_argument("--data-root", type=str, default=Config.data_root)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--img-size", type=int, default=Config.img_size)
    parser.add_argument("--save-dir", type=str, default=Config.save_dir)
    parser.add_argument("--num-workers", type=int, default=Config.num_workers)
    parser.add_argument("--no-amp", action="store_true")
    args=parser.parse_args()

    cfg=Config(data_root=args.data_root, epochs=args.epochs, batch_size=args.batch_size,
               img_size=args.img_size, save_dir=args.save_dir, num_workers=args.num_workers,
               amp=not args.no_amp)

    root=Path(cfg.data_root)
    if not root.exists():
        raise FileNotFoundError(f"data_root '{root}' não encontrado. Ex.: --data-root data")

    set_seed(cfg.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    ckpt=train_task1(cfg, device)
    evaluate_task2(cfg, ckpt, device)

if __name__=="__main__":
    main()
