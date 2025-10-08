# mobileNet-multiModal.py
# -*- coding: utf-8 -*-

# EXEMPLO DE USO:
# python mobileNet-multiModal.py --data-root "caminho/para/pasta_de_imagens" --train-csv "caminho/completo/para/arquivo.csv"

import os, csv, time, random, io
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from PIL import Image, UnidentifiedImageError

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
import argparse

# --- NOVA IMPORTAÇÃO ---
from tqdm import tqdm

# ==========================
# Config
# ==========================
@dataclass
class Config:
    data_root: str = "/home/gdaudt/Área de trabalho/Projeto-Transformador-II/data_agumentation/"
    train_csv: str = "/home/gdaudt/Área de trabalho/Projeto-Transformador-II/data_agumentation/data-balanced-50-50-sample/dataset_balanceado_amostra.csv"
    img_size: int = 180
    batch_size: int = 128
    epochs: int = 1
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 6
    seed: int = 42
    amp: bool = True
    test_ratio: float = 0.20
    val_ratio: float = 0.30
    save_dir: str = "mobileNetMultiModal"
    model_prefix: str = "mobilenet_v2_final"

# ==========================
# Features selecionadas
# ==========================
SELECTED_ATTRS = [
    "LowerBody-Color-Grey", "LowerBody-Color-Orange", "LowerBody-Color-Pink",
    "LowerBody-Color-Purple", "LowerBody-Color-Red", "LowerBody-Color-White",
    "LowerBody-Color-Yellow", "LowerBody-Color-Other", "LowerBody-Type-Trousers&Shorts",
    "LowerBody-Type-Skirt&Dress", "Accessory-Backpack", "Accessory-Bag",
    "Accessory-Glasses-Normal", "Accessory-Glasses-Sun", "Accessory-Hat"
]

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

# ==========================
# Leitura de CSV
# ==========================
class Schema:
    def __init__(self, mode: str, img_col: str, label_col: Optional[str], attr_cols: List[str]):
        self.mode = mode; self.img_col = img_col
        self.label_col = label_col; self.attr_cols = attr_cols

def read_csv_rows(csv_path: Path) -> Tuple[Schema, List[Tuple[str, np.ndarray]]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Arquivo CSV não encontrado no caminho literal: {csv_path}")

    with open(csv_path, 'r', encoding='utf-8') as f:
        raw_lines = [ln.rstrip("\n") for ln in f if ln.strip() != ""]
    if not raw_lines: raise ValueError(f"{csv_path} está vazio.")
    
    if raw_lines[0].lstrip().startswith("#"):
        attr_names = [s.strip() for s in raw_lines[0].lstrip("#").strip().split(",")]
        selected_indices = [i for i, name in enumerate(attr_names) if name in SELECTED_ATTRS]
        final_attr_names = [attr_names[i] for i in selected_indices]
        if len(final_attr_names) != len(SELECTED_ATTRS):
            print(f"[AVISO] {len(SELECTED_ATTRS) - len(final_attr_names)} atributos de SELECTED_ATTRS não foram encontrados no CSV.")
        schema = Schema("multi", "img_path", None, final_attr_names)
        rows = []
        for ln in raw_lines[1:]:
            if ln.lstrip().startswith("#"): continue
            parts = [p.strip() for p in ln.split(",")]
            img_rel = parts[0]
            vals = parts[1:1+len(attr_names)]
            y = np.array([int(float(vals[i])) if vals[i] != "" else 0 for i in selected_indices], dtype=np.float32)
            rows.append((img_rel, y))
        return schema, rows

    try:
        reader = csv.DictReader(raw_lines)
        header = reader.fieldnames
        if not header:
            raise ValueError("Cabeçalho do CSV está vazio ou não foi encontrado.")

        image_path_col = header[0]
        final_attr_names = [h for h in header if h in SELECTED_ATTRS]
        if not final_attr_names:
             raise ValueError("Nenhum dos atributos em SELECTED_ATTRS foi encontrado no cabeçalho do CSV.")

        schema = Schema("multi", image_path_col, None, final_attr_names)
        
        rows = []
        for row in reader:
            img_rel = row[image_path_col]
            y = np.array([int(float(row.get(attr, 0) or 0)) for attr in final_attr_names], dtype=np.float32)
            rows.append((img_rel, y))
        return schema, rows

    except Exception as e:
        print(f"Erro ao processar o CSV: {e}")
        raise ValueError("Não foi possível interpretar o arquivo CSV. Verifique o formato.")

# ==========================
# Datasets
# ==========================
class TaskDataset(Dataset):
    def __init__(self, csv_path: Path, image_root: Path):
        self.schema, self.samples = read_csv_rows(csv_path)
        self.image_root = image_root
        self.transform = None
        self.num_attrs = len(self.schema.attr_cols)
        print(f"[INFO] Dataset carregado com {self.num_attrs} atributos de '{csv_path.name}'.")

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        rel, y = self.samples[i]
        img_path = self.image_root / rel.replace("\\", "/")
        img = safe_open_image(img_path)
        if self.transform: img = self.transform(img)
        return img, torch.tensor(y, dtype=torch.float32), rel

# ==========================
# Modelo
# ==========================
class MobileNetHead(nn.Module):
    def __init__(self, out_dim: int, pretrained: bool = True):
        super().__init__()
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.mobilenet_v2(weights=weights)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        in_feats = 1280
        self.bn = nn.BatchNorm1d(in_feats)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(in_feats, out_dim)
    def forward(self, x):
        x = self.features(x); x = self.pool(x).flatten(1)
        x = self.bn(x); x = self.dropout(x)
        return self.fc(x)

# ==========================
# Transforms
# ==========================
def make_transforms(size):
    train_tf = transforms.Compose([
        transforms.Resize((size, size)), transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(0.2,0.2,0.2,0.1), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),])
    test_tf = transforms.Compose([
        transforms.Resize((size, size)), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),])
    return train_tf, test_tf

# ==========================
# Métricas e Avaliação
# ==========================
def average_precision(y_true, y_score):
    order = np.argsort(-y_score)
    y_true = y_true[order]
    if y_true.sum() == 0: return 0.0
    cumsum = np.cumsum(y_true)
    precision = cumsum / (np.arange(len(y_true)) + 1)
    return float((precision * y_true).sum() / y_true.sum())

def f1_macro(y_true, y_score, thr=0.5):
    y_pred = (y_score >= thr).astype(np.int32); eps = 1e-9; f1s = []
    for j in range(y_true.shape[1]):
        tp = np.logical_and(y_pred[:,j]==1, y_true[:,j]==1).sum()
        fp = np.logical_and(y_pred[:,j]==1, y_true[:,j]==0).sum()
        fn = np.logical_and(y_pred[:,j]==0, y_true[:,j]==1).sum()
        p = tp/(tp+fp+eps); r = tp/(tp+fn+eps)
        f1s.append(2*p*r/(p+r+eps))
    return float(np.mean(f1s))

def evaluate_model(model, loader, device, text_header=""):
    model.eval()
    all_logits, all_targets = [], []
    # --- LOOP DE AVALIAÇÃO COM BARRA DE PROGRESSO ---
    pbar_eval = tqdm(loader, desc=f"Avaliando {text_header}", leave=False, colour="cyan")
    with torch.no_grad():
        for x,y,_ in pbar_eval:
            x=x.to(device); logits=model(x)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(y.numpy())
    if not all_logits: return 0.0, 0.0
    y_score=np.vstack(all_logits); y_true=np.vstack(all_targets)
    APs=[average_precision(y_true[:,j], y_score[:,j]) for j in range(y_true.shape[1])]
    mAP=100.0*float(np.mean(APs))
    proba=1.0/(1.0+np.exp(-y_score))
    F1=100.0*f1_macro(y_true, proba, thr=0.5)
    if text_header:
        print(f"{text_header} mAP={mAP:.2f}%  F1={F1:.2f}%")
    return mAP, F1

# ==========================
# Seção de Predição
# ==========================
def interpret_binary_predictions(probabilities: np.ndarray, attribute_names: List[str], threshold: float = 0.5) -> Dict[str, Any]:
    preds_map = {name: prob for name, prob in zip(attribute_names, probabilities)}
    results = {}; warnings = []
    has_glasses_normal = preds_map.get('Accessory-Glasses-Normal', 0) > threshold
    has_glasses_sun = preds_map.get('Accessory-Glasses-Sun', 0) > threshold
    results['usa_oculos'] = has_glasses_normal or has_glasses_sun
    has_backpack = preds_map.get('Accessory-Backpack', 0) > threshold
    has_bag = preds_map.get('Accessory-Bag', 0) > threshold
    results['carrega_bolsa_mochila'] = has_backpack or has_bag
    results['usa_chapeu'] = preds_map.get('Accessory-Hat', 0) > threshold
    lower_body_colors = [name for name in attribute_names if 'LowerBody-Color' in name]
    if lower_body_colors:
        results['veste_parte_de_baixo'] = any(preds_map.get(name, 0) > threshold for name in lower_body_colors)
    else:
        results['veste_parte_de_baixo'] = "Não foi possível determinar"
        warnings.append("Nenhuma feature 'LowerBody-Color-*' em SELECTED_ATTRS.")
    if not any('UpperBody-Color' in name for name in attribute_names):
        warnings.append("Não é possível responder sobre 'roupa de cima'.")
    if 'LowerBody-Length-Short' not in attribute_names:
        warnings.append("Não é possível responder se 'roupa de baixo é curta'.")
    if 'Gender-Female' not in attribute_names:
        warnings.append("Não é possível responder sobre 'gênero'.")
    if not any(age in name for age in ['Age-Young', 'Age-Adult', 'Age-Old'] for name in attribute_names):
        warnings.append("Não é possível responder sobre 'idade'.")
    if warnings: results['avisos'] = warnings
    return results

def predict_and_interpret(image_path: str, cfg: Config, device: torch.device):
    save_path = Path(cfg.save_dir) / f"{cfg.model_prefix}.pth"
    if not save_path.exists():
        print(f"Erro: Modelo treinado não encontrado em '{save_path}'."); return
    ckpt = torch.load(save_path, map_location=device)
    attr_names = ckpt.get("attrs", SELECTED_ATTRS)
    model = MobileNetHead(out_dim=len(attr_names)).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    _, test_tf = make_transforms(cfg.img_size)
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = test_tf(image).unsqueeze(0).to(device)
    except FileNotFoundError:
        print(f"Erro: Arquivo de imagem não encontrado em '{image_path}'"); return
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.sigmoid(logits).cpu().numpy().flatten()
    interpreted_results = interpret_binary_predictions(probabilities, attr_names)
    print("\n--- Resultado da Predição Binária ---")
    for key, value in interpreted_results.items():
        if key == 'avisos':
            print("\n[AVISOS IMPORTANTES]")
            for aviso in value: print(f"- {aviso}")
        else: print(f"{key}: {value}")
    print("\n--- Probabilidades Detalhadas dos Atributos ---")
    for name, prob in zip(attr_names, probabilities):
        print(f"- {name}: {prob:.2f}")

# ==========================
# Main
# ==========================
def main():
    parser=argparse.ArgumentParser("MobileNetV2 para Atributos com Caminho CSV Literal")
    parser.add_argument("--data-root", type=str, default=Config.data_root,
                        help="Caminho para o diretório raiz onde as pastas de imagens estão localizadas.")
    parser.add_argument("--train-csv", type=str, default=Config.train_csv,
                        help="Caminho completo e literal para o arquivo .csv de anotações.")
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--img-size", type=int, default=Config.img_size)
    parser.add_argument("--save-dir", type=str, default=Config.save_dir)
    parser.add_argument("--num-workers", type=int, default=Config.num_workers)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--predict", type=str, default=None, help="Caminho para uma imagem para fazer a predição.")
    args=parser.parse_args()

    cfg=Config(data_root=args.data_root, train_csv=args.train_csv, epochs=args.epochs,
               batch_size=args.batch_size, img_size=args.img_size, save_dir=args.save_dir,
               num_workers=args.num_workers, amp=not args.no_amp)
    
    set_seed(cfg.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    if args.predict:
        predict_and_interpret(args.predict, cfg, device)
        return

    image_root_path = Path(cfg.data_root)
    csv_literal_path = Path(cfg.train_csv)

    if not image_root_path.is_dir():
        raise FileNotFoundError(f"O diretório raiz de imagens (--data-root) não foi encontrado: '{image_root_path}'")
    
    full_dataset = TaskDataset(csv_path=csv_literal_path, image_root=image_root_path)
    tr_tf, te_tf = make_transforms(cfg.img_size)
    
    original_indices = [i for i, (p,_) in enumerate(full_dataset.samples) if "_agu" not in Path(p).name]
    augmented_indices = [i for i, (p,_) in enumerate(full_dataset.samples) if "_agu" in Path(p).name]
    random.shuffle(original_indices)
    test_split_idx = int(len(original_indices) * cfg.test_ratio)
    test_indices = original_indices[:test_split_idx]
    train_val_pool_indices = original_indices[test_split_idx:] + augmented_indices
    random.shuffle(train_val_pool_indices)
    val_split_idx = int(len(train_val_pool_indices) * cfg.val_ratio)
    val_indices = train_val_pool_indices[:val_split_idx]
    train_indices = train_val_pool_indices[val_split_idx:]

    print(f"\nDataset Total: {len(full_dataset)} imagens")
    print(f" -> Treino: {len(train_indices)} | Validação: {len(val_indices)} | Teste: {len(test_indices)}\n")

    class TransformedSubset(Dataset):
        def __init__(self, subset, transform, image_root):
            self.subset=subset; self.transform=transform; self.image_root=image_root
        def __getitem__(self, index):
            _, y, rel = self.subset[index]
            img=safe_open_image(self.image_root / rel.replace("\\", "/"))
            return self.transform(img), y, rel
        def __len__(self): return len(self.subset)

    train_ds=TransformedSubset(Subset(full_dataset, train_indices), tr_tf, image_root_path)
    val_ds=TransformedSubset(Subset(full_dataset, val_indices), te_tf, image_root_path)
    test_ds=TransformedSubset(Subset(full_dataset, test_indices), te_tf, image_root_path)
    
    train_ld=DataLoader(train_ds, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_ld=DataLoader(val_ds, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    test_ld=DataLoader(test_ds, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    out_dim = full_dataset.num_attrs
    model = MobileNetHead(out_dim=out_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    
    scaler = torch.amp.GradScaler(enabled=cfg.amp and device.type=="cuda")
    
    os.makedirs(cfg.save_dir, exist_ok=True)
    save_path = Path(cfg.save_dir) / f"{cfg.model_prefix}.pth"

    best_metric = -1.0
    for ep in range(1, cfg.epochs + 1):
        model.train(); running_loss, n, t0 = 0, 0, time.time()
        
        # --- LOOP DE TREINO COM BARRA DE PROGRESSO ---
        pbar = tqdm(train_ld, desc=f"Época {ep}/{cfg.epochs}", leave=False, colour="green")
        for x, y, _ in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.autocast(device.type, enabled=scaler.is_enabled()):
                logits = model(x); loss = loss_fn(logits, y)
                
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            running_loss += loss.item() * x.size(0); n += x.size(0)
            
            # Atualiza a informação de loss na barra de progresso
            pbar.set_postfix(loss=f"{(running_loss / n):.4f}")

        # --- AVALIAÇÃO ---
        val_mAP, _ = evaluate_model(model, val_ld, device, text_header="Validação")
        
        # O print original agora serve como um resumo no final de cada época
        print(f"Fim da Época [{ep:02d}/{cfg.epochs}] -> Perda (loss): {(running_loss/n):.4f} | mAP Validação: {val_mAP:.2f}% | Tempo: {time.time()-t0:.1f}s")
        
        if val_mAP > best_metric:
            best_metric = val_mAP
            payload = {"model": model.state_dict(), "attrs": full_dataset.schema.attr_cols}
            torch.save(payload, save_path)
            print(f"  -> Melhor modelo salvo em: {save_path} (mAP: {best_metric:.2f}%)")

    print("\nTreino concluído.")
    print("\nIniciando avaliação final no conjunto de teste...")
    if save_path.exists():
        ckpt = torch.load(save_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print("Melhor modelo carregado para o teste final.")
        evaluate_model(model, test_ld, device, text_header="[RESULTADO DO TESTE FINAL]")
    else:
        print("[AVISO] Nenhum checkpoint foi salvo.")

if __name__ == "__main__":
    main()