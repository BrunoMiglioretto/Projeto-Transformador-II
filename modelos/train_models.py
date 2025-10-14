import os
import pandas as pd
import numpy as np
from collections import defaultdict
from PIL import Image
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

# Ignorar avisos de convergência do Scikit-learn para relatórios mais limpos
warnings.filterwarnings("ignore", category=UserWarning)

# ======================================================================================
# CLASSE PARA REDIRECIONAR A SAÍDA DO TERMINAL
# ======================================================================================
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

# ======================================================================================
# 1. CLASSE DE CONFIGURAÇÃO
# ======================================================================================
class Config:
    """
    Classe de configuração para todos os hiperparâmetros e caminhos.
    """
    # Caminhos
    DATA_ROOT: str = "/home/gdaudt/Área de trabalho/generated_balanced_datasets/dataset2_geometric_affine_perspective"
    TRAIN_CSV: str = os.path.join(DATA_ROOT, "dataset_labels.csv")
    OUTPUT_DIR: str = "./Dataset2_geometric_affine_perspective-output"
    
    DATASET_NAME: str = os.path.basename(DATA_ROOT)

    # Parâmetros de Treinamento
    MODELS_TO_TRAIN: list = ['MobileNetV2', 'EfficientNet-B0']
    IMG_SIZE: int = 200
    BATCH_SIZE: int = 32
    EPOCHS: int = 6
    LEARNING_RATE: float = 1e-4

    # Parâmetros de Divisão do Dataset
    RANDOM_STATE: int = 42
    DATA_USAGE_PERCENT: float = 0.40
    TRAIN_SPLIT_SIZE: float = 0.60
    VALIDATION_SPLIT_SIZE: float = 0.20
    TEST_SPLIT_SIZE: float = 0.20

    # Dispositivo
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# ======================================================================================
# 2. DATASET E DATALOADER
# ======================================================================================
class AttributeDataset(Dataset):
    def __init__(self, df, data_root, transform=None):
        self.df = df
        self.data_root = data_root
        self.transform = transform
        self.image_paths = df['image_path'].values
        self.labels = df['label'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_root, self.image_paths[idx])
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: File not found {img_path}. Skipping.")
            return None, None
            
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label

def get_transforms(img_size):
    return {
        'train': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    
def collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch:
        return torch.tensor([]), torch.tensor([])
    return torch.utils.data.dataloader.default_collate(batch)

# ======================================================================================
# 3. MODELOS
# ======================================================================================
def get_model(model_name: str, num_classes: int, pretrained: bool = True):
    weights = 'DEFAULT' if pretrained else None
    if model_name == 'ResNet50':
        model = models.resnet50(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'MobileNetV2':
        model = models.mobilenet_v2(weights=weights)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'EfficientNet-B0':
        model = models.efficientnet_b0(weights=weights)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError(f"Modelo '{model_name}' não suportado.")
    return model

# ======================================================================================
# 4. FUNÇÕES DE TREINAMENTO E AVALIAÇÃO
# ======================================================================================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct_predictions, total_samples = 0.0, 0, 0
    for inputs, labels in tqdm(dataloader, desc="Treinando"):
        if inputs.nelement() == 0: continue
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        correct_predictions += torch.sum(preds == labels.data)
        total_samples += len(inputs)
    epoch_loss = running_loss / total_samples if total_samples > 0 else 0
    epoch_acc = correct_predictions.double() / total_samples if total_samples > 0 else 0
    return epoch_loss, epoch_acc.item() if isinstance(epoch_acc, torch.Tensor) else epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct_predictions, total_samples = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validando/Testando"):
            if inputs.nelement() == 0: continue
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            correct_predictions += torch.sum(preds == labels.data)
            total_samples += len(inputs)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    epoch_loss = running_loss / total_samples if total_samples > 0 else 0
    epoch_acc = correct_predictions.double() / total_samples if total_samples > 0 else 0
    return epoch_loss, epoch_acc.item() if isinstance(epoch_acc, torch.Tensor) else epoch_acc, all_preds, all_labels

# ======================================================================================
# 5. UTILITÁRIOS (PLOTS E SALVAMENTO)
# ======================================================================================
def save_accuracy_plot(train_acc, val_acc, save_path, title='Acurácia por Época'):
    plt.figure(figsize=(10, 5))
    plt.plot(train_acc, label='Acurácia de Treino')
    plt.plot(val_acc, label='Acurácia de Validação')
    plt.title(title)
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def save_confusion_matrix(y_true, y_pred, class_names, save_path, title='Matriz de Confusão'):
    if not y_true or not y_pred: return
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('Rótulo Verdadeiro')
    plt.xlabel('Rótulo Previsto')
    plt.savefig(save_path)
    plt.close()

def save_average_accuracy_plot(overall_results, save_path):
    plt.figure(figsize=(12, 7))
    for model_name, data in overall_results.items():
        plt.plot(data['avg_train_acc'], label=f'{model_name} - Treino Médio', linestyle='--')
        plt.plot(data['avg_val_acc'], label=f'{model_name} - Validação Média', linewidth=2.5)
    plt.title('Acurácia Média de Todos Atributos por Modelo')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia Média')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"\n[INFO] Gráfico de acurácia média salvo em: {save_path}")

# ======================================================================================
# 6. FUNÇÃO PRINCIPAL
# ======================================================================================
def main():
    assert np.isclose(Config.TRAIN_SPLIT_SIZE + Config.VALIDATION_SPLIT_SIZE + Config.TEST_SPLIT_SIZE, 1.0), \
        "A soma das porcentagens de divisão (TRAIN, VALIDATION, TEST) deve ser 1.0"

    print(f"[INFO] Usando dispositivo: {Config.DEVICE}")
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    df = pd.read_csv(Config.TRAIN_CSV)
    
    print("\n" + "="*80)
    print("[INFO] Preparando e dividindo o dataset...")
    shuffled_df = df.sample(frac=1, random_state=Config.RANDOM_STATE).reset_index(drop=True)
    num_rows_to_use = int(len(shuffled_df) * Config.DATA_USAGE_PERCENT)
    subset_df = shuffled_df.iloc[:num_rows_to_use]
    print(f"[INFO] Total de amostras após shuffle: {len(shuffled_df)}")
    print(f"[INFO] Usando {Config.DATA_USAGE_PERCENT*100:.1f}% do dataset: {len(subset_df)} amostras.")
    
    train_end_idx = int(len(subset_df) * Config.TRAIN_SPLIT_SIZE)
    validation_end_idx = train_end_idx + int(len(subset_df) * Config.VALIDATION_SPLIT_SIZE)
    
    train_df = subset_df.iloc[:train_end_idx]
    val_df = subset_df.iloc[train_end_idx:validation_end_idx]
    test_df = subset_df.iloc[validation_end_idx:]
    
    print(f"[INFO] Divisão final -> Treino: {len(train_df)} | Validação: {len(val_df)} | Teste: {len(test_df)}")
    print("="*80)
    
    feature_cols = [col for col in df.columns if col != 'image_path']
    feature_groups = {col: [col] for col in feature_cols}
    print(f"[INFO] Atributos encontrados para treinamento individual: {list(feature_groups.keys())}")

    overall_results = {}
    all_feature_performances = []
    
    # <--- ALTERAÇÃO: Lista para armazenar todos os dados de resumo para o relatório geral --->
    general_summary_list = []

    for model_name in Config.MODELS_TO_TRAIN:
        model_output_dir = os.path.join(Config.OUTPUT_DIR, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        log_file_path = os.path.join(model_output_dir, 'terminal_output.txt')
        original_stdout = sys.stdout
        sys.stdout = Logger(log_file_path)

        try:
            print("\n" + "="*80)
            print(f"[INFO] Iniciando ciclo para o modelo: {model_name}")
            print(f"[INFO] A saída está sendo salva em: {log_file_path}")
            print("="*80)

            model_train_acc_histories, model_val_acc_histories = [], []
            model_all_true_labels, model_all_pred_labels = [], []
            
            model_summary_data = []

            for feature, columns in feature_groups.items():
                print(f"\n--- Treinando para o atributo: {feature} ---")
                
                feature_train_df = train_df[['image_path'] + columns].copy()
                feature_val_df = val_df[['image_path'] + columns].copy()
                feature_test_df = test_df[['image_path'] + columns].copy()

                feature_train_df.rename(columns={columns[0]: 'label'}, inplace=True)
                feature_val_df.rename(columns={columns[0]: 'label'}, inplace=True)
                feature_test_df.rename(columns={columns[0]: 'label'}, inplace=True)

                class_names = [f'Not_{feature}', feature]
                num_classes = 2

                transforms_dict = get_transforms(Config.IMG_SIZE)
                train_dataset = AttributeDataset(feature_train_df, Config.DATA_ROOT, transforms_dict['train'])
                val_dataset = AttributeDataset(feature_val_df, Config.DATA_ROOT, transforms_dict['val'])
                test_dataset = AttributeDataset(feature_test_df, Config.DATA_ROOT, transforms_dict['val'])

                train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=collate_fn)
                val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=collate_fn)
                test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=collate_fn)

                model = get_model(model_name, num_classes).to(Config.DEVICE)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

                train_acc_history, val_acc_history = [], []
                best_val_acc = 0.0
                
                feature_output_dir = os.path.join(model_output_dir, feature.replace('&','_'))
                os.makedirs(feature_output_dir, exist_ok=True)

                for epoch in range(Config.EPOCHS):
                    print(f"\nÉpoca {epoch + 1}/{Config.EPOCHS}")
                    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, Config.DEVICE)
                    val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, Config.DEVICE)
                    print(f"Treino -> Perda: {train_loss:.3f}, Acurácia: {train_acc:.3f}")
                    print(f"Validação -> Perda: {val_loss:.3f}, Acurácia: {val_acc:.3f}")
                    train_acc_history.append(train_acc)
                    val_acc_history.append(val_acc)
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        torch.save(model.state_dict(), os.path.join(feature_output_dir, 'best_model.pth'))
                
                save_accuracy_plot(train_acc_history, val_acc_history, os.path.join(feature_output_dir, 'accuracy_plot.png'))
                
                print("\n[INFO] Avaliando no conjunto de teste...")
                model.load_state_dict(torch.load(os.path.join(feature_output_dir, 'best_model.pth')))
                _, test_acc, y_pred, y_true = evaluate(model, test_loader, criterion, Config.DEVICE)
                
                model_all_true_labels.extend(y_true)
                model_all_pred_labels.extend(y_pred)

                macc = accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                
                print(f"Acurácia (mACC) no Teste: {macc:.3f}")
                
                report_for_log = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
                print(report_for_log)
                
                report_for_csv = classification_report(y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True)
                report_df = pd.DataFrame(report_for_csv).transpose()
                
                float_cols = ['precision', 'recall', 'f1-score', 'support']
                for col in float_cols:
                    if col in report_df.columns:
                        report_df[col] = pd.to_numeric(report_df[col], errors='coerce')
                report_df = report_df.round(3)

                csv_report_path = os.path.join(feature_output_dir, 'classification_report.csv')
                report_df.to_csv(csv_report_path)

                save_confusion_matrix(y_true, y_pred, class_names, os.path.join(feature_output_dir, 'confusion_matrix.png'))
                
                model_summary_data.append({
                    "model_name": model_name,
                    "attribute": feature,
                    "accuracy": macc,
                    "f1_score": f1,
                    "precision": precision,
                    "recall": recall,
                    "dataset": Config.DATASET_NAME
                })
                
                all_feature_performances.append({
                    "model": model_name, "feature": feature, "train_acc": train_acc_history[-1] if train_acc_history else 0,
                    "val_acc": best_val_acc, "test_acc": macc
                })
                model_train_acc_histories.append(train_acc_history)
                model_val_acc_histories.append(val_acc_history)

            if model_summary_data:
                summary_df = pd.DataFrame(model_summary_data)
                summary_df = summary_df.round({'accuracy': 3, 'f1_score': 3, 'precision': 3, 'recall': 3})
                summary_csv_path = os.path.join(model_output_dir, f'{model_name}_summary_report.csv')
                summary_df.to_csv(summary_csv_path, index=False)
                print(f"\n[INFO] Relatório de resumo do modelo salvo em: {summary_csv_path}")
                
                # <--- ALTERAÇÃO: Adiciona os dados de resumo do modelo à lista geral --->
                general_summary_list.extend(model_summary_data)

            print("\n" + "-"*80)
            print(f"[INFO] Métricas Agregadas para o Modelo: {model_name}")
            print("-" * 80)
            general_macc = accuracy_score(model_all_true_labels, model_all_pred_labels)
            general_f1 = f1_score(model_all_true_labels, model_all_pred_labels, average='weighted', zero_division=0)
            print(f"  - mACC Geral (todas as features): {general_macc:.3f}")
            print(f"  - F1-Score Ponderado Geral: {general_f1:.3f}")
            save_confusion_matrix(model_all_true_labels, model_all_pred_labels, 
                                  class_names=['Não Presente', 'Presente'], 
                                  save_path=os.path.join(model_output_dir, 'general_confusion_matrix.png'),
                                  title=f'Matriz de Confusão Geral - {model_name}')
            print(f"  - Matriz de confusão geral salva em: {model_output_dir}")

            if model_train_acc_histories and any(model_train_acc_histories):
                max_len = Config.EPOCHS
                padded_train_histories = [h + [h[-1]] * (max_len - len(h)) if h else [0]*max_len for h in model_train_acc_histories]
                padded_val_histories = [h + [h[-1]] * (max_len - len(h)) if h else [0]*max_len for h in model_val_acc_histories]
                avg_train_acc = np.mean(np.array(padded_train_histories), axis=0)
                avg_val_acc = np.mean(np.array(padded_val_histories), axis=0)
                overall_results[model_name] = {'avg_train_acc': avg_train_acc, 'avg_val_acc': avg_val_acc}
                save_accuracy_plot(avg_train_acc, avg_val_acc,
                                   save_path=os.path.join(model_output_dir, 'general_accuracy_plot.png'),
                                   title=f'Acurácia Média das Features - {model_name}')
                print(f"  - Gráfico de acurácia média salvo em: {model_output_dir}")
        finally:
            if sys.stdout is not original_stdout:
                sys.stdout.close()
                sys.stdout = original_stdout
            print(f"[INFO] Saída para o modelo '{model_name}' foi salva em '{log_file_path}'")

    print("\n" + "="*80)
    print("[INFO] Gerando Relatórios Finais")
    print("="*80)
    
    if overall_results:
        save_average_accuracy_plot(overall_results, os.path.join(Config.OUTPUT_DIR, 'average_accuracy_comparison.png'))
    if all_feature_performances:
        performance_df = pd.DataFrame(all_feature_performances)
        performance_df = performance_df.round({'train_acc': 3, 'val_acc': 3, 'test_acc': 3})
        performance_df_sorted = performance_df.sort_values(by="test_acc", ascending=False)
        csv_path = os.path.join(Config.OUTPUT_DIR, 'final_performance_report.csv')
        performance_df_sorted.to_csv(csv_path, index=False)
        print(f"[INFO] Relatório de performance final salvo em: {csv_path}")

    # <--- ALTERAÇÃO: Cria e salva o relatório de resumo geral --->
    if general_summary_list:
        general_summary_df = pd.DataFrame(general_summary_list)
        general_summary_df = general_summary_df.round({'accuracy': 3, 'f1_score': 3, 'precision': 3, 'recall': 3})
        general_csv_path = os.path.join(Config.OUTPUT_DIR, 'general_summary_report.csv')
        general_summary_df.to_csv(general_csv_path, index=False)
        print(f"[INFO] Relatório de resumo geral salvo em: {general_csv_path}")


if __name__ == '__main__':
    main()