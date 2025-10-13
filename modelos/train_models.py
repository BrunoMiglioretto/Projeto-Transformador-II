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
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

# Ignorar avisos de convergência do Scikit-learn para relatórios mais limpos
warnings.filterwarnings("ignore", category=UserWarning)

# ======================================================================================
# <--- ALTERAÇÃO: CLASSE PARA REDIRECIONAR A SAÍDA DO TERMINAL
# ======================================================================================
class Logger:
    """
    Classe para redirecionar a saída (stdout) para um arquivo de log
    e também exibi-la no terminal.
    """
    def __init__(self, filename):
        self.terminal = sys.stdout
        # Abre o arquivo no modo 'write' ('w') para sobrescrever a cada execução
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # O método flush é necessário para compatibilidade com a interface de arquivos.
        self.terminal.flush()
        self.log.flush()

    def close(self):
        # Fecha o arquivo de log.
        self.log.close()

# ======================================================================================
# 1. CLASSE DE CONFIGURAÇÃO
# ======================================================================================
class Config:
    """
    Classe de configuração para todos os hiperparâmetros e caminhos.
    """
    # Caminhos
    DATA_ROOT: str = "../data"
    TRAIN_CSV: str = os.path.join(DATA_ROOT, "annotations/phase1/train/train.csv")
    OUTPUT_DIR: str = "./DATASET-ORIGINAL-BASELINE"

    # Parâmetros de Treinamento
    MODELS_TO_TRAIN: list = [
                                'MobileNetV2',
                                'EfficientNet-B0',
                                'ResNet50',
                            ]
    IMG_SIZE: int = 200
    BATCH_SIZE: int = 64
    EPOCHS: int = 6
    LEARNING_RATE: float = 1e-4

    # Divisão do Dataset
    TEST_SPLIT_SIZE: float = 0.20 # 20% para teste final
    VALIDATION_SPLIT_SIZE: float = 0.25 # 25% de 80% resulta em 20% do total

    # Dispositivo
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# ======================================================================================
# 2. DATASET E DATALOADER
# ======================================================================================
class AttributeDataset(Dataset):
    """
    Dataset customizado para carregar imagens e seus respectivos rótulos.
    """
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
    """
    Define as transformações para as imagens.
    """
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
    return torch.utils.data.dataloader.default_collate(batch)

# ======================================================================================
# 3. MODELOS
# ======================================================================================
def get_model(model_name: str, num_classes: int, pretrained: bool = True):
    """
    Carrega um modelo pré-treinado e ajusta a camada de classificação.
    """
    model = None
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
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for inputs, labels in tqdm(dataloader, desc="Treinando"):
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
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validando/Testando"):
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
    # Este print inicial ainda vai para o terminal original
    print(f"[INFO] Usando dispositivo: {Config.DEVICE}")
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # Carregar o CSV
    df = pd.read_csv(Config.TRAIN_CSV)
    
    # Identificar todas as colunas de atributos (features)
    feature_cols = [col for col in df.columns if col != 'image_path']
    
    # Criar um dicionário onde cada atributo é seu próprio grupo
    feature_groups = {col: [col] for col in feature_cols}
        
    print(f"[INFO] Atributos encontrados para treinamento individual: {list(feature_groups.keys())}")

    overall_results = {}
    
    # Lista para armazenar os resultados para o CSV final
    all_feature_performances = []

    # Loop principal: para cada modelo
    for model_name in Config.MODELS_TO_TRAIN:
        
        # <--- ALTERAÇÃO: INÍCIO DO REDIRECIONAMENTO DE SAÍDA
        # Cria o diretório do modelo antes de criar o log
        model_output_dir = os.path.join(Config.OUTPUT_DIR, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        
        log_file_path = os.path.join(model_output_dir, 'terminal_output.txt')
        original_stdout = sys.stdout  # Salva a saída padrão original
        sys.stdout = Logger(log_file_path) # Redireciona a saída para nosso logger

        try:
            print("\n" + "="*80)
            print(f"[INFO] Iniciando ciclo para o modelo: {model_name}")
            print(f"[INFO] A saída deste terminal está sendo salva em: {log_file_path}")
            print("="*80)

            model_train_acc_histories = []
            model_val_acc_histories = []
            
            # Agregadores para métricas gerais do modelo
            model_all_true_labels = []
            model_all_pred_labels = []


            # Loop secundário: para cada atributo individual
            for feature, columns in feature_groups.items():
                print(f"\n--- Treinando para o atributo: {feature} ---")
                
                feature_df = df[['image_path'] + columns].copy()
                feature_df.rename(columns={columns[0]: 'label'}, inplace=True)
                
                class_names = [f'Not_{feature}', feature] 
                num_classes = 2
                
                if len(feature_df['label'].unique()) > 1:
                    train_val_df, test_df = train_test_split(
                        feature_df, test_size=Config.TEST_SPLIT_SIZE, random_state=42, stratify=feature_df['label']
                    )
                    train_df, val_df = train_test_split(
                        train_val_df, test_size=Config.VALIDATION_SPLIT_SIZE, random_state=42, stratify=train_val_df['label']
                    )
                else:
                    train_val_df, test_df = train_test_split(
                        feature_df, test_size=Config.TEST_SPLIT_SIZE, random_state=42
                    )
                    train_df, val_df = train_test_split(
                        train_val_df, test_size=Config.VALIDATION_SPLIT_SIZE, random_state=42
                    )


                # Preparar Datasets e DataLoaders
                transforms_dict = get_transforms(Config.IMG_SIZE)
                train_dataset = AttributeDataset(train_df, Config.DATA_ROOT, transforms_dict['train'])
                val_dataset = AttributeDataset(val_df, Config.DATA_ROOT, transforms_dict['val'])
                test_dataset = AttributeDataset(test_df, Config.DATA_ROOT, transforms_dict['val'])

                train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=collate_fn)
                val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=collate_fn)
                test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=collate_fn)

                model = get_model(model_name, num_classes).to(Config.DEVICE)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

                train_acc_history, val_acc_history = [], []
                best_val_acc = 0.0
                
                feature_output_dir = os.path.join(Config.OUTPUT_DIR, model_name, feature.replace('&','_'))
                os.makedirs(feature_output_dir, exist_ok=True)

                for epoch in range(Config.EPOCHS):
                    print(f"\nÉpoca {epoch + 1}/{Config.EPOCHS}")
                    
                    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, Config.DEVICE)
                    val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, Config.DEVICE)
                    
                    print(f"Treino -> Perda: {train_loss:.4f}, Acurácia: {train_acc:.4f}")
                    print(f"Validação -> Perda: {val_loss:.4f}, Acurácia: {val_acc:.4f}")

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
                print(f"Acurácia (mACC) no Teste: {macc:.4f}")
                
                report = classification_report(y_true, y_pred, target_names=class_names)
                print(report)
                
                with open(os.path.join(feature_output_dir, 'classification_report.txt'), 'w') as f:
                    f.write(report)

                save_confusion_matrix(y_true, y_pred, class_names, os.path.join(feature_output_dir, 'confusion_matrix.png'))
                
                # Adicionar resultados para o relatório final
                all_feature_performances.append({
                    "model": model_name,
                    "feature": feature,
                    "train_acc": train_acc_history[-1],
                    "val_acc": best_val_acc,
                    "test_acc": macc
                })

                model_train_acc_histories.append(train_acc_history)
                model_val_acc_histories.append(val_acc_history)

            # --- Métricas e plots agregados por modelo ---
            print("\n" + "-"*80)
            print(f"[INFO] Métricas Agregadas para o Modelo: {model_name}")
            print("-" * 80)
            
            # Calcular e exibir métricas gerais
            general_macc = accuracy_score(model_all_true_labels, model_all_pred_labels)
            general_f1 = f1_score(model_all_true_labels, model_all_pred_labels, average='weighted')
            print(f"  - mACC Geral (todas as features): {general_macc:.4f}")
            print(f"  - F1-Score Ponderado Geral: {general_f1:.4f}")

            # Salvar matriz de confusão geral para o modelo
            save_confusion_matrix(model_all_true_labels, model_all_pred_labels, 
                                class_names=['Não Presente', 'Presente'], 
                                save_path=os.path.join(model_output_dir, 'general_confusion_matrix.png'),
                                title=f'Matriz de Confusão Geral - {model_name}')
            print(f"  - Matriz de confusão geral salva em: {model_output_dir}")

            # Calcular e plotar acurácia média do modelo
            if model_train_acc_histories:
                avg_train_acc = np.mean(np.array(model_train_acc_histories), axis=0)
                avg_val_acc = np.mean(np.array(model_val_acc_histories), axis=0)
                
                overall_results[model_name] = {
                    'avg_train_acc': avg_train_acc,
                    'avg_val_acc': avg_val_acc
                }
                
                save_accuracy_plot(avg_train_acc, avg_val_acc,
                                save_path=os.path.join(model_output_dir, 'general_accuracy_plot.png'),
                                title=f'Acurácia Média das Features - {model_name}')
                print(f"  - Gráfico de acurácia média salvo em: {model_output_dir}")
        
        finally:
            # <--- ALTERAÇÃO: FIM DO REDIRECIONAMENTO
            # Garante que a saída padrão seja restaurada, mesmo se ocorrer um erro
            if sys.stdout is not original_stdout:
                sys.stdout.close() # Fecha o arquivo
                sys.stdout = original_stdout # Restaura
            
            # Este print final para o modelo irá para o terminal original
            print(f"[INFO] Saída para o modelo '{model_name}' foi salva em '{log_file_path}'")


    # --- Relatórios finais ---
    print("\n" + "="*80)
    print("[INFO] Gerando Relatórios Finais")
    print("="*80)
    
    # Gerar o gráfico final comparativo entre modelos
    if overall_results:
        save_average_accuracy_plot(overall_results, os.path.join(Config.OUTPUT_DIR, 'average_accuracy_comparison.png'))

    # Gerar e salvar o CSV de performance ordenado
    if all_feature_performances:
        performance_df = pd.DataFrame(all_feature_performances)
        performance_df_sorted = performance_df.sort_values(by="test_acc", ascending=False)
        csv_path = os.path.join(Config.OUTPUT_DIR, 'final_performance_report.csv')
        performance_df_sorted.to_csv(csv_path, index=False)
        print(f"[INFO] Relatório de performance final salvo em: {csv_path}")

if __name__ == '__main__':
    main()