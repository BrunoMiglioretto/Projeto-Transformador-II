import os
import pandas as pd
import numpy as np
import time  # <--- ADICIONADO
from collections import defaultdict
from PIL import Image
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# Ajustado para incluir train_test_split para a nova divisão
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
from scipy import stats  # Importado para Hard Voting (stats.mode)

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
# 1. CLASSE DE CONFIGURAÇÃO (AJUSTADA)
# ======================================================================================
class Config:
    """
    Classe de configuração para todos os hiperparâmetros e caminhos.
    """
    # Caminhos
    DATA_ROOT: str = "/home/gdaudt/Área de trabalho/Projeto-Transformador-II/data/PAR2025"
    
    # === CAMINHOS DE ARQUIVOS TXT AJUSTADOS ===
    # Arquivo .txt para treino
    TRAIN_TXT: str = os.path.join(DATA_ROOT, "training_set.txt")
    # Arquivo .txt único que será dividido em Validação e Teste
    VAL_TEST_TXT: str = os.path.join(DATA_ROOT, "validation_set.txt")
    
    OUTPUT_DIR: str = "./output-com_transfer_leaning"
    DATASET_NAME: str = os.path.basename(DATA_ROOT)

    # Parâmetros de Treinamento
    MODELS_TO_TRAIN: list = ['MobileNetV2', 'EfficientNet-B0', 'SwinV2-T']
    IMG_SIZE: int = 200
    BATCH_SIZE: int = 64
    EPOCHS: int = 10
    LEARNING_RATE: float = 1e-4

    # === PARÂMETROS DE DIVISÃO (RE-ADICIONADOS) ===
    VAL_SPLIT_FOR_TEST: float = 0.5 
    RANDOM_STATE: int = 42 

    # Dispositivo
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    
# ======================================================================================
# 2. DATASET E DATALOADER
# ======================================================================================
class AttributeDataset(Dataset):
    def __init__(self, df, data_root, transform=None):
        self.df = df
        # data_root agora será o caminho específico (ex: .../PAR2025/training_set)
        self.data_root = data_root 
        self.transform = transform
        self.image_paths = df['image_path'].values
        self.labels = df['label'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Caminho da imagem é relativo ao DATA_ROOT (que agora é a pasta correta)
        img_path = os.path.join(self.data_root, self.image_paths[idx])
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            # print(f"Warning: File not found {img_path}. Skipping.") # Removido para poluir menos
            return None, None
            
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label

def get_transforms(img_size):
    return {
        'train': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val_test': transforms.Compose([
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
    if model_name == 'MobileNetV2':
        model = models.mobilenet_v2(weights=weights)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'EfficientNet-B0':
        model = models.efficientnet_b0(weights=weights)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'SwinV2-T':
        model = models.swin_v2_t(weights=weights)
        num_ftrs = model.head.in_features
        model.head = nn.Linear(num_ftrs, num_classes)
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
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validando/Testando"):
            if inputs.nelement() == 0: continue
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            correct_predictions += torch.sum(preds == labels.data)
            total_samples += len(inputs)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy()) 

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0
    epoch_acc = correct_predictions.double() / total_samples if total_samples > 0 else 0
    return epoch_loss, epoch_acc.item() if isinstance(epoch_acc, torch.Tensor) else epoch_acc, all_preds, all_labels, np.array(all_probs)

# ======================================================================================
# 5. UTILITÁRIOS (PLOTS, SALVAMENTO E SPLIT)
# ======================================================================================

def get_original_image_path(path_str):
    """
    Identifica o nome do arquivo da imagem original, removendo sufixos de aumento de dados.
    Assume que os arquivos aumentados contêm '_aug_' em seu nome.
    """
    if '_aug_' in path_str:
        base_name = path_str.split('_aug_')[0]
        original_name = f"{base_name}.jpg" 
        return original_name
    return os.path.basename(path_str)


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
    
    # Lida com o caso de ser multiclasse ou binário para a CM
    labels_present = sorted(list(np.unique(y_true + y_pred)))
    if 0 not in labels_present: labels_present = [0] + labels_present
    if 1 not in labels_present and len(labels_present) < 3 : labels_present = labels_present + [1]
    
    cm = confusion_matrix(y_true, y_pred, labels=labels_present)
    plt.figure(figsize=(10, 8))
    
    # Ajusta os nomes das classes se for binário
    display_names = class_names if len(labels_present) <= 2 else labels_present
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=display_names, yticklabels=display_names)
    plt.title(title)
    plt.ylabel('Rótulo Verdadeiro')
    plt.xlabel('Rótulo Previsto')
    plt.savefig(save_path)
    plt.close()

# ======================================================================================
# 6. FUNÇÃO PRINCIPAL (LÓGICA DE DADOS CORRIGIDA)
# ======================================================================================

def main():
    print(f"[INFO] Usando dispositivo: {Config.DEVICE}")
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # --- LÓGICA DE DADOS AJUSTADA PARA DIVIDIR VAL_TEST_TXT ---
    print("\n" + "="*80)
    print("[INFO] Carregando arquivos TXT...")
    
    # Defina os nomes das colunas, pois os arquivos TXT não têm cabeçalho.
    column_names = [
        'image_path', 
        'upper_color', 
        'lower_color', 
        'gender', 
        'bag', 
        'hat'
    ]

    try:
        # Carrega o TXT de treino SEM cabeçalho
        train_df = pd.read_csv(
            Config.TRAIN_TXT, 
            header=None,        
            names=column_names  
        ) 
        
        # Carrega o TXT de Val/Teste SEM cabeçalho
        val_test_df = pd.read_csv(
            Config.VAL_TEST_TXT, 
            header=None,        
            names=column_names  
        )
        
    except FileNotFoundError as e:
        print(f"[ERRO] Arquivo TXT não encontrado: {e}")
        print(f"Por favor, verifique os caminhos em Config:")
        print(f"  TRAIN_TXT: {Config.TRAIN_TXT}")
        print(f"  VAL_TEST_TXT: {Config.VAL_TEST_TXT}")
        return
    except Exception as e:
        print(f"[ERRO] Não foi possível ler os arquivos TXT. Verifique o formato.")
        print(f"Erro: {e}")
        return

    print(f"[INFO] Total de amostras de Treino (bruto): {len(train_df)}")
    print(f"[INFO] Total de amostras de Val/Teste (bruto): {len(val_test_df)}")
    
    # --- LÓGICA DE DIVISÃO ROBUSTA (ANTI-DATA LEAKAGE) ---
    print("[INFO] Preparando e dividindo o dataset de Val/Teste...")

    val_test_df['original_path'] = val_test_df['image_path'].apply(get_original_image_path)
    unique_original_paths = val_test_df['original_path'].unique()
    print(f"[INFO] Encontradas {len(unique_original_paths)} imagens originais únicas no arquivo de val/teste.")

    val_ids, test_ids = train_test_split(
        unique_original_paths,
        test_size=Config.VAL_SPLIT_FOR_TEST,
        random_state=Config.RANDOM_STATE
    )

    val_df = val_test_df[val_test_df['original_path'].isin(val_ids)].copy()
    test_df = val_test_df[val_test_df['original_path'].isin(test_ids)].copy()

    # Adiciona a coluna 'original_path' ao train_df também
    train_df['original_path'] = train_df['image_path'].apply(get_original_image_path)

    print(f"[INFO] Divisão final (IDs): Validação: {len(val_ids)} | Teste: {len(test_ids)}")
    print(f"[INFO] Total de amostras: Treino: {len(train_df)} | Validação: {len(val_df)} | Teste: {len(test_df)}")
    print("="*80)
    
    feature_cols = [col for col in train_df.columns if col not in ['image_path', 'original_path']]
    print(f"[INFO] Atributos encontrados para treinamento: {feature_cols}")

    general_summary_list = []
    ensemble_summary_list = []

    # === LOOP PRINCIPAL: POR ATRIBUTO ===
    for feature in feature_cols: 
        print(f"\n{'='*80}")
        print(f"[INFO] Iniciando processamento para o atributo: {feature}")
        print(f"{'='*80}")

        # 1. Preparar DataFrames para este atributo
        feature_train_df = train_df[['image_path', feature]].copy().rename(columns={feature: 'label'})
        feature_val_df = val_df[['image_path', feature]].copy().rename(columns={feature: 'label'})
        feature_test_df = test_df[['image_path', feature]].copy().rename(columns={feature: 'label'})

        # --- Lógica de Classes (AJUSTADA para multiclasse) ---
        all_labels = pd.concat([feature_train_df['label'], feature_val_df['label'], feature_test_df['label']])
        unique_labels = sorted(all_labels.unique())
        
        is_color_feature = feature in ['upper_color', 'lower_color']
        
        if is_color_feature:
            feature_train_df['label'] = feature_train_df['label'] - 1
            feature_val_df['label'] = feature_val_df['label'] - 1
            feature_test_df['label'] = feature_test_df['label'] - 1
            
            num_classes = 11 # 11 classes (índices 0-10)
            class_names = [str(i) for i in range(1, 12)] # Nomes 1 a 11
            report_labels = list(range(11)) # Labels para o relatório 0 a 10
            
            print(f"[INFO] Atributo de COR detectado. Labels ajustados de 1-11 para 0-10 Para CrossEntropyLoss. Num classes: {num_classes}")
            
        else: # gender, bag, hat
            num_classes = 2
            class_names = [f'Not_{feature}', feature]
            report_labels = [0, 1]
            print(f"[INFO] Atributo BINÁRIO detectado. Num classes: {num_classes}")
             
        # 2. Preparar DataLoaders para este atributo
        train_img_dir = os.path.join(Config.DATA_ROOT, 'training_set')
        val_img_dir = os.path.join(Config.DATA_ROOT, 'validation_set')
        test_img_dir = val_img_dir 
        
        transforms_dict = get_transforms(Config.IMG_SIZE)
        
        train_dataset = AttributeDataset(feature_train_df, train_img_dir, transforms_dict['train'])
        val_dataset = AttributeDataset(feature_val_df, val_img_dir, transforms_dict['val_test'])
        test_dataset = AttributeDataset(feature_test_df, test_img_dir, transforms_dict['val_test'])
        
        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn, pin_memory=True)
        
        feature_test_results = {'preds': {}, 'probs': {}, 'labels': None}

        # === LOOP SECUNDÁRIO: TREINAR TODOS OS MODELOS PARA ESTE ATRIBUTO ===
        for model_name in Config.MODELS_TO_TRAIN:
            
            feature_output_dir = os.path.join(Config.OUTPUT_DIR, feature.replace('&','_'), model_name)
            os.makedirs(feature_output_dir, exist_ok=True)
            
            log_file_path = os.path.join(feature_output_dir, 'terminal_output.txt')
            original_stdout = sys.stdout
            sys.stdout = Logger(log_file_path)

            try:
                print("\n" + "="*80)
                print(f"[INFO] Treinando modelo: {model_name} para o atributo: {feature} (Classes: {num_classes})")
                print(f"[INFO] A saída está sendo salva em: {log_file_path}")
                print("="*80)

                model = get_model(model_name, num_classes).to(Config.DEVICE)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

                train_acc_history, val_acc_history = [], []
                best_val_acc = 0.0
                best_model_path = os.path.join(feature_output_dir, 'best_model.pth')

                # 3. Loop de Treinamento
                
                # ==================================================================
                # === INÍCIO DA MEDIÇÃO DE TEMPO ===
                # ==================================================================
                start_time = time.time()
                
                for epoch in range(Config.EPOCHS):
                    print(f"\nÉpoca {epoch + 1}/{Config.EPOCHS}")
                    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, Config.DEVICE)
                    val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, Config.DEVICE)
                    
                    print(f"Treino -> Perda: {train_loss:.4f}, Acurácia: {train_acc:.4f}")
                    print(f"Validação -> Perda: {val_loss:.4f}, Acurácia: {val_acc:.4f}")
                    
                    train_acc_history.append(train_acc)
                    val_acc_history.append(val_acc)
                    
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        torch.save(model.state_dict(), best_model_path)
                
                # ==================================================================
                # === FIM DA MEDIÇÃO DE TEMPO ===
                # ==================================================================
                end_time = time.time()
                total_train_time_sec = end_time - start_time
                print(f"\n[INFO] Tempo total de treinamento: {total_train_time_sec:.2f} segundos")
                
                
                save_accuracy_plot(train_acc_history, val_acc_history, os.path.join(feature_output_dir, 'accuracy_plot.png'))
                
                # 4. Avaliação no Teste (com o melhor modelo salvo)
                print("\n[INFO] Avaliando no conjunto de teste com o melhor modelo...")
                
                # Adiciona uma verificação para o caso de nenhum modelo ser salvo (val_acc 0.0)
                if not os.path.exists(best_model_path):
                    print(f"[AVISO] Nenhum modelo foi salvo (best_model.path não encontrado). Provavelmente a acurácia de validação foi 0.0.")
                    print("[AVISO] Pulando avaliação de teste para este modelo.")
                    # Pula para o 'finally'
                    continue 

                model.load_state_dict(torch.load(best_model_path))
                
                test_loss, test_acc, y_pred, y_true, y_probs = evaluate(model, test_loader, criterion, Config.DEVICE)
                
                feature_test_results['preds'][model_name] = np.array(y_pred)
                feature_test_results['probs'][model_name] = np.array(y_probs)
                if feature_test_results['labels'] is None:
                    feature_test_results['labels'] = np.array(y_true)

                # 5. Cálculo de Métricas Detalhadas
                macc = accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                
                print(f"Acurácia (mACC) no Teste: {macc:.4f}")

                print(classification_report(y_true, y_pred, target_names=class_names, labels=report_labels, zero_division=0))
                
                if is_color_feature:
                    tn, fp, fn, tp = 0, 0, 0, 0
                    acertou = int(np.sum(np.array(y_true) == np.array(y_pred)))
                    errou = int(len(y_true) - acertou)
                else:
                    try:
                        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                        if cm.size == 4:
                            tn, fp, fn, tp = cm.ravel()
                        else:
                            tn, fp, fn, tp = 0, 0, 0, 0
                    except Exception:
                        tn, fp, fn, tp = 0, 0, 0, 0
                    acertou = int(tp + tn)
                    errou = int(fp + fn)
                
                save_confusion_matrix(y_true, y_pred, class_names, os.path.join(feature_output_dir, 'confusion_matrix.png'), title=f'CM {feature}')
                
                # 6. Adicionar ao Relatório Geral
                general_summary_list.append({
                    "model": model_name, 
                    "attribute": feature, 
                    "test_accuracy": macc, 
                    "test_loss": test_loss,
                    "f1_score": f1,
                    "precision": precision, 
                    "recall": recall, 
                    "true_positives": int(tp),
                    "false_negatives": int(fn),
                    "false_positives": int(fp),
                    "true_negatives": int(tn),
                    "total_acertos": acertou,
                    "total_erros": errou,
                    "total_train_time_sec": total_train_time_sec, # <--- COLUNA ADICIONADA
                    "dataset": Config.DATASET_NAME
                })

            finally:
                if sys.stdout is not original_stdout:
                    sys.stdout.close()
                    sys.stdout = original_stdout
                print(f"[INFO] Saída para '{model_name}' (feature: {feature}) salva em '{log_file_path}'")
        
        # === FIM DO LOOP DE MODELOS ===

        # 7. Calcular Ensemble para o Atributo Atual
        print(f"\n[INFO] Calculando Ensemble (Hard & Soft Voting) para o atributo: {feature}")
        y_true = feature_test_results['labels']
        
        if y_true is None or len(y_true) == 0:
            print(f"[AVISO] Não há labels de teste para {feature}. Pulando ensemble.")
            continue
            
        def get_cm_metrics(y_true, y_pred, labels):
            if len(labels) > 2: # Multiclasse
                tp, fn, fp, tn = 0, 0, 0, 0
            else: # Binário
                cm = confusion_matrix(y_true, y_pred, labels=labels)
                tn, fp, fn, tp = (cm.ravel() if cm.size == 4 else (0,0,0,0))
            return int(tp), int(fn), int(fp), int(tn)

        # --- Votação Majoritária (Hard Voting) ---
        try:
            preds_stack = np.stack([feature_test_results['preds'][m] for m in Config.MODELS_TO_TRAIN], axis=1)
            hard_votes, _ = stats.mode(preds_stack, axis=1, keepdims=False)
            
            acc_hard = accuracy_score(y_true, hard_votes)
            f1_hard = f1_score(y_true, hard_votes, average='weighted', zero_division=0)
            p_hard = precision_score(y_true, hard_votes, average='weighted', zero_division=0)
            r_hard = recall_score(y_true, hard_votes, average='weighted', zero_division=0)
            tp_h, fn_h, fp_h, tn_h = get_cm_metrics(y_true, hard_votes, report_labels)
            
            print(f"  - Hard Voting -> Acurácia: {acc_hard:.4f}, F1: {f1_hard:.4f}")
        except Exception as e:
            print(f"Erro no Hard Voting: {e}")
            acc_hard, f1_hard, p_hard, r_hard, tn_h, fp_h, fn_h, tp_h = 0,0,0,0,0,0,0,0

        # --- Média das Probabilidades (Soft Voting) ---
        try:
            probs_stack = np.stack([feature_test_results['probs'][m] for m in Config.MODELS_TO_TRAIN], axis=1)
            avg_probs = np.mean(probs_stack, axis=1) 
            soft_votes = np.argmax(avg_probs, axis=1)
            
            acc_soft = accuracy_score(y_true, soft_votes)
            f1_soft = f1_score(y_true, soft_votes, average='weighted', zero_division=0)
            p_soft = precision_score(y_true, soft_votes, average='weighted', zero_division=0)
            r_soft = recall_score(y_true, soft_votes, average='weighted', zero_division=0)
            tp_s, fn_s, fp_s, tn_s = get_cm_metrics(y_true, soft_votes, report_labels)
            
            print(f"  - Soft Voting -> Acurácia: {acc_soft:.4f}, F1: {f1_soft:.4f}")
        except Exception as e:
            print(f"Erro no Soft Voting: {e}")
            acc_soft, f1_soft, p_soft, r_soft, tn_s, fp_s, fn_s, tp_s = 0,0,0,0,0,0,0,0

        # 8. Adicionar ao Relatório de Ensemble
        ensemble_summary_list.append({
            "attribute": feature,
            "hard_accuracy": acc_hard,
            "hard_f1_score": f1_hard,
            "hard_precision": p_hard,
            "hard_recall": r_hard,
            "hard_tp": tp_h, "hard_fn": fn_h, "hard_fp": fp_h, "hard_tn": tn_h,
            "soft_accuracy": acc_soft,
            "soft_f1_score": f1_soft,
            "soft_precision": p_soft,
            "soft_recall": r_soft,
            "soft_tp": tp_s, "soft_fn": fn_s, "soft_fp": fp_s, "soft_tn": tn_s,
        })

    # === FIM DO LOOP DE ATRIBUTOS ===

    # 9. Gerar Relatórios Finais
    print("\n" + "="*80)
    print("[INFO] Gerando Relatórios Finais")
    print("="*80)

    if general_summary_list:
        general_summary_df = pd.DataFrame(general_summary_list)
        # Reordena colunas para colocar o tempo no final
        cols = [c for c in general_summary_df.columns if c not in ['dataset', 'total_train_time_sec']]
        cols.append('total_train_time_sec')
        cols.append('dataset')
        general_summary_df = general_summary_df[cols]
        
        general_summary_df = general_summary_df.round(4)
        general_csv_path = os.path.join(Config.OUTPUT_DIR, 'general_summary_report.csv')
        general_summary_df.to_csv(general_csv_path, index=False)
        print(f"[INFO] Relatório de resumo geral (por modelo) salvo em: {general_csv_path}")
    
    if ensemble_summary_list:
        ensemble_summary_df = pd.DataFrame(ensemble_summary_list)
        ensemble_summary_df = ensemble_summary_df.round(4)
        ensemble_csv_path = os.path.join(Config.OUTPUT_DIR, 'ensemble_summary_report.csv')
        ensemble_summary_df.to_csv(ensemble_csv_path, index=False)
        print(f"[INFO] Relatório de resumo de Ensemble salvo em: {ensemble_csv_path}")

    print("[INFO] Processamento concluído.")

if __name__ == '__main__':
    main()