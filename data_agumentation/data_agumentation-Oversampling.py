import pandas as pd
import matplotlib.pyplot as plt
import os
from torchvision import transforms
from PIL import Image
import seaborn as sns
from tqdm import tqdm
import random
import shutil
import concurrent.futures
import collections

# ===============================================================
# 1. CLASSE DE CONFIGURAÇÃO
# ===============================================================
class Config:
    """
    Classe para centralizar todas as configurações do script.
    """
    # --- CAMINHOS DE ENTRADA ---
    BASE_PATH = "/mnt/c/Users/gdaud/Desktop/3 - Experiência Criativa Projeto Transformador II - TCC/PROJETO-TCC-WSL/Projeto-Transformador-II/data/"
    TRAIN_CSV_PATH = os.path.join(BASE_PATH, "Market1501/annotations/phase1/train/train.csv")

    # --- DIRETÓRIO PRINCIPAL DE SAÍDA ---
    # CORREÇÃO: Removido as aspas simples extras e a barra inicial para um caminho mais seguro
    SAVE_DIR = 'data-oversampling/augmented_images'

    # --- NOMES DAS PASTAS E ARQUIVOS GERADOS ---
    AUGMENTED_ONLY_DIR_NAME = 'augmented_only'
    ALL_IMAGES_DIR_NAME = 'all_images'
    FINAL_CSV_NAME = 'dataset_aumentado.csv'
    PLOT_BEFORE_NAME = 'class_distribution_before.png'
    PLOT_AFTER_NAME = 'class_distribution_after.png'

    # --- PARÂMETROS DE AUMENTO ---
    TARGET_IMAGES_PER_CLASS = 55000
    AUGMENTATION_MODE = 'BALANCE'
    TRANSFORMATION_TYPE = 'SIMPLE'

# ===============================================================
# 2. DEFINIÇÃO DAS TRANSFORMAÇÕES
# ===============================================================

# --- TRANSFORMAÇÃO SIMPLES ---
simple_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=(256, 128), scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.2),
])

# --- TRANSFORMAÇÃO COMPLETA ---
complete_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=(256, 128), scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10)
    ], p=0.8),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
    ], p=0.8),
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
    ], p=0.5),
    transforms.RandomChoice([
        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=1.0),
        transforms.RandomAutocontrast(p=1.0),
        transforms.RandomEqualize(p=1.0),
        transforms.RandomSolarize(threshold=192.0, p=1.0),
        transforms.RandomPosterize(bits=4, p=1.0),
    ]),
    transforms.RandomGrayscale(p=0.2),
])

# ===============================================================
# 3. FUNÇÕES DE TRABALHO
# ===============================================================
def copy_image(paths):
    full_original_path, destination_path = paths
    if os.path.exists(full_original_path) and not os.path.exists(destination_path):
        shutil.copy(full_original_path, destination_path)
    return True

def process_and_augment_image(args):
    original_row_dict, n_count, all_images_dir, augmented_only_dir, transform_pipeline = args
    original_img_name = os.path.basename(original_row_dict['image_path'])
    original_img_path = os.path.join(all_images_dir, original_img_name)
    try:
        image = Image.open(original_img_path).convert("RGB")
        augmented_image = transform_pipeline(image)
        base_name, ext = os.path.splitext(original_img_name)
        augmented_filename = f"{base_name}_augmented_{n_count}{ext}"

        augmented_only_save_path = os.path.join(augmented_only_dir, augmented_filename)
        augmented_image.save(augmented_only_save_path)
        all_images_save_path = os.path.join(all_images_dir, augmented_filename)
        shutil.copy(augmented_only_save_path, all_images_save_path)

        new_row = original_row_dict.copy()
        new_row['image_path'] = os.path.join(Config.SAVE_DIR, Config.ALL_IMAGES_DIR_NAME, augmented_filename).replace("\\", "/")
        return new_row
    except Exception as e:
        return None

# ===============================================================
# 4. SCRIPT PRINCIPAL
# ===============================================================
if __name__ == "__main__":
    os.makedirs(Config.SAVE_DIR, exist_ok=True)

    print(f"Usando o tipo de transformação: '{Config.TRANSFORMATION_TYPE}'")
    if Config.TRANSFORMATION_TYPE == 'SIMPLE':
        pil_augmentation_transform = simple_transform
    elif Config.TRANSFORMATION_TYPE == 'COMPLETE':
        pil_augmentation_transform = complete_transform
    else:
        raise ValueError(f"Tipo de transformação '{Config.TRANSFORMATION_TYPE}' inválido. Use 'SIMPLE' ou 'COMPLETE'.")

    df = pd.read_csv(Config.TRAIN_CSV_PATH)

    # --- SEÇÃO DE ANÁLISE INICIAL ---
    print("Analisando o desbalanceamento das classes...")
    counts = []
    for column in df.columns[1:]:
        value_counts = df[column].astype(int).value_counts()
        count_0 = value_counts.get(0, 0)
        count_1 = value_counts.get(1, 0)
        counts.append([column, "0", count_0])
        counts.append([column, "1", count_1])
    counts_df = pd.DataFrame(counts, columns=["Feature", "Class", "Count"])

    print("Gerando gráfico de distribuição de classes ANTES do aumento...")
    g = sns.catplot(data=counts_df, kind="bar", x="Feature", y="Count", hue="Class", height=5, aspect=3.5)
    g.set_xticklabels(rotation=75, ha="right")
    g.fig.suptitle('Distribuição de Classes Antes do Aumento de Dados', y=1.03)
    plot_path_before = os.path.join(Config.SAVE_DIR, Config.PLOT_BEFORE_NAME)
    g.savefig(plot_path_before)
    plt.close()
    print(f"Gráfico salvo em: {plot_path_before}")

    print("\n" + "="*65)
    print(" Contagem de dados por classe (antes do aumento de dados)".upper())
    print("="*65)
    summary_df = counts_df.pivot_table(index='Feature', columns='Class', values='Count').reset_index()
    summary_df.columns.name = None
    summary_df = summary_df[['Feature', '1', '0']].rename(columns={'1': 'Com Atributo (Classe 1)', '0': 'Sem Atributo (Classe 0)'})
    print(summary_df.to_string(index=False))
    print("="*65 + "\n")

    # --- PREPARAÇÃO DAS PASTAS E CÓPIA DAS IMAGENS ---
    augmented_only_dir = os.path.join(Config.SAVE_DIR, Config.AUGMENTED_ONLY_DIR_NAME)
    all_images_dir = os.path.join(Config.SAVE_DIR, Config.ALL_IMAGES_DIR_NAME)
    os.makedirs(augmented_only_dir, exist_ok=True)
    os.makedirs(all_images_dir, exist_ok=True)

    print("Copiando imagens originais...")
    copy_tasks = [
        (os.path.join(Config.BASE_PATH, row['image_path']), os.path.join(all_images_dir, os.path.basename(row['image_path'])))
        for _, row in df.iterrows()
    ]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(copy_image, copy_tasks), total=len(copy_tasks), desc="Copiando originais"))

    # --- PROCESSO DE AUMENTO DE DADOS ---
    print(f"\nIniciando o processo de aumento de dados (Modo: {Config.AUGMENTATION_MODE})")
    all_augmentation_tasks = []
    for feature in df.columns[1:]:
        images_to_generate = 0
        if Config.AUGMENTATION_MODE == 'BALANCE':
            current_count = df[feature].sum()
            if current_count < Config.TARGET_IMAGES_PER_CLASS:
                images_to_generate = Config.TARGET_IMAGES_PER_CLASS - current_count
        elif Config.AUGMENTATION_MODE == 'AUGMENT_ALL':
            images_to_generate = Config.TARGET_IMAGES_PER_CLASS

        if images_to_generate <= 0:
            continue

        print(f"\nFeature '{feature}': Preparando {images_to_generate} novas imagens.")
        feature_df = df[df[feature] == 1]
        if len(feature_df) == 0: continue

        source_rows = [row for _, row in feature_df.iterrows()]
        num_repeats = (images_to_generate // len(source_rows)) + 1
        generation_pool = (source_rows * num_repeats)[:images_to_generate]
        random.shuffle(generation_pool)
        usage_counter = collections.Counter()

        for original_row in generation_pool:
            original_img_name = os.path.basename(original_row['image_path'])
            usage_counter[original_img_name] += 1
            n_count = usage_counter[original_img_name]
            task_args = (original_row.to_dict(), n_count, all_images_dir, augmented_only_dir, pil_augmentation_transform)
            all_augmentation_tasks.append(task_args)

    new_rows = []
    if all_augmentation_tasks:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(process_and_augment_image, all_augmentation_tasks), total=len(all_augmentation_tasks), desc="Gerando imagens"))
            new_rows = [row for row in results if row is not None]

    # --- FINALIZAÇÃO E CRIAÇÃO DO CSV ---
    print("\nAtualizando os paths das imagens originais no dataframe...")
    df['image_path'] = df['image_path'].apply(
        lambda path: os.path.join(Config.SAVE_DIR, Config.ALL_IMAGES_DIR_NAME, os.path.basename(path)).replace("\\", "/")
    )
    if new_rows:
        augmented_df = pd.DataFrame(new_rows)
        final_df = pd.concat([df, augmented_df], ignore_index=True)
    else:
        final_df = df

    final_csv_path = os.path.join(Config.SAVE_DIR, Config.FINAL_CSV_NAME)
    final_df.to_csv(final_csv_path, index=False)
    print(f"\nProcesso concluído! Novo CSV salvo em: '{final_csv_path}'")

    # --- SEÇÃO DE ANÁLISE FINAL (RESTAURADA) ---
    print("\nAnalisando a distribuição das classes APÓS o aumento...")
    final_counts = []
    for column in final_df.columns[1:]:
        value_counts = final_df[column].astype(int).value_counts()
        count_0 = value_counts.get(0, 0)
        count_1 = value_counts.get(1, 0)
        final_counts.append([column, "0", count_0])
        final_counts.append([column, "1", count_1])
    final_counts_df = pd.DataFrame(final_counts, columns=["Feature", "Class", "Count"])

    print("Gerando gráfico de distribuição de classes APÓS o aumento...")
    g_after = sns.catplot(data=final_counts_df, kind="bar", x="Feature", y="Count", hue="Class", height=5, aspect=3.5)
    g_after.set_xticklabels(rotation=75, ha="right")
    g_after.fig.suptitle('Distribuição de Classes Após o Aumento de Dados', y=1.03)
    plot_path_after = os.path.join(Config.SAVE_DIR, Config.PLOT_AFTER_NAME)
    g_after.savefig(plot_path_after)
    plt.close()
    print(f"Gráfico final salvo em: {plot_path_after}")
