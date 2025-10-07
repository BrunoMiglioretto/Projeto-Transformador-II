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
    O objetivo é balancear cada atributo para que a classe 0 e a
    classe 1 tenham o mesmo número de amostras (50/50), operando
    sobre uma amostra aleatória do dataset original.
    """
    # --- CAMINHOS DE ENTRADA ---
    BASE_PATH = "/home/gdaudt/Área de trabalho/Projeto-Transformador-II/data"
    TRAIN_CSV_PATH = os.path.join(BASE_PATH, "annotations/phase1/train/train.csv")

    # --- DIRETÓRIO PRINCIPAL DE SAÍDA ---
    SAVE_DIR = 'data-balanced-50-50-sample'

    # --- NOMES DAS PASTAS E ARQUIVOS GERADOS ---
    AUGMENTED_ONLY_DIR_NAME = 'augmented_only'
    ALL_IMAGES_DIR_NAME = 'all_images'
    FINAL_CSV_NAME = 'dataset_balanceado_amostra.csv'
    PLOT_BEFORE_NAME = 'class_distribution_before_sample.png'
    # Nomes específicos para os dois gráficos de análise final
    PLOT_AFTER_SHARED_SCALE_NAME = 'class_distribution_after_shared_scale.png'
    PLOT_AFTER_INDEPENDENT_SCALES_NAME = 'class_distribution_after_independent_scales.png'


    # --- PARÂMETROS DE AMOSTRAGEM ---
    # Defina a porcentagem do dataset a ser processada (ex: 0.05 para 5%).
    # Defina 1.0 para usar o dataset completo.
    SAMPLE_PERCENTAGE = 1.0
    RANDOM_SEED = 42  # Semente para garantir que a amostragem seja sempre a mesma

    # --- PARÂMETROS DE AUMENTO ---
    TRANSFORMATION_TYPE = 'SIMPLE'


# ===============================================================
# 2. DEFINIÇÃO DAS TRANSFORMAÇÕES
# ===============================================================

# --- TRANSFORMAÇÃO SIMPLES ---
simple_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=(256, 128), scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    # transforms.RandomGrayscale(p=0.2), # Readicionado para robustez
])

# --- TRANSFORMAÇÃO COMPLETA (OPCIONAL) ---
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
    """Copia um único arquivo de imagem da origem para o destino."""
    full_original_path, destination_path = paths
    if os.path.exists(full_original_path) and not os.path.exists(destination_path):
        shutil.copy(full_original_path, destination_path)
    return True


def process_and_augment_image(args):
    """Aplica a transformação a uma imagem, salva e retorna os dados para a nova linha."""
    original_row_dict, n_count, all_images_dir, augmented_only_dir, transform_pipeline = args
    original_img_name = os.path.basename(original_row_dict['image_path'])
    original_img_path = os.path.join(Config.BASE_PATH, original_row_dict['image_path'])
    try:
        image = Image.open(original_img_path).convert("RGB")
        augmented_image = transform_pipeline(image)
        base_name, ext = os.path.splitext(original_img_name)
        augmented_filename = f"{base_name}_aug_{n_count}{ext}"

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
    # --- 1. SETUP E CONFIGURAÇÃO ---
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    print(f"Usando o tipo de transformação: '{Config.TRANSFORMATION_TYPE}'")
    if Config.TRANSFORMATION_TYPE == 'SIMPLE':
        pil_augmentation_transform = simple_transform
    elif Config.TRANSFORMATION_TYPE == 'COMPLETE':
        pil_augmentation_transform = complete_transform
    else:
        raise ValueError(f"Tipo de transformação '{Config.TRANSFORMATION_TYPE}' inválido.")

    # --- 2. LEITURA E AMOSTRAGEM DO DATASET ---
    df_full = pd.read_csv(Config.TRAIN_CSV_PATH)
    if 0.0 < Config.SAMPLE_PERCENTAGE < 1.0:
        print("\n" + "=" * 65)
        print(f"Amostrando {Config.SAMPLE_PERCENTAGE * 100:.2f}% do dataset original (semente: {Config.RANDOM_SEED}).")
        df_sample = df_full.sample(frac=Config.SAMPLE_PERCENTAGE, random_state=Config.RANDOM_SEED).reset_index(drop=True)
        print(f"Dataset original: {len(df_full)} linhas. | Dataset amostrado: {len(df_sample)} linhas.")
        print("=" * 65 + "\n")
    else:
        print("\nProcessando o dataset completo (100%).\n")
        df_sample = df_full

    df_original = df_sample.copy()

    # --- 3. ANÁLISE INICIAL E PREPARAÇÃO DE PASTAS ---
    print("Analisando o desbalanceamento das classes na amostra...")
    counts_df = pd.DataFrame(
        [[col, str(val), count] for col in df_original.columns[1:] for val, count in df_original[col].value_counts().items()],
        columns=["Feature", "Class", "Count"]
    )
    g = sns.catplot(data=counts_df, kind="bar", x="Feature", y="Count", hue="Class", height=6, aspect=3)
    g.set_xticklabels(rotation=75, ha="right")
    g.fig.suptitle('Distribuição de Classes Antes do Aumento (Amostra)', y=1.03)
    g.savefig(os.path.join(Config.SAVE_DIR, Config.PLOT_BEFORE_NAME))
    plt.close()

    print("\n" + "=" * 65)
    print(" Contagem de dados por classe (antes do aumento de dados)".upper())
    print("=" * 65)
    summary_df = counts_df.pivot_table(index='Feature', columns='Class', values='Count', fill_value=0).reset_index()
    summary_df.columns.name = None
    if '1' not in summary_df: summary_df['1'] = 0
    if '0' not in summary_df: summary_df['0'] = 0
    summary_df = summary_df[['Feature', '1', '0']].rename(columns={'1': 'Com Atributo (Classe 1)', '0': 'Sem Atributo (Classe 0)'})
    print(summary_df.to_string(index=False))
    print("=" * 65 + "\n")

    augmented_only_dir = os.path.join(Config.SAVE_DIR, Config.AUGMENTED_ONLY_DIR_NAME)
    all_images_dir = os.path.join(Config.SAVE_DIR, Config.ALL_IMAGES_DIR_NAME)
    os.makedirs(augmented_only_dir, exist_ok=True)
    os.makedirs(all_images_dir, exist_ok=True)

    print("Copiando imagens originais (da amostra) para o diretório de trabalho...")
    copy_tasks = [ (os.path.join(Config.BASE_PATH, row['image_path']), os.path.join(all_images_dir, os.path.basename(row['image_path']))) for _, row in df_original.iterrows() ]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(copy_image, copy_tasks), total=len(copy_tasks), desc="Copiando originais"))

    # --- 4. PROCESSO DE AUMENTO DE DADOS ---
    print("\nIniciando o processo de balanceamento de classes (50/50) na amostra")
    generated_dfs_list = []
    features_to_process = df_original.columns[1:]
    for feature in features_to_process:
        value_counts = df_original[feature].value_counts()
        count_0, count_1 = value_counts.get(0, 0), value_counts.get(1, 0)
        if count_1 > count_0: minority_class_label, minority_count, majority_count = 0, count_0, count_1
        elif count_0 > count_1: minority_class_label, minority_count, majority_count = 1, count_1, count_0
        else: print(f"\nFeature '{feature}' já está balanceada ({count_0}/{count_1}). Pulando."); continue
        images_to_generate = majority_count - minority_count
        print(f"\nFeature '{feature}': Balanceando. Classe minoritária é '{minority_class_label}' ({minority_count} imgs).")
        print(f"-> Gerando {images_to_generate} novas imagens para igualar à classe majoritária ({majority_count} imgs).")
        source_images_df = df_original[df_original[feature] == minority_class_label]
        if len(source_images_df) == 0: continue
        source_rows = [row for _, row in source_images_df.iterrows()]
        generation_pool = (source_rows * (images_to_generate // len(source_rows) + 1))[:images_to_generate]
        random.shuffle(generation_pool)
        tasks_for_this_feature = []
        usage_counter = collections.Counter()
        for original_row in generation_pool:
            original_img_name = os.path.basename(original_row['image_path'])
            usage_counter[original_img_name] += 1
            n_count = usage_counter[original_img_name]
            args = (original_row.to_dict(), n_count, all_images_dir, augmented_only_dir, pil_augmentation_transform)
            tasks_for_this_feature.append(args)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(process_and_augment_image, tasks_for_this_feature), total=len(tasks_for_this_feature), desc=f"Gerando para '{feature}'"))
            new_rows = [row for row in results if row is not None]
        if new_rows: generated_dfs_list.append(pd.DataFrame(new_rows))

    # --- 5. FINALIZAÇÃO E ANÁLISE COM DOIS GRÁFICOS ---
    print("\nConsolidando o dataframe final...")
    final_df = pd.concat([df_original] + generated_dfs_list, ignore_index=True)

    print("Atualizando os caminhos (paths) das imagens no dataframe final...")
    final_df['image_path'] = final_df['image_path'].apply(
        lambda path: path if Config.SAVE_DIR in path else os.path.join(Config.SAVE_DIR, Config.ALL_IMAGES_DIR_NAME, os.path.basename(path)).replace("\\", "/")
    )

    final_csv_path = os.path.join(Config.SAVE_DIR, Config.FINAL_CSV_NAME)
    final_df.to_csv(final_csv_path, index=False)
    print(f"\nProcesso concluído! Novo CSV salvo em: '{final_csv_path}'")
    print(f"Dataset inicial (amostra): {len(df_original)} imagens.")
    print(f"Dataset final (balanceado): {len(final_df)} imagens.")

    print("\nAnalisando a distribuição das classes APÓS o balanceamento...")
    final_counts_df = pd.DataFrame(
        [[col, str(val), count] for col in final_df.columns[1:] for val, count in final_df[col].astype(int).value_counts().items()],
        columns=["Feature", "Class", "Count"]
    )

    # --- GRÁFICO 1: ESCALAS INDEPENDENTES ---
    print("Gerando gráfico final com ESCALAS INDEPENDENTES por atributo...")
    g_independent = sns.catplot(
        data=final_counts_df,
        kind='bar',
        x='Class',
        y='Count',
        hue='Class',
        col='Feature',
        col_wrap=6,
        sharey=False, # Opção chave para escalas independentes
        height=3,
        aspect=1.2
    )
    g_independent.fig.suptitle('Distribuição Pós-Balanceamento (Escalas Independentes por Atributo)', y=1.03, size=16)
    g_independent.set_titles("{col_name}")
    g_independent.set_axis_labels("Classe", "Contagem")
    plot_path_independent = os.path.join(Config.SAVE_DIR, Config.PLOT_AFTER_INDEPENDENT_SCALES_NAME)
    g_independent.savefig(plot_path_independent)
    plt.close()
    print(f"Gráfico com escalas independentes salvo em: {plot_path_independent}")

    # --- GRÁFICO 2: ESCALA GERAL ---
    print("Gerando gráfico final com ESCALA GERAL para todos os atributos...")
    g_after_shared = sns.catplot(data=final_counts_df, kind="bar", x="Feature", y="Count", hue="Class", height=6, aspect=3)
    g_after_shared.set_xticklabels(rotation=75, ha="right")
    g_after_shared.fig.suptitle('Distribuição de Classes Após o Balanceamento (Escala Geral Compartilhada)', y=1.03, size=16)
    plot_path_shared = os.path.join(Config.SAVE_DIR, Config.PLOT_AFTER_SHARED_SCALE_NAME)
    g_after_shared.savefig(plot_path_shared)
    plt.close()
    print(f"Gráfico com escala geral salvo em: {plot_path_shared}")

    # --- Tabela de Resumo Final ---
    print("\n" + "=" * 65)
    print(" Contagem de dados por classe (após o balanceamento)".upper())
    print("=" * 65)
    final_summary_df = final_counts_df.pivot_table(index='Feature', columns='Class', values='Count', fill_value=0).reset_index()
    final_summary_df.columns.name = None
    if '1' not in final_summary_df: final_summary_df['1'] = 0
    if '0' not in final_summary_df: final_summary_df['0'] = 0
    final_summary_df = final_summary_df[['Feature', '1', '0']].rename(columns={'1': 'Com Atributo (Classe 1)', '0': 'Sem Atributo (Classe 0)'})
    print(final_summary_df.to_string(index=False))
    print("=" * 65 + "\n")