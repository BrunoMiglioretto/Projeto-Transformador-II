import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from tqdm import tqdm
import random
import shutil
import concurrent.futures
import collections
from PIL import Image
from torchvision import transforms

# ===============================================================
# 1. CLASSE DE CONFIGURAÇÃO GLOBAL
# ===============================================================
class GlobalConfig:
    """
    Configurações globais que se aplicam a todos os datasets a serem gerados.
    """
    # --- CAMINHOS DE ENTRADA ---
    BASE_PATH = "/home/gdaudt/Área de trabalho/Projeto-Transformador-II/data"
    TRAIN_CSV_PATH = os.path.join(BASE_PATH, "annotations/phase1/train/train.csv")

    # --- DIRETÓRIO PRINCIPAL DE SAÍDA (Onde as pastas dos datasets serão criadas) ---
    MAIN_OUTPUT_DIR = 'generated_balanced_datasets'
    
    # --- PARÂMETROS DE AMOSTRAGEM ---
    # Defina a porcentagem do dataset a ser processada (ex: 0.1 para 10%).
    # Defina 1.0 para usar o dataset completo.
    SAMPLE_PERCENTAGE = 1.0 # Usando 5% para um teste rápido
    RANDOM_SEED = 42


# ===============================================================
# 2. DEFINIÇÃO DAS "RECEITAS" DE DATA AUGMENTATION
# ===============================================================

# --- Receita 1: Transformações de Cor e Recorte Simples ---
augmentation_recipe_1 = {
    "name": "crop_color_jitter",
    "number": 1,
    "description": (
        "Técnicas focadas em recorte, inversão e variação de cor.\n\n"
        "1. RandomResizedCrop: Recorta uma área aleatória da imagem e redimensiona.\n"
        "   - size=(256, 128)\n"
        "   - scale=(0.8, 1.0)\n"
        "2. RandomHorizontalFlip: Inverte a imagem horizontalmente com 50% de chance.\n"
        "3. ColorJitter: Altera aleatoriamente o brilho, contraste, saturação e matiz.\n"
        "   - brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1"
    ),
    "pipeline": transforms.Compose([
        transforms.RandomResizedCrop(size=(256, 128), scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ])
}

# --- Receita 2: Transformações Geométricas Agressivas ---
augmentation_recipe_2 = {
    "name": "geometric_affine_perspective",
    "number": 2,
    "description": (
        "Técnicas focadas em distorções geométricas severas.\n\n"
        "1. RandomHorizontalFlip: Inverte a imagem horizontalmente com 50% de chance.\n"
        "2. RandomAffine: Aplica rotação, translação e cisalhamento.\n"
        "   - degrees=15, translate=(0.1, 0.1), shear=10\n"
        "3. RandomPerspective: Aplica uma distorção de perspectiva aleatória.\n"
        "   - distortion_scale=0.4, p=0.6"
    ),
    "pipeline": transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        transforms.RandomPerspective(distortion_scale=0.4, p=0.6),
        transforms.Resize((256, 128)) # Garantir o tamanho final
    ])
}

# --- Receita 3: Transformações de Filtros e Qualidade de Imagem ---
augmentation_recipe_3 = {
    "name": "photometric_filters",
    "number": 3,
    "description": (
        "Técnicas focadas em filtros, ruído e qualidade fotométrica.\n\n"
        "1. RandomApply(GaussianBlur): Aplica desfoque gaussiano com 50% de chance.\n"
        "   - kernel_size=3, sigma=(0.1, 2.0)\n"
        "2. RandomAdjustSharpness: Ajusta a nitidez da imagem.\n"
        "   - sharpness_factor=2.0, p=0.5\n"
        "3. RandomAutocontrast: Aplica auto-contraste com 50% de chance.\n"
        "4. RandomGrayscale: Converte a imagem para escala de cinza com 25% de chance.\n"
    ),
    "pipeline": transforms.Compose([
        transforms.Resize((256, 128)), # Redimensionar primeiro para manter consistência
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),
        transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=0.5),
        transforms.RandomAutocontrast(p=0.5),
        transforms.RandomGrayscale(p=0.25)
    ])
}

# Lista de todas as receitas que serão processadas
ALL_RECIPES = [
    augmentation_recipe_1,
    augmentation_recipe_2,
    augmentation_recipe_3,
]

# ===============================================================
# 3. FUNÇÕES DE TRABALHO (WORKERS)
# ===============================================================
def copy_image(paths):
    """Copia um único arquivo de imagem da origem para o destino."""
    full_original_path, destination_path = paths
    if os.path.exists(full_original_path) and not os.path.exists(destination_path):
        shutil.copy(full_original_path, destination_path)
    return True


def process_and_augment_image(args):
    """Aplica a transformação a uma imagem, salva e retorna os dados para a nova linha."""
    original_row_dict, n_count, augmented_only_dir, transform_pipeline = args
    original_img_name = os.path.basename(original_row_dict['image_path'])
    original_img_path = os.path.join(GlobalConfig.BASE_PATH, original_row_dict['image_path'])
    try:
        image = Image.open(original_img_path).convert("RGB")
        augmented_image = transform_pipeline(image)
        base_name, ext = os.path.splitext(original_img_name)
        augmented_filename = f"{base_name}_aug_{n_count}{ext}"
        
        save_path = os.path.join(augmented_only_dir, augmented_filename)
        augmented_image.save(save_path)

        new_row = original_row_dict.copy()
        new_row['image_path'] = augmented_filename # Apenas o nome do arquivo por enquanto
        new_row['is_augmented'] = True
        return new_row
    except Exception:
        return None


# ===============================================================
# 4. FUNÇÃO PRINCIPAL DE GERAÇÃO DE DATASET
# ===============================================================
def generate_balanced_dataset(recipe: dict, base_df: pd.DataFrame):
    """
    Executa todo o pipeline para gerar um dataset balanceado para uma dada "receita" de augmentation.
    """
    config_name = recipe['name']
    number = recipe['number']
    transform_pipeline = recipe['pipeline']
    techniques_description = recipe['description']
    
    print("\n" + "#" * 80)
    print(f"# INICIANDO GERAÇÃO DO DATASET: '{config_name}' (dataset{number})")
    print("#" * 80 + "\n")

    # --- 1. Setup dos diretórios específicos para este dataset ---
    dataset_dir = os.path.join(GlobalConfig.MAIN_OUTPUT_DIR, f'dataset{number}_{config_name}')
    original_images_dir = os.path.join(dataset_dir, 'original_images')
    augmented_images_dir = os.path.join(dataset_dir, 'augmented_images')
    initial_analysis_dir = os.path.join(dataset_dir, 'initial_sample_analysis') # <-- NOVO
    os.makedirs(original_images_dir, exist_ok=True)
    os.makedirs(augmented_images_dir, exist_ok=True)
    os.makedirs(initial_analysis_dir, exist_ok=True) # <-- NOVO

    # --- 2. Análise Inicial da amostra (salva dentro da pasta do dataset) ---
    print("Analisando o desbalanceamento inicial da amostra...")
    initial_counts_df = pd.DataFrame(
        [[col, str(val), count] for col in base_df.columns[1:] for val, count in base_df[col].value_counts().items()],
        columns=["Feature", "Class", "Count"]
    )
    g_before = sns.catplot(data=initial_counts_df, kind="bar", x="Feature", y="Count", hue="Class", height=6, aspect=3)
    g_before.set_xticklabels(rotation=75, ha="right")
    g_before.fig.suptitle('Distribuição de Classes Antes do Aumento (Amostra Original)', y=1.03)
    plot_before_path = os.path.join(initial_analysis_dir, 'class_distribution_before_augmentation.png')
    g_before.savefig(plot_before_path)
    plt.close()
    print(f"Gráfico da distribuição inicial salvo em: '{plot_before_path}'")
    
    # Tabela de resumo inicial
    print("\n" + "=" * 65)
    print(" Contagem de dados por classe (antes do aumento de dados)".upper())
    print("=" * 65)
    summary_df = initial_counts_df.pivot_table(index='Feature', columns='Class', values='Count', fill_value=0).reset_index()
    summary_df.columns.name = None
    if '1' not in summary_df: summary_df['1'] = 0
    if '0' not in summary_df: summary_df['0'] = 0
    summary_df = summary_df[['Feature', '1', '0']].rename(columns={'1': 'Com Atributo (1)', '0': 'Sem Atributo (0)'})
    print(summary_df.to_string(index=False))
    print("=" * 65 + "\n")

    # --- 3. Copiando imagens originais da amostra ---
    print(f"Copiando {len(base_df)} imagens originais para '{original_images_dir}'...")
    copy_tasks = [
        (os.path.join(GlobalConfig.BASE_PATH, row['image_path']), os.path.join(original_images_dir, os.path.basename(row['image_path'])))
        for _, row in base_df.iterrows()
    ]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(copy_image, copy_tasks), total=len(copy_tasks), desc="Copiando originais"))
    
    # --- 4. Processo de Aumento de Dados (Balanceamento) ---
    print("\nIniciando processo de balanceamento de classes (50/50)...")
    generated_dfs_list = []
    features_to_process = base_df.columns[1:]
    
    for feature in features_to_process:
        value_counts = base_df[feature].value_counts()
        count_0, count_1 = value_counts.get(0, 0), value_counts.get(1, 0)

        if count_0 == count_1:
            print(f"\nFeature '{feature}' já está balanceada ({count_0}/{count_1}). Pulando.")
            continue
        
        minority_class_label = 1 if count_1 < count_0 else 0
        minority_count = min(count_1, count_0)
        majority_count = max(count_1, count_0)
        
        images_to_generate = majority_count - minority_count
        
        print(f"\nFeature '{feature}': Balanceando. Classe minoritária é '{minority_class_label}' ({minority_count} imgs).")
        print(f"-> Gerando {images_to_generate} novas imagens para igualar à classe majoritária ({majority_count} imgs).")
        
        source_images_df = base_df[base_df[feature] == minority_class_label]
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
            args = (original_row.to_dict(), n_count, augmented_images_dir, transform_pipeline)
            tasks_for_this_feature.append(args)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(process_and_augment_image, tasks_for_this_feature), total=len(tasks_for_this_feature), desc=f"Gerando para '{feature}'"))
            new_rows = [row for row in results if row is not None]

        if new_rows:
            generated_dfs_list.append(pd.DataFrame(new_rows))

    # --- 5. Consolidação e Finalização ---
    print("\nConsolidando o dataframe final...")
    df_original_copy = base_df.copy()
    df_original_copy['is_augmented'] = False
    df_original_copy['image_path'] = df_original_copy['image_path'].apply(os.path.basename)

    final_df = pd.concat([df_original_copy] + generated_dfs_list, ignore_index=True)

    def update_path(row):
        if row['is_augmented']:
            return os.path.join('augmented_images', row['image_path']).replace("\\", "/")
        else:
            return os.path.join('original_images', row['image_path']).replace("\\", "/")

    final_df['image_path'] = final_df.apply(update_path, axis=1)
    final_df.drop(columns=['is_augmented'], inplace=True)

    final_csv_path = os.path.join(dataset_dir, 'dataset_labels.csv')
    final_df.to_csv(final_csv_path, index=False)
    
    with open(os.path.join(dataset_dir, 'techniques_used.txt'), 'w') as f:
        f.write(techniques_description)

    print(f"\nDataset '{config_name}' concluído!")
    print(f"  - CSV salvo em: '{final_csv_path}'")
    print(f"  - Total de imagens originais: {len(df_original_copy)}")
    print(f"  - Total de imagens no dataset final: {len(final_df)}")

    # --- 6. Análise Gráfica Pós-Balanceamento ---
    print("\nGerando gráficos de análise pós-balanceamento...")
    final_counts_df = pd.DataFrame(
        [[col, str(val), count] for col in final_df.columns[1:] for val, count in final_df[col].astype(int).value_counts().items()],
        columns=["Feature", "Class", "Count"]
    )
    
    g_independent = sns.catplot(data=final_counts_df, kind='bar', x='Class', y='Count', hue='Class',
                                col='Feature', col_wrap=6, sharey=False, height=3, aspect=1.2)
    g_independent.fig.suptitle(f'Pós-Balanceamento ({config_name}) - Escalas Independentes', y=1.03, size=16)
    g_independent.set_titles("{col_name}")
    g_independent.set_axis_labels("Classe", "Contagem")
    plot_path_ind = os.path.join(dataset_dir, 'distribution_after_independent_scales.png')
    g_independent.savefig(plot_path_ind)
    plt.close()

    g_shared = sns.catplot(data=final_counts_df, kind="bar", x="Feature", y="Count", hue="Class", height=6, aspect=3)
    g_shared.set_xticklabels(rotation=75, ha="right")
    g_shared.fig.suptitle(f'Pós-Balanceamento ({config_name}) - Escala Geral', y=1.03, size=16)
    plot_path_shared = os.path.join(dataset_dir, 'distribution_after_shared_scale.png')
    g_shared.savefig(plot_path_shared)
    plt.close()

    print(f"Gráficos de análise salvos em: '{dataset_dir}'")
    
    print("\n" + "=" * 65)
    print(f" Contagem Final para o Dataset '{config_name}' ".upper())
    print("=" * 65)
    final_summary_df = final_counts_df.pivot_table(index='Feature', columns='Class', values='Count', fill_value=0).reset_index()
    final_summary_df.columns.name = None
    if '1' not in final_summary_df: final_summary_df['1'] = 0
    if '0' not in final_summary_df: final_summary_df['0'] = 0
    final_summary_df = final_summary_df[['Feature', '1', '0']].rename(columns={'1': 'Com Atributo (1)', '0': 'Sem Atributo (0)'})
    print(final_summary_df.to_string(index=False))
    print("=" * 65 + "\n")


# ===============================================================
# 5. SCRIPT PRINCIPAL DE EXECUÇÃO
# ===============================================================
if __name__ == "__main__":
    # --- 1. Setup inicial e leitura do dataset ---
    os.makedirs(GlobalConfig.MAIN_OUTPUT_DIR, exist_ok=True)
    
    df_full = pd.read_csv(GlobalConfig.TRAIN_CSV_PATH)

    # --- 2. Amostragem (se aplicável) ---
    if 0.0 < GlobalConfig.SAMPLE_PERCENTAGE < 1.0:
        print("=" * 65)
        print(f"Amostrando {GlobalConfig.SAMPLE_PERCENTAGE * 100:.2f}% do dataset (semente: {GlobalConfig.RANDOM_SEED}).")
        df_sample = df_full.sample(frac=GlobalConfig.SAMPLE_PERCENTAGE, random_state=GlobalConfig.RANDOM_SEED).reset_index(drop=True)
        print(f"Dataset original: {len(df_full)} | Dataset amostrado: {len(df_sample)}.")
        print("=" * 65 + "\n")
    else:
        print("\nProcessando o dataset completo (100%).\n")
        df_sample = df_full

    # --- 3. Loop de Geração dos Datasets ---
    # A análise inicial agora é feita dentro desta função para cada dataset
    for recipe in ALL_RECIPES:
        generate_balanced_dataset(recipe, base_df=df_sample)

    print("\nTODOS OS PROCESSOS FORAM CONCLUÍDOS COM SUCESSO!")