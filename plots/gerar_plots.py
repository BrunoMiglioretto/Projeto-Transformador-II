#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import argparse  # <-- Importa a biblioteca de argumentos

# Configura o backend do matplotlib para 'Agg' (não interativo)
matplotlib.use('Agg')

# --- Configurações Estáticas ---
# Define os nomes das colunas
col_names = ['filename', 'upper_color', 'lower_color', 'gender', 'bag', 'hat']

# Mapeamento de cores
color_map = {
    1: 'black', 2: 'blue', 3: 'brown', 4: 'gray', 5: 'green', 
    6: 'orange', 7: 'pink', 8: 'purple', 9: 'red', 10: 'white', 11: 'yellow'
}

# --- Função Auxiliar para Plotagem (Sem Alterações) ---
def plot_on_ax(ax, data_counts, title, plot_color, mapping=None):
    """
    Desenha um gráfico de barras em um eixo (ax) do matplotlib.
    """
    try:
        data_counts_mapped = data_counts.copy()
        
        # Se um mapeamento for fornecido (para cores)
        if mapping:
            data_counts_mapped.index = data_counts_mapped.index.map(lambda x: mapping.get(int(x), x))
            # Garante a ordem decrescente pela contagem
            data_counts_mapped = data_counts_mapped.sort_values(ascending=False)
        else:
            # Para 'gender', 'bag', 'hat', é melhor ordenar pelo índice (0, 1)
            data_counts_mapped = data_counts_mapped.sort_index()

        # Plota no eixo fornecido
        data_counts_mapped.plot(kind='bar', color=plot_color, edgecolor='black', ax=ax)
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Classe', fontsize=10)
        ax.set_ylabel('Contagem', fontsize=10)
        # Ajusta a rotação dos rótulos do eixo x
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right' if mapping else 'center')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Adiciona os rótulos de dados (contagem) acima de cada barra
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', 
                        xytext=(0, 9), 
                        textcoords='offset points',
                        fontsize=9)
    except Exception as e:
        print(f"Erro ao plotar no eixo '{title}': {e}")

# --- Função Principal de Processamento ---
def main(train_path, test_path):
    """
    Função principal que executa a lógica de carregamento e plotagem.
    """
    # Lista para armazenar os nomes das imagens geradas
    generated_images = []

    # Lista de arquivos para processar (agora baseada nos argumentos)
    datasets_info = [
        {'file_path': train_path, 'label': 'Treinamento', 'color': 'skyblue'},
        {'file_path': test_path, 'label': 'Validação', 'color': 'salmon'}
    ]

    # --- Processamento Principal ---
    for dataset in datasets_info:
        file_path = dataset['file_path']
        label = dataset['label']
        color = dataset['color']
        
        print(f"\n--- Processando Arquivo: {file_path} ({label}) ---")
        
        try:
            # Adiciona verificação se o arquivo existe
            if not os.path.exists(file_path):
                print(f"ERRO: Arquivo não encontrado: {file_path}")
                continue # Pula para o próximo dataset

            df = pd.read_csv(file_path, header=None, names=col_names)
            print(f"Dados de {label} carregados. Total de {len(df)} linhas.")

            # --- Gráfico 1: Gender, Bag, Hat (Juntos) ---
            print(f"Gerando gráfico combinado para 'gender', 'bag', 'hat' ({label})...")
            try:
                # Cria uma figura com 1 linha e 3 colunas (subplots)
                fig1, axes1 = plt.subplots(1, 3, figsize=(20, 7))
                
                # Calcula as contagens
                gender_counts = df['gender'].value_counts()
                bag_counts = df['bag'].value_counts()
                hat_counts = df['hat'].value_counts()
                
                # Plota cada um em seu respectivo eixo (ax)
                plot_on_ax(axes1[0], gender_counts, 'Gender', color)
                plot_on_ax(axes1[1], bag_counts, 'Bag', color)
                plot_on_ax(axes1[2], hat_counts, 'Hat', color)
                
                fig1.suptitle(f'Distribuição (Gender, Bag, Hat) - {label}', fontsize=18)
                fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
                
                img_file1 = f"{label.lower()}_gbh_distribution.png"
                fig1.savefig(img_file1)
                plt.close(fig1) 
                generated_images.append(img_file1)
                print(f"Gráfico salvo: {img_file1}")

            except Exception as e:
                print(f"Erro ao gerar gráfico 'gbh' para {label}: {e}")

            # --- Gráfico 2: Upper Color, Lower Color (Juntos) ---
            print(f"Gerando gráfico combinado para 'upper_color', 'lower_color' ({label})...")
            try:
                # Cria uma figura com 1 linha e 2 colunas
                fig2, axes2 = plt.subplots(1, 2, figsize=(22, 8))
                
                # Calcula as contagens
                upper_counts = df['upper_color'].value_counts()
                lower_counts = df['lower_color'].value_counts()
                
                plot_on_ax(axes2[0], upper_counts, 'Upper Color', color, mapping=color_map)
                plot_on_ax(axes2[1], lower_counts, 'Lower Color', color, mapping=color_map)
                
                fig2.suptitle(f'Distribuição (Upper Color, Lower Color) - {label}', fontsize=18)
                fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
                
                img_file2 = f"{label.lower()}_colors_distribution.png"
                fig2.savefig(img_file2)
                plt.close(fig2) 
                generated_images.append(img_file2)
                print(f"Gráfico salvo: {img_file2}")

            except Exception as e:
                print(f"Erro ao gerar gráfico 'colors' para {label}: {e}")

        except Exception as e:
            print(f"Erro ao processar o arquivo {file_path}: {e}")

    print(f"\n--- Processamento Concluído ---")
    print(f"Total de {len(generated_images)} imagens geradas: {', '.join(generated_images)}")

# --- Ponto de Entrada do Script ---
if __name__ == "__main__":
    # Configura o parser de argumentos
    parser = argparse.ArgumentParser(description='Gera gráficos de distribuição de features dos datasets.')
    parser.add_argument('--train', type=str, required=True, help='Caminho para o arquivo de treinamento (ex: training_set.txt)')
    parser.add_argument('--test', type=str, required=True, help='Caminho para o arquivo de validação/teste (ex: validation_set.txt)')
    
    # Analisa os argumentos da linha de comando
    args = parser.parse_args()
    
    # Chama a função principal com os caminhos
    main(args.train, args.test)