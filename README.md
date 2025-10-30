# Projeto-Transformador-II — README de Artefatos, Resultados e Reprodutibilidade

> **Escopo:** classificação em imagens para os atributos `upper_color`, `lower_color`, `gender`, `bag` e `hat` do dataset **PAR2025**. Os experimentos avaliaram arquiteturas **EfficientNet-B0**, **MobileNetV2** e **SwinV2-T**, com e sem **transfer learning**, e com **data augmentation**.

## Resultados dos Experimentos

Devido ao tamanho dos arquivos de saída — especialmente os **pesos dos modelos (`.pth`)**, figuras de treinamento e matrizes de confusão —, os resultados completos **não foram incluídos diretamente neste repositório**, uma vez que o GitHub impõe um limite de **100 MB por arquivo** e **2 GB por pacote de push**.

Todos os **artefatos experimentais completos** (modelos, relatórios CSV e gráficos) estão disponíveis no seguinte link público:

> 📂 **[Acesse aqui os resultados completos no Google Drive](https://drive.google.com/drive/folders/1MN4NYTUKyvus_9fwL0zg_hl0edMl5mJh?usp=sharing)**

**Conteúdo do link:**

* Checkpoints dos modelos (`best_model.pth`) treinados em cada configuração (com/sem transfer learning, com/sem data augmentation).
* Gráficos de acurácia e matrizes de confusão por arquitetura e tarefa.
* Relatórios consolidados (`general_summary_report.csv` e `ensemble_summary_report.csv`).
* Logs de treinamento (`terminal_output.txt`) e parâmetros utilizados.

> **Observação:** os arquivos `.pth` e demais saídas são grandes (~100 MB cada) e foram movidos para o Google Drive apenas para **garantir reprodutibilidade** e **armazenamento persistente**, conforme práticas comuns em projetos de Deep Learning.

---


## 1\) Estrutura de pastas (nível superior)

  * `analysis/`

      * **`par.ipynb`**: notebook de exploração, preparação, treino/avaliação e análise do **PAR2025**.
        **Pré-requisitos:** `./data/PAR2025/training_set.txt` e `./data/PAR2025/validation_set.txt` (gerados pelo `download_datasets.py`).
        **Saídas típicas:** figuras de curvas/discordâncias e tabelas consolidadas (salvas em `plots/` e `modelos/{...}/general_summary_report.csv`).

  * `data_agumentation/`

      * **`data_agumentation-50-50.py`**: script de *data augmentation* com política **50/50** (gera amostras **apenas** para a classe minoritária de cada atributo binário, até equilibrar \~50/50). Centraliza transformações usadas durante os treinos experimentais.
        **Como usar:** não recebe CLI; **edite os parâmetros** no topo do arquivo (ex.: `BASE_PATH`, `TRAIN_CSV_PATH`, `SAVE_DIR`, `SAMPLE_PERCENTAGE`, `TRANSFORMATION_TYPE = {SIMPLE|COMPLETE}`, `RANDOM_SEED`).
        **O que ele faz:**
          * Lê o CSV de treino e cria um **dataset balanceado** (gera imagens sintéticas com `torchvision.transforms`).
          * Salva **apenas as sintetizadas** em `.../augmented_only/` e **todas** (originais + sintetizadas) em `.../all_images/`.
          * Exporta **CSV final** com caminhos atualizados (`dataset_balanceado_amostra.csv`) e **gráficos** de distribuição antes/depois.

  * `modelos/`

      * **`BASELINE-com_transfer_leaning/`**: resultados dos treinos **com** transferência de aprendizado (pesos pré-treinados). Estruturado por **tarefa** (`bag/`, `gender/`, `hat/`, `lower_color/`, `upper_color/`) e, dentro de cada tarefa, por **arquitetura** (`EfficientNet-B0/`, `MobileNetV2/`, `SwinV2-T/`). Cada arquitetura contém:

          * `accuracy_plot.png`: curva(s) de acurácia por época (treino/val).
          * `confusion_matrix.png`: matriz de confusão no conjunto de validação/teste (conforme configurado).
          * `terminal_output.txt`: *stdout* completo do treino (épocas, métricas, tempos, etc.).

      * **`BASELINE-sem_transfer_learning/`**: mesma estrutura acima, porém **sem** pesos pré-treinados (treino do zero).

      * **`output-com_transfer_learning-DataAgumentationDuranteTreinamento/`**: saídas adicionais de experimentos **com transfer learning** + **data augmentation aplicada em tempo de treino**. Mantém o mesmo padrão por tarefa/arquitetura (figuras, checkpoints e logs).

      * **`output-sem_transfer_learning-DataAgumentationDuranteTreinamento/`**: saídas equivalentes **sem transfer learning** + **data augmentation durante o treino**.

      * **`ensemble_summary_report.csv`**: tabela-resumo com métricas agregadas por **ensembles** (hard/soft voting).

      * **`general_summary_report.csv`**: tabela-resumo **geral** por execução (tarefa, arquitetura, *seed*, *augmentation*, acurácia, F1, *loss* mínima, época do melhor, etc.).

      * **`train_models.py`**: *runner* principal de treinamento (varre tarefas e arquiteturas).
        **Entradas esperadas:** `--data-root PATH` apontando para o diretório que contém `training_set.txt` e `validation_set.txt` (ex.: `./data/PAR2025`).
        **Parâmetros (linha de comando):**

          * `--data-root PATH` → raiz do dataset (obrigatório).
          * `--epochs INT` → número de épocas (padrão do script: 10).
          * `--no-pretrained` → **desativa** transfer learning (por padrão, **usa** pesos pré-treinados).
            **Padrões internos (editáveis no arquivo):** `IMG_SIZE=200`, `BATCH_SIZE=32`, `LEARNING_RATE=1e-4`, `RANDOM_STATE=42`, `MODELS_TO_TRAIN=['MobileNetV2','EfficientNet-B0','SwinV2-T']`.
            **Pré-processamento:** normalização ImageNet; treino com `RandomResizedCrop`, `HorizontalFlip(0.5)` e `RandomErasing(0.4)`; validação/teste com `Resize`.
            **Estratégia de *split*:** reparte o arquivo `validation_set.txt` em **val** e **teste** (50/50) **por ID de imagem original** (removendo sufixos `_aug_`).
            **Saídas:**

        <!-- end list -->

        ```
        ./output-{com|sem}_transfer_learning-DataAgumentationDuranteTreinamento/
          {bag|gender|hat|lower_color|upper_color}/{MobileNetV2|EfficientNet-B0|SwinV2-T}/
            ├─ best_model.pth
            ├─ accuracy_plot.png
            ├─ confusion_matrix.png
            └─ terminal_output.txt
          ├─ general_summary_report.csv
          └─ ensemble_summary_report.csv
        ```

        *(Se rodar o script a partir de `modelos/`, as pastas sairão em `modelos/output-*`, como na sua árvore.)*

  * `plots/`

      * **`gerar_plots_PAR2025.py`**: script utilitário para gerar gráficos de distribuição de classes e outras figuras a partir dos **TXTs do PAR2025**.
        **Parâmetros (CLI):**
            **Saídas geradas (no diretório corrente):**
          * **`PAR2025_treinamento_colors_distribution.png`** e **`PAR2025_validação_colors_distribution.png`** (cores upper/lower).
          * **`PAR2025_treinamento_gbh_distribution.png`** e **`PAR2025_validação_gbh_distribution.png`** (`gender`, `bag`, `hat`).
   
      * **`PAR2025_treinamento_colors_distribution.png`** e **`PAR2025_validação_colors_distribution.png`**: distribuição de cores (upper/lower) nos conjuntos de treino e validação.
      * **`PAR2025_treinamento_gbh_distribution.png`** e **`PAR2025_validação_gbh_distribution.png`**: distribuição de `gender`, `bag`, `hat` nos conjuntos de treino e validação.

  * `download_datasets.py`
    Script para **baixar/organizar** o conjunto de dados **PAR2025** e os templates de submissão.
    **Parâmetro (CLI):** `--data-dir PATH` (padrão: `./data`).
    **O que ele baixa/prepara:**

      * **PAR2025** em `{DATA_DIR}/PAR2025/` (gera os TXTs `training_set.txt` e `validation_set.txt` e organiza imagens).
      * **Templates de submissão** em `./submission_templates/`.
        **Estrutura criada:**

    <!-- end list -->

    ```
    {DATA_DIR}/
    └── PAR2025/
        ├── training_set/
        └── validation_set/
    ./submission_templates/
    ```

  * `environment.yml`
    Definição do ambiente Conda (Python + libs de visão computacional/ML).
    **Principais dependências esperadas:** PyTorch/torchvision, scikit-learn, pandas, numpy, matplotlib, Pillow, tqdm, **gdown** (para downloads), e utilitários de leitura de imagens.

-----

## 2\) Organização dentro de `modelos/`

**Padrão de organização (idêntico em todas as variações):**

```
modelos/
├── {VARIANTE}/
│   ├── bag/
│   │   ├── EfficientNet-B0/
│   │   │   ├── accuracy_plot.png
│   │   │   ├── best_model.pth
│   │   │   ├── confusion_matrix.png
│   │   │   └── terminal_output.txt
│   │   ├── MobileNetV2/
│   │   └── SwinV2-T/
│   ├── gender/
│   ├── hat/
│   ├── lower_color/
│   ├── upper_color/
│   ├── ensemble_summary_report.csv
│   └── general_summary_report.csv
└── train_models.py
```

Onde `{VARIANTE}` $\in$ {`BASELINE-com_transfer_leaning`, `BASELINE-sem_transfer_learning`, `output-com_transfer_learning-DataAgumentationDuranteTreinamento`, `output-sem_transfer_learning-DataAgumentationDuranteTreinamento`}.

**Arquivos por arquitetura (em cada tarefa):**

  * **`best_model.pth`**: checkpoint com os melhores pesos.
  * **`accuracy_plot.png`**: desempenho por época (treino/val).
  * **`confusion_matrix.png`**: erros/acertos por classe.
  * **`terminal_output.txt`**: log literal da execução (hints de parâmetros, lr, tempos, etc.).

**Relatórios CSV:**

  * **`general_summary_report.csv`**: uma linha por execução (tarefa $\times$ arquitetura $\times$ variação), com métricas-chave e metadados.
  * **`ensemble_summary_report.csv`**: agregações/combinações quando aplicável (médias, *voting*, *stacking*, etc.).

-----

## 3\) Como reproduzir

1.  **Criar ambiente** (Conda):

    ```bash
    conda env create -f environment.yml
    conda activate projeto-transformador-ii
    ```

2.  **Baixar dados** (PAR2025 e templates):

    ```bash
    python download_datasets.py --data-dir ./data
    ```

3.  **Executar treinamentos**:

      * **Com transfer learning (padrão):**

        ```bash
        python modelos/train_models.py \
          --data-root ./data/PAR2025 \
          --epochs 10
        ```

      * **Sem transfer learning (treino do zero):**

        ```bash
        python modelos/train_models.py \
          --data-root ./data/PAR2025 \
          --epochs 10 \
          --no-pretrained
        ```