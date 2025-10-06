# Documentação do Script de Aumento de Dados

## Visão Geral

Este script realiza o **aumento de dados** (*Data Augmentation*) em um dataset de imagens para combater o **desbalanceamento de classes**. Ele gera novas imagens sintéticas para atributos com poucos exemplos, criando um dataset mais equilibrado para o treinamento de modelos de Machine Learning. 🎯

A execução é otimizada com **processamento paralelo** para acelerar a criação de imagens e gera relatórios visuais sobre a distribuição das classes.

---

## Pipeline de Execução

O script opera em uma sequência lógica, utilizando as configurações definidas na classe `Config`. Com as definições atuais (`AUGMENTATION_MODE = 'BALANCE'` e `TRANSFORMATION_TYPE = 'SIMPLE'`), o fluxo é o seguinte:

1.  **Análise Inicial**: O script lê o `TRAIN_CSV_PATH`, calcula a distribuição de imagens para cada atributo e exibe uma tabela e um gráfico de barras (`class_distribution_before.png`) para mostrar o desbalanceamento inicial.

2.  **Preparação do Ambiente**: As pastas de saída, como `all_images`, são criadas no diretório `SAVE_DIR`. Todas as imagens originais são copiadas para a pasta `all_images` para servirem de base para o aumento.

3.  **Cálculo de Tarefas (Modo `BALANCE`)**: O script analisa cada atributo do dataset. Para um atributo, se o número de imagens for **menor** que o `TARGET_IMAGES_PER_CLASS` (55.000), ele calcula a diferença necessária para atingir essa meta. Atributos que já possuem 55.000 ou mais imagens são ignorados.

4.  **Aumento de Dados (Modo `SIMPLE`)**: Para os atributos que precisam de mais imagens, o script aplica o pipeline de transformações `simple_transform`. Este pipeline realiza três operações básicas:
    * `RandomResizedCrop`: Recorta uma área aleatória da imagem e a redimensiona.
    * `RandomHorizontalFlip`: Inverte a imagem horizontalmente (espelha).
    * `ColorJitter`: Altera sutilmente o brilho, contraste e saturação.

    Este processo é executado em **paralelo**, utilizando múltiplos núcleos de CPU para gerar as novas imagens de forma eficiente. As imagens criadas são salvas, e novas entradas são preparadas para o CSV final.

5.  **Finalização**: O script consolida as informações das imagens originais e das novas imagens aumentadas em um único arquivo CSV (`dataset_aumentado.csv`), com todos os caminhos de imagem apontando para a pasta `all_images`.

---

## Bibliotecas Utilizadas 📚

* **pandas**: Usada para ler, manipular e salvar os dados das anotações em formato CSV.
* **matplotlib & seaborn**: Utilizadas para gerar os gráficos de barras que mostram a distribuição das classes.
* **os & shutil**: Usadas para interagir com o sistema de arquivos, como criar pastas e copiar imagens.
* **Pillow (PIL)**: Essencial para abrir e manipular os arquivos de imagem.
* **torchvision.transforms**: O núcleo da funcionalidade de aumento de dados, fornecendo todas as transformações de imagem.
* **tqdm**: Cria as barras de progresso para acompanhar tarefas demoradas.
* **concurrent.futures**: Habilita o processamento paralelo para acelerar a geração das imagens.
* **collections**: Usada para nomear os arquivos de imagem aumentados de forma única.

---

## Variáveis de Configuração (`Config`)

A seguir, uma breve descrição de cada variável na classe `Config` e seus possíveis valores.

* `BASE_PATH`: Caminho para a pasta raiz onde o dataset original está localizado.
    * **Valor**: String com o caminho para o diretório.

* `TRAIN_CSV_PATH`: Caminho completo para o arquivo CSV de anotações do conjunto de treino.
    * **Valor**: String com o caminho para o arquivo `.csv`.

* `SAVE_DIR`: Diretório principal onde todas as pastas e arquivos gerados serão salvos.
    * **Valor**: String com o nome do diretório de saída.

* `AUGMENTED_ONLY_DIR_NAME`: Nome da subpasta que armazenará **apenas** as imagens geradas.
    * **Valor**: String com o nome da pasta.

* `ALL_IMAGES_DIR_NAME`: Nome da subpasta que armazenará as imagens **originais e as geradas**.
    * **Valor**: String com o nome da pasta.

* `FINAL_CSV_NAME`: Nome do arquivo CSV final que conterá as anotações do dataset completo (original + aumentado).
    * **Valor**: String com o nome do arquivo `.csv`.

* `PLOT_BEFORE_NAME` / `PLOT_AFTER_NAME`: Nomes dos arquivos de imagem para os gráficos de distribuição de classes antes e depois do aumento.
    * **Valor**: String com o nome do arquivo `.png`.

* `TARGET_IMAGES_PER_CLASS`: O número alvo de imagens por classe no modo `BALANCE`.
    * **Valor**: Número inteiro (ex: `55000`).

* `AUGMENTATION_MODE`: Define a estratégia de aumento de dados.
    * **Valores possíveis**: `'BALANCE'`, `'AUGMENT_ALL'`.

* `TRANSFORMATION_TYPE`: Define qual conjunto de transformações de imagem será aplicado.
    * **Valores possíveis**: `'SIMPLE'`, `'COMPLETE'`.
