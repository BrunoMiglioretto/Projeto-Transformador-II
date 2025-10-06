# Documentação: Script de Balanceamento de Atributos via Data Augmentation

Este documento detalha o funcionamento do script em Python `data_augmentation.py`, projetado para mitigar o desbalanceamento de classes em datasets de imagens com múltiplos atributos (multi-label).

## 1. Objetivo Principal

O objetivo central do script é aumentar a representatividade de classes minoritárias para cada atributo de um dataset. A meta é alcançar um balanço próximo de **50/50** entre a presença (Classe 1) e a ausência (Classe 0) de cada característica, gerando novas imagens sintéticas através de técnicas de *data augmentation*.

---

## 2. Fluxo de Execução do Script

O script opera em quatro etapas principais:

1.  **Configuração e Amostragem**: Carrega as configurações, lê o dataset completo e, opcionalmente, extrai uma amostra aleatória menor para processamento, tornando o processo mais rápido para testes e validação.
2.  **Análise Inicial**: Gera um relatório visual (gráfico) e textual (tabela) mostrando o estado de desbalanceamento do dataset (ou da amostra) antes de qualquer modificação.
3.  **Processo de Aumento (Augmentation)**: Itera sobre cada atributo (feature), identifica a classe minoritária e gera novas imagens sintéticas a partir dela até que sua contagem se iguale à da classe majoritária (com base na amostra original).
4.  **Análise Final**: Salva o novo dataset (CSV com caminhos para as imagens originais e aumentadas) e gera dois gráficos finais e uma tabela para analisar o resultado do balanceamento.

---

## 3. Lógica de Balanceamento: O Efeito de "Contaminação Cruzada"

A parte mais importante para entender os resultados é a lógica de balanceamento e seus efeitos colaterais.

### A Lógica Implementada

O script adota uma abordagem de **balanceamento independente por atributo**:
- Para cada atributo (ex: `Age-Young`), ele olha **exclusivamente para a amostra original** (`df_original`).
- Ele calcula a diferença entre a classe majoritária e a minoritária (ex: 4000 imagens `Age-Young=0` vs. 200 `Age-Young=1`).
- Ele gera a quantidade necessária de imagens da classe minoritária para igualar a contagem (neste caso, 3800 novas imagens de `Age-Young=1`).
- Este processo é repetido para **todos os outros atributos**, sempre usando o mesmo dataset original como referência.

### A Consequência: Por que o Balanço Final não é Perfeito?

Embora a intenção seja um balanço de 50/50, os gráficos finais mostram que isso não acontece perfeitamente para todos os atributos. O motivo é um fenômeno que podemos chamar de **contaminação cruzada**.

**Uma imagem possui múltiplos atributos simultaneamente.**

Vamos a um exemplo prático:
1.  **Balanceando `Accessory-Hat` (Usa Chapéu)**: Este é um atributo raro. Para balanceá-lo, o script pode precisar criar milhares de novas imagens de pessoas usando chapéu.
2.  **O Efeito Colateral**: As imagens originais usadas para gerar essas novas amostras "com chapéu" também possuem outros atributos. Por exemplo, imagine que a maioria das pessoas com chapéu no dataset original também eram `Gender-Female=0` (homens).
3.  **O Resultado**: Ao gerar milhares de imagens "com chapéu", o script também está, indiretamente, injetando milhares de imagens de `Gender-Female=0` no dataset final.

**Conclusão da Lógica**: Ao consertar o desequilíbrio de um atributo, o script inevitavelmente "perturba" o equilíbrio de todos os outros. O resultado final é a soma de todas essas perturbações. O dataset fica **muito mais balanceado** do que no início, mas não perfeitamente 50/50 em todas as frentes.

---

## 4. Interpretação dos Gráficos de Saída

O script gera três gráficos principais para diagnóstico:

### Gráfico 1: `class_distribution_before_sample.png`

- **O que mostra**: A distribuição de classes (0 e 1) para cada atributo **antes** do processo de aumento.
- **Como interpretar**: Geralmente mostra um grande desequilíbrio, com barras azuis (Classe 0) muito maiores que as laranjas (Classe 1) para a maioria dos atributos. Este é o nosso ponto de partida.

### Gráfico 2: `class_distribution_after_shared_scale.png`

- **O que mostra**: A distribuição final de todas as classes, com todos os atributos em um único gráfico e uma **escala de contagem (eixo Y) compartilhada**.
- **Como interpretar**: Este gráfico dá uma visão geral da composição final do dataset. Ele mostra a magnitude real das contagens e evidencia o efeito da contaminação cruzada. Por exemplo, você pode notar que a contagem total para `Age-Adult` aumentou drasticamente, pois muitas imagens geradas para outros atributos também pertenciam a essa categoria.

### Gráfico 3: `class_distribution_after_independent_scales.png`

- **O que mostra**: A distribuição final, mas com um subplot para cada atributo, onde cada um tem sua **própria escala de contagem (eixo Y)**.
- **Como interpretar**: O objetivo deste gráfico é verificar se as barras azul e laranja estão próximas em altura *dentro de seu próprio contexto*. Embora o balanço não seja perfeito (pelo motivo da contaminação cruzada), este gráfico torna mais fácil ver se a disparidade foi drasticamente reduzida para cada atributo individualmente, que era o objetivo principal.

---

## 5. Configurações Atuais (`Config` class)

A configuração atual do script define os seguintes parâmetros chave:

-   `BASE_PATH` e `TRAIN_CSV_PATH`: Apontam para os dados de entrada do dataset Market1501.
-   `SAVE_DIR`: `'data-balanced-50-50-sample'`. Todas as saídas (imagens, CSV, gráficos) serão salvas neste diretório.
-   `SAMPLE_PERCENTAGE`: **`0.6`** (ou 60%). O script não processará o dataset inteiro, mas uma amostra aleatória contendo 60% dos dados, para acelerar a execução.
-   `RANDOM_SEED`: **`42`**. Garante que a amostra aleatória de 60% seja sempre a mesma a cada execução, tornando os resultados reproduzíveis.
-   `TRANSFORMATION_TYPE`: **`'SIMPLE'`**. Utiliza o conjunto mais simples e rápido de transformações de imagem (`simple_transform`) para gerar as novas amostras. Isso inclui recortes, inversão horizontal, variações de cor e conversão para escala de cinza.