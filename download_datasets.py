import argparse
import shutil
import sys
import time
import urllib
import zipfile
from pathlib import Path, PurePosixPath

import gdown
import numpy as np
from tqdm import tqdm


#
# FUNÇÃO AUXILIAR MANTIDA DO SCRIPT ORIGINAL
#
def extract_zip(src, dst):
    """Extrai um arquivo zip para um diretório de destino."""
    print(f"Extraindo '{src.name}' para '{dst}'...")
    with zipfile.ZipFile(src, "r") as zf:
        for member in tqdm(zf.infolist(), desc="Progresso da extração"):
            try:
                zf.extract(member, dst)
            except zipfile.error as err:
                print(f"Erro ao extrair {member.filename}: {err}")


#
# FUNÇÃO PRINCIPAL AJUSTADA PARA SEU DATASET
#
def prepare_datasets(path):
    """
    Baixa e prepara o dataset PAR2025 conforme as imagens fornecidas.
    """
    # 1. Define e cria o diretório de dados principal (ex: ./data)
    dataset_path = Path(path)
    dataset_path.mkdir(parents=True, exist_ok=True)
    print(f"Diretório de dados assegurado: {dataset_path.resolve()}")

    # 2. Define a URL e o local de download do .zip principal
    url = "https://drive.google.com/file/d/1ZPlmw2PxYctWjFBFKRw-CVM6G4Agd_ar/view?usp=sharing"
    main_zipfile = dataset_path / "PAR2025_main.zip"

    # 3. Baixa o arquivo .zip principal
    print(f"Baixando dataset principal de {url}...")
    gdown.download(url, output=str(main_zipfile), quiet=False, use_cookies=False, fuzzy=True)

    # 4. Extrai o .zip principal para o diretório 'data'
    #    Isso deve criar a pasta 'data/PAR2025'
    extract_zip(main_zipfile, dataset_path)

    # 5. Define o caminho para a pasta PAR2025 (criada no passo anterior)
    par_dir = dataset_path / "PAR2025"
    if not par_dir.exists() or not par_dir.is_dir():
        print(f"ERRO: A pasta {par_dir} não foi criada corretamente.")
        print("Por favor, verifique se o .zip baixado contém a pasta 'PAR2025'.")
        return

    print(f"Pasta do dataset encontrada: {par_dir.resolve()}")

    # 6. Define os caminhos para os .zip internos (training_set.zip, etc.)
    train_zip = par_dir / "training_set.zip"
    val_zip = par_dir / "validation_set.zip"

    # 7. Extrai o 'training_set.zip' dentro de 'data/PAR2025/'
    #    Isso criará a pasta 'data/PAR2025/training_set'
    if train_zip.exists():
        extract_zip(train_zip, par_dir)
        train_zip.unlink()  # Remove o .zip após a extração
        print("Extração de 'training_set' concluída.")
    else:
        print(f"AVISO: {train_zip.name} não encontrado em {par_dir}")

    # 8. Extrai o 'validation_set.zip' dentro de 'data/PAR2025/'
    #    Isso criará a pasta 'data/PAR2025/validation_set'
    if val_zip.exists():
        extract_zip(val_zip, par_dir)
        val_zip.unlink()  # Remove o .zip após a extração
        print("Extração de 'validation_set' concluída.")
    else:
        print(f"AVISO: {val_zip.name} não encontrado em {par_dir}")

    # 9. Remove o .zip principal que foi baixado
    main_zipfile.unlink()
    print(f"Download e preparação concluídos. Arquivos temporários removidos.")
    print(f"Estrutura final em: {par_dir}")


#
# BLOCO DE EXECUÇÃO MANTIDO DO SCRIPT ORIGINAL
#
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script de Download e Preparação do Dataset PAR2025",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",  # Cria a pasta 'data' no mesmo diretório do script
        help="Diretório do dataset. O dataset baixado será armazenado aqui.",
    )
    args = parser.parse_args()

    prepare_datasets(args.data_dir)