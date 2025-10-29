import argparse
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

import gdown
import numpy as np
from tqdm import tqdm


def download_url(url, dst):
    """
    Baixa um arquivo de uma URL para um destino.
    (Mantida caso seja necessária, embora o PAR2025 use gdown)
    """
    print(f'* Baixando de: "{url}"')
    print(f'* Para: "{dst}"')

    def _reporthook(count, block_size, total_size):
        global start_time
        if count == 0:
            start_time = time.time()
            return
        duration = time.time() - start_time + 1e-6
        progress_size = int(count * block_size)
        speed = int(progress_size / (1024 * duration))
        percent = int(count * block_size * 100 / total_size + 1e-6)
        sys.stdout.write(
            "\r...%d%%, %d MB, %d KB/s, %d segundos" % (percent, progress_size / (1024 * 1024), speed, duration)
        )
        sys.stdout.flush()

    if dst.exists():
        print("Arquivo já existe. Pulando download.")
        return
    else:
        try:
            urllib.request.urlretrieve(url, dst, _reporthook)
            sys.stdout.write("\n")
        except Exception as e:
            print(f"\nErro ao baixar {url}: {e}")
            if dst.exists():
                dst.unlink()  # Remove arquivo incompleto


def extract_zip(src, dst):
    """Extrai um arquivo zip para um diretório de destino com barra de progresso."""
    print(f"Extraindo '{src.name}' para '{dst}'...")
    try:
        with zipfile.ZipFile(src, "r") as zf:
            for member in tqdm(zf.infolist(), desc=f"Extraindo {src.name}"):
                try:
                    zf.extract(member, dst)
                except zipfile.error as err:
                    print(f"Erro ao extrair {member.filename}: {err}")
    except zipfile.BadZipFile:
        print(f"ERRO: O arquivo '{src.name}' não é um zip válido ou está corrompido.")
    except FileNotFoundError:
        print(f"ERRO: Arquivo zip não encontrado em '{src}'")


def prepare_par2025(base_data_path):
    """
    Baixa e prepara o dataset PAR2025 em 'base_data_path/PAR2025'.
    """
    print("\n--- 1. Preparando Dataset PAR2025 ---")
    dataset_path = Path(base_data_path)
    
    url = "https://drive.google.com/file/d/1ZPlmw2PxYctWjFBFKRw-CVM6G4Agd_ar/view?usp=sharing"
    main_zipfile = dataset_path / "PAR2025_main.zip"

    print(f"Baixando dataset principal (PAR2025) de {url}...")
    gdown.download(url, output=str(main_zipfile), quiet=False, use_cookies=False, fuzzy=True)

    if main_zipfile.exists():
        extract_zip(main_zipfile, dataset_path)
    else:
        print(f"ERRO: Download do {main_zipfile.name} falhou.")
        return

    par_dir = dataset_path / "PAR2025"
    if not par_dir.exists() or not par_dir.is_dir():
        print(f"ERRO: A pasta {par_dir} não foi criada corretamente.")
        print("Por favor, verifique se o .zip baixado contém a pasta 'PAR2025'.")
        return

    print(f"Pasta do dataset PAR2025: {par_dir.resolve()}")

    train_zip = par_dir / "training_set.zip"
    val_zip = par_dir / "validation_set.zip"

    if train_zip.exists():
        extract_zip(train_zip, par_dir)
        train_zip.unlink()  # Remove o .zip após a extração
        print("Extração de 'training_set' (PAR2025) concluída.")
    else:
        print(f"AVISO: {train_zip.name} não encontrado em {par_dir}")

    if val_zip.exists():
        extract_zip(val_zip, par_dir)
        val_zip.unlink()  # Remove o .zip após a extração
        print("Extração de 'validation_set' (PAR2025) concluída.")
    else:
        print(f"AVISO: {val_zip.name} não encontrado em {par_dir}")

    main_zipfile.unlink()
    print("Preparação do PAR2025 concluída.")


def prepare_all_datasets(base_data_dir):
    """
    Função principal que orquestra o download do dataset PAR2025.
    - PAR2025 vai para 'base_data_dir/PAR2025'
    """
    dataset_path = Path(base_data_dir)
    dataset_path.mkdir(parents=True, exist_ok=True)
    print(f"Diretório base de dados: {dataset_path.resolve()}")

    prepare_par2025(dataset_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script de Download e Preparação do Dataset PAR2025",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Diretório base do dataset. (ex: ./data)",
    )
    args = parser.parse_args()

    prepare_all_datasets(args.data_dir)

    print("\n--- Download e preparação do PAR2025 concluídos! ---")
    print(f"Estrutura final esperada:")
    print(f"  {args.data_dir}/")
    print(f"  └── PAR2025/")
    print(f"      ├── training_set/")
    print(f"      └── validation_set/")