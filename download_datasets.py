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

#
# FUNÇÕES AUXILIARES (COMBINADAS DE AMBOS OS SCRIPTS)
#

def download_url(url, dst):
    """
    Baixa um arquivo de uma URL para um destino.
    Usado especificamente para o dataset PETA (Dropbox).
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


#
# FUNÇÕES DE PREPARAÇÃO DE DATASET (SEPARADAS)
#

def prepare_par2025(base_data_path):
    """
    Baixa e prepara o dataset PAR2025 em 'base_data_path/PAR2025'.
    (Lógica do Script 1)
    """
    print("\n--- 1. Preparando Dataset PAR2025 ---")
    
    # O dataset_path aqui é o 'data', ex: ./data
    dataset_path = Path(base_data_path)
    
    # 1. Define a URL e o local de download do .zip principal
    url = "https://drive.google.com/file/d/1ZPlmw2PxYctWjFBFKRw-CVM6G4Agd_ar/view?usp=sharing"
    main_zipfile = dataset_path / "PAR2025_main.zip"

    # 2. Baixa o arquivo .zip principal
    print(f"Baixando dataset principal (PAR2025) de {url}...")
    gdown.download(url, output=str(main_zipfile), quiet=False, use_cookies=False, fuzzy=True)

    # 3. Extrai o .zip principal para o diretório 'data'
    #    Isso deve criar a pasta 'data/PAR2025'
    if main_zipfile.exists():
        extract_zip(main_zipfile, dataset_path)
    else:
        print(f"ERRO: Download do {main_zipfile.name} falhou.")
        return

    # 4. Define o caminho para a pasta PAR2025 (criada no passo anterior)
    par_dir = dataset_path / "PAR2025"
    if not par_dir.exists() or not par_dir.is_dir():
        print(f"ERRO: A pasta {par_dir} não foi criada corretamente.")
        print("Por favor, verifique se o .zip baixado contém a pasta 'PAR2025'.")
        return

    print(f"Pasta do dataset PAR2025: {par_dir.resolve()}")

    # 5. Define os caminhos para os .zip internos (training_set.zip, etc.)
    train_zip = par_dir / "training_set.zip"
    val_zip = par_dir / "validation_set.zip"

    # 6. Extrai o 'training_set.zip' dentro de 'data/PAR2025/'
    if train_zip.exists():
        extract_zip(train_zip, par_dir)
        train_zip.unlink()  # Remove o .zip após a extração
        print("Extração de 'training_set' (PAR2025) concluída.")
    else:
        print(f"AVISO: {train_zip.name} não encontrado em {par_dir}")

    # 7. Extrai o 'validation_set.zip' dentro de 'data/PAR2025/'
    if val_zip.exists():
        extract_zip(val_zip, par_dir)
        val_zip.unlink()  # Remove o .zip após a extração
        print("Extração de 'validation_set' (PAR2025) concluída.")
    else:
        print(f"AVISO: {val_zip.name} não encontrado em {par_dir}")

    # 8. Remove o .zip principal que foi baixado
    main_zipfile.unlink()
    print("Preparação do PAR2025 concluída.")


def prepare_market(upar_path):
    """Prepara o Market1501 em 'upar_path/Market1501'."""
    print("\n--- 2. Preparando Dataset Market1501 (UPAR) ---")
    market_1501_path = upar_path / "Market1501"
    market_1501_zipfile = upar_path / "market_1501.zip"
    url = "https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view?resourcekey=0-8nyl7K9_x37HlQm34MmrYQ"
    
    print("Baixando Market 1501 dataset...")
    gdown.download(url, output=str(market_1501_zipfile), quiet=False, use_cookies=False, fuzzy=True)
    
    if market_1501_zipfile.exists():
        print("Extraindo Market 1501 dataset...")
        extract_zip(market_1501_zipfile, upar_path)
        
        # Renomeia a pasta extraída
        src_rename_path = upar_path / "Market-1501-v15.09.15"
        if src_rename_path.exists():
            src_rename_path.rename(market_1501_path)
        else:
            print(f"AVISO: Pasta esperada '{src_rename_path}' não encontrada após extração.")
            
        market_1501_zipfile.unlink()  # Remove o zip
    else:
        print(f"ERRO: Download do {market_1501_zipfile.name} falhou.")
    print("Market1501 concluído.")


def prepare_pa100k(upar_path):
    """Prepara o PA100k em 'upar_path/PA100k'."""
    print("\n--- 3. Preparando Dataset PA100k (UPAR) ---")
    pa100k_path = upar_path / "PA100k"
    pa100k_path.mkdir(parents=True, exist_ok=True)
    
    print("Baixando PA100k dataset (pode levar um tempo)...")
    url = "https://drive.google.com/drive/folders/1d_D0Yh7C262gr0ef9EqkvG_M3fqgAWa2?usp=sharing"
    gdown.download_folder(url, output=str(pa100k_path), quiet=False, use_cookies=False)
    
    data_zip = pa100k_path / "data.zip"
    if data_zip.exists():
        print("Extraindo PA100k dataset...")
        extract_zip(data_zip, pa100k_path)
        data_zip.unlink()  # Remove o zip
    else:
        print(f"AVISO: {data_zip.name} não encontrado em {pa100k_path}. Download pode ter falhado.")
    print("PA100k concluído.")


def prepare_peta(upar_path):
    """Prepara o PETA em 'upar_path/PETA'."""
    print("\n--- 4. Preparando Dataset PETA (UPAR) ---")
    
    # PETA requer um arquivo de mapeamento local
    mapping_file = Path("peta_file_mapping.txt")
    if not mapping_file.exists():
        print(f"ERRO: '{mapping_file}' não encontrado no diretório atual ({Path.cwd()}).")
        print("Este arquivo é essencial para organizar o dataset PETA.")
        print("Pulando preparação do PETA.")
        return

    peta_path = upar_path / "PETA"
    peta_path.mkdir(parents=True, exist_ok=True)
    peta_zipfile = peta_path / "peta.zip"
    
    print("Baixando PETA dataset...")
    url = "https://www.dropbox.com/s/52ylx522hwbdxz6/PETA.zip?dl=1"
    download_url(url, peta_zipfile)
    
    if peta_zipfile.exists():
        print("Extraindo PETA dataset...")
        extract_zip(peta_zipfile, peta_path)
        
        peta_img_path = peta_path / "images"
        peta_img_path.mkdir(parents=True, exist_ok=True)
        
        print("Lendo mapeamento de arquivos PETA...")
        mapping = {row[0]: row[1] for row in np.genfromtxt(mapping_file, dtype=str, delimiter=",")}
        
        print("Organizando arquivos PETA...")
        for file in tqdm(peta_path.glob("*/*/*/*"), desc="Movendo arquivos PETA"):
            if file.suffix == ".txt":
                continue
            
            # Cria o caminho relativo ao 'upar_path'
            try:
                relative_file_path = str(PurePosixPath(file.relative_to(upar_path)))
            except ValueError:
                print(f"AVISO: Não foi possível obter caminho relativo para {file}")
                continue

            if relative_file_path in mapping:
                dest_path = upar_path / mapping[relative_file_path]
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(file), str(dest_path))
            else:
                # Apenas imprime o aviso se o arquivo não estiver já no destino
                if not str(file).startswith(str(peta_img_path)):
                    print(f"AVISO: {relative_file_path} não encontrado no mapping.")

        peta_zipfile.unlink()  # Remove o zip
        # Remove pastas vazias que sobraram da extração
        shutil.rmtree(peta_path / "PETA_release", ignore_errors=True)
    else:
        print(f"ERRO: Download do {peta_zipfile.name} falhou.")
    print("PETA concluído.")


def prepare_annotations(upar_path):
    """Prepara as anotações em 'upar_path/annotations'."""
    print("\n--- 5. Preparando Anotações (UPAR) ---")
    anno_path = upar_path / "annotations"
    anno_path.mkdir(parents=True, exist_ok=True)
    anno_zipfile = anno_path / "development.zip"
    
    print("Baixando anotações...")
    url = "https://drive.google.com/file/d/1FMX9nUrXArxW4wkORO6Z7zp7xy7JBjUM/view?usp=sharing"
    gdown.download(url, output=str(anno_zipfile), quiet=False, use_cookies=False, fuzzy=True)
    
    if anno_zipfile.exists():
        print("Extraindo anotações...")
        extract_zip(anno_zipfile, anno_path)
        anno_zipfile.unlink()  # Remove o zip
    else:
        print(f"ERRO: Download do {anno_zipfile.name} falhou.")
    print("Anotações UPAR concluídas.")


def prepare_templates(base_path):
    """Prepara os templates de submissão em 'base_path' (ex: ./)."""
    print("\n--- 6. Preparando Templates de Submissão ---")
    template_zipfile = base_path / "submission_templates.zip"
    
    print("Baixando templates...")
    url = "https://drive.google.com/file/d/11ZxT8kixkV-vAj8aixS8n2aGJ5Rw0OQy/view?usp=sharing"
    gdown.download(url, output=str(template_zipfile), quiet=False, use_cookies=False, fuzzy=True)
    
    if template_zipfile.exists():
        print("Extraindo templates...")
        extract_zip(template_zipfile, base_path)
        template_zipfile.unlink()  # Remove o zip
    else:
        print(f"ERRO: Download do {template_zipfile.name} falhou.")
    print("Templates de submissão concluídos.")


#
# FUNÇÃO PRINCIPAL DE ORQUESTRAÇÃO
#

def prepare_all_datasets(base_data_dir):
    """
    Função principal que orquestra o download de todos os datasets.
    - PAR2025 vai para 'base_data_dir/PAR2025'
    - UPAR datasets vão para 'base_data_dir/UPAR/'
    """
    # Pasta principal, ex: ./data
    dataset_path = Path(base_data_dir)
    dataset_path.mkdir(parents=True, exist_ok=True)
    print(f"Diretório base de dados: {dataset_path.resolve()}")

    # Sub-pasta para os datasets UPAR, ex: ./data/UPAR
    upar_path = dataset_path / "UPAR"
    upar_path.mkdir(parents=True, exist_ok=True)
    print(f"Diretório datasets UPAR: {upar_path.resolve()}")

    # --- 1. Preparar PAR2025 ---
    # (Logs internos na função)
    prepare_par2025(dataset_path)

    # --- 2. Preparar Datasets UPAR ---
    # (Logs internos nas funções)
    prepare_market(upar_path)
    prepare_pa100k(upar_path)
    prepare_peta(upar_path)

    # --- 3. Preparar Anotações UPAR ---
    # (Logs internos na função)
    prepare_annotations(upar_path)


#
# BLOCO DE EXECUÇÃO
#

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script de Download e Preparação de Datasets (PAR2025 & UPAR Challenge)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Diretório base dos datasets. (ex: ./data)",
    )
    args = parser.parse_args()

    # Prepara os datasets principais (PAR2025 e UPAR)
    prepare_all_datasets(args.data_dir)

    # Prepara os templates de submissão (vão para ./)
    prepare_templates(Path("./"))

    print("\n--- Todos os downloads e preparações foram concluídos! ---")
    print(f"Estrutura final esperada:")
    print(f"  {args.data_dir}/")
    print(f"  ├── PAR2025/")
    print(f"  │   ├── training_set/")
    print(f"  │   └── validation_set/")
    print(f"  └── UPAR/")
    print(f"      ├── Market1501/")
    print(f"      ├── PA100k/")
    print(f"      ├── PETA/")
    print(f"      └── annotations/")
    print(f"  ./submission_templates/")