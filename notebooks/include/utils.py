import pandas as pd
import os
import zipfile
import gzip
import json
import gdown
from io import BytesIO
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq


def baixar_arquivo_zip(url):
    """
    Faz o download de um arquivo ZIP do Google Drive para a memória.
    
    Args:
        url (str): A URL de download direta do Google Drive.
    
    Returns:
        BytesIO: Um objeto BytesIO contendo o conteúdo do arquivo ZIP.
    """
    print("Baixando o arquivo ZIP do Google Drive para a memória...")
    zip_data = BytesIO()
    gdown.download(url, zip_data, quiet=False)
    zip_data.seek(0)  # Move o cursor do buffer para o início
    return zip_data


def extrair_arquivo_gz_do_zip(zip_data, gz_file_to_extract, dest_dir, json_file_to_extract):
    """
    Extrai um arquivo GZ específico de um ZIP armazenado em memória e salva o JSON descompactado no diretório de destino.
    
    Args:
        zip_data (BytesIO): O conteúdo do arquivo ZIP em memória.
        gz_file_to_extract (str): O caminho do arquivo GZ dentro do ZIP.
        dest_dir (str): O diretório de destino para salvar o arquivo JSON extraído.
        json_file_to_extract (str): O nome do arquivo JSON a ser salvo.
    
    Returns:
        str: O caminho completo do arquivo JSON extraído.
    """
    extracted_json_file_path = os.path.join(dest_dir, json_file_to_extract)
    os.makedirs(dest_dir, exist_ok=True)  # Cria o diretório de destino se não existir

    # Abrindo o arquivo ZIP da memória
    with zipfile.ZipFile(zip_data, 'r') as zip_ref:
        # Verifica se o arquivo desejado está no ZIP
        if gz_file_to_extract in zip_ref.namelist():
            print(f"Extraindo {gz_file_to_extract} do ZIP para a memória...")
            # Extraindo o arquivo GZ do ZIP em memória
            with zip_ref.open(gz_file_to_extract) as gz_file:
                # Usando gzip para descompactar o conteúdo e tqdm para barra de progresso
                with gzip.open(gz_file, 'rb') as json_file:
                    with open(extracted_json_file_path, 'wb') as output_file:
                        # Definindo o tamanho total para tqdm
                        file_size = zip_ref.getinfo(gz_file_to_extract).file_size
                        with tqdm(total=file_size, desc="Descompactando JSON", unit="B", unit_scale=True) as pbar:
                            while True:
                                chunk = json_file.read(1024)  # Lê em pedaços de 1024 bytes
                                if not chunk:
                                    break
                                output_file.write(chunk)
                                pbar.update(len(chunk))  # Atualiza a barra de progresso
            print(f"{json_file_to_extract} salvo em {extracted_json_file_path}")
        else:
            print(f"Arquivo {gz_file_to_extract} não encontrado no ZIP!")
            raise FileNotFoundError(f"Arquivo {gz_file_to_extract} não encontrado no ZIP!")

    return extracted_json_file_path


# Função para ler JSON em pedaços e criar um DataFrame
def read_large_json_with_pandas(file_path, num_lines=None, chunksize=10000):
    """
    Lê um arquivo JSON grande em pedaços usando Pandas, mostrando uma barra de progresso.
    Processa o arquivo em pedaços para evitar problemas de memória.
    
    Args:
        file_path (str): O caminho para o arquivo JSON.
        num_lines (int, opcional): O número de linhas a serem lidas. Se None, lê o arquivo todo.
        chunksize (int): O número de linhas a serem lidas por vez.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo {file_path} não encontrado!")
    
    # Inicializando a barra de progresso
    file_size = os.path.getsize(file_path)
    chunk_list = []
    
    with tqdm(total=file_size, desc="Lendo JSON", unit="B", unit_scale=True) as pbar:
        for chunk in pd.read_json(file_path, lines=True, chunksize=chunksize):
            if num_lines is not None and len(chunk_list) >= num_lines:
                break
            chunk_list.append(chunk)
            pbar.update(chunk.memory_usage(deep=True).sum())
        
    # Concatenar todos os pedaços em um único DataFrame
    df = pd.concat(chunk_list, ignore_index=True)
    return df


def contar_linhas_json(file_path):
    """
    Conta o número total de linhas em um arquivo JSON onde cada linha é um objeto JSON.
    
    Args:
        file_path (str): O caminho para o arquivo JSON.
        
    Returns:
        int: O número total de linhas no arquivo JSON.
    """
    with open(file_path, 'r') as file:
        num_linhas = sum(1 for _ in file)
    return num_linhas


def save_dataframe(df, file_path):
    """
    Salva um DataFrame em um arquivo Parquet com barra de progresso e mensagem de conclusão.
    
    Args:
        df (pd.DataFrame): O DataFrame a ser salvo.
        file_path (str): O caminho onde o arquivo Parquet será salvo.
    """
    # Calcula o número total de linhas para a barra de progresso
    total_rows = len(df)
    
    # Função auxiliar para salvar o DataFrame em partes
    def save_in_chunks(df, file_path, chunksize=10000):
        # Cria uma barra de progresso
        with tqdm(total=total_rows, desc="Salvando DataFrame", unit="rows") as pbar:
            # Salva cada pedaço em um arquivo Parquet temporário
            temp_files = []
            for start in range(0, total_rows, chunksize):
                end = min(start + chunksize, total_rows)
                chunk_df = df.iloc[start:end]
                temp_file = f"{file_path}_{start}_{end}.parquet"
                chunk_df.to_parquet(temp_file, index=False)
                temp_files.append(temp_file)
                pbar.update(end - start)
            
            # Combina todos os arquivos temporários em um único arquivo Parquet
            if temp_files:
                parquet_files = [pa.parquet.read_table(file) for file in temp_files]
                combined_table = pa.concat_tables(parquet_files)
                pq.write_table(combined_table, file_path)
                
                # Remove os arquivos temporários
                for file in temp_files:
                    os.remove(file)

    # Salva o DataFrame em partes para permitir a barra de progresso
    save_in_chunks(df, file_path)
    
    # Mensagem de conclusão
    print(f"{file_path} salvo com sucesso!")


# Carregar o DataFrame do arquivo Parquet
def load_dataframe(file_path):
    return pd.read_parquet(file_path)