# 01_extracao.py
# Objetivo: Ler todos os arquivos CSV brutos da PRF e consolidar em um único DataFrame.

import pandas as pd
import os
import glob

# Caminho da pasta com os arquivos brutos
PASTA_BRUTOS = "dados_brutos"
PASTA_TRATADOS = "dados_tratados"

def carregar_dados_brutos():
    """
    Lê todos os arquivos CSV dentro da pasta dados_brutos e concatena em um único DataFrame.
    """
    # Procura todos os arquivos .csv na pasta
    arquivos = glob.glob(os.path.join(PASTA_BRUTOS, "*.csv"))

    if not arquivos:
        print("Nenhum arquivo CSV encontrado na pasta 'dados_brutos'.")
        print("Baixe os arquivos em: https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf")
        return None

    print(f"Encontrados {len(arquivos)} arquivo(s) CSV.")

    lista_dataframes = []

    for arquivo in arquivos:
        print(f"Lendo: {arquivo}")
        try:
            # A PRF geralmente usa ';' como separador e encoding 'latin-1' ou 'utf-8'
            # Vamos tentar detectar automaticamente
            df = pd.read_csv(arquivo, sep=';', encoding='latin-1', low_memory=False)
            lista_dataframes.append(df)
            print(f"  -> {df.shape[0]} linhas, {df.shape[1]} colunas")
        except Exception as e:
            print(f"  -> Erro ao ler {arquivo}: {e}")

    if not lista_dataframes:
        return None

    # Junta todos os DataFrames em um só
    df_consolidado = pd.concat(lista_dataframes, ignore_index=True)

    print("\n--- Resumo dos dados brutos ---")
    print(f"Total de linhas: {df_consolidado.shape[0]}")
    print(f"Total de colunas: {df_consolidado.shape[1]}")
    print("\nColunas disponíveis:")
    print(list(df_consolidado.columns))

    return df_consolidado

if __name__ == "__main__":
    df = carregar_dados_brutos()

    if df is not None:
        # Salva o consolidado bruto (ainda sujo) para a próxima etapa
        os.makedirs(PASTA_TRATADOS, exist_ok=True)
        caminho_saida = os.path.join(PASTA_TRATADOS, "dados_consolidados_brutos.csv")
        df.to_csv(caminho_saida, index=False, sep=';', encoding='utf-8')
        print(f"\nDados brutos consolidados salvos em: {caminho_saida}")