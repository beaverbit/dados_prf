# 01_extracao.py
# Objetivo: Ler todos os arquivos CSV brutos da PRF em dados_brutos/ e consolidar em um único DataFrame.

import pandas as pd
import os
import glob

# Caminho das pastas
PASTA_BRUTOS = "dados_brutos"
PASTA_TRATADOS = "dados_tratados"
ARQ_SAIDA = os.path.join(PASTA_TRATADOS, "dados_consolidados_brutos.csv")


def carregar_dados_brutos():
    """
    Lê todos os arquivos CSV dentro de dados_brutos/ e concatena em um único DataFrame.

    Tenta automaticamente encoding UTF-8 e, em caso de falha, recorre a Latin-1.
    Separador detectado automaticamente pelo engine do pandas.
    """
    arquivos = sorted(glob.glob(os.path.join(PASTA_BRUTOS, "*.csv")))

    if not arquivos:
        print(f"Nenhum arquivo CSV encontrado na pasta '{PASTA_BRUTOS}/'.")
        print("Baixe os arquivos em: https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf")
        return None

    print(f"Encontrados {len(arquivos)} arquivo(s) CSV.\n")

    lista_dataframes = []

    for arquivo in arquivos:
        print(f"Lendo: {arquivo}")
        df = None

        # Tentativa 1: UTF-8 (padrão atual da PRF)
        try:
            df = pd.read_csv(
                arquivo,
                sep=";",
                encoding="utf-8",
                low_memory=False,
            )
        except UnicodeDecodeError:
            # Tentativa 2: Latin-1 (compatibilidade com arquivos antigos)
            try:
                df = pd.read_csv(
                    arquivo,
                    sep=";",
                    encoding="latin-1",
                    low_memory=False,
                )
                print("  -> Encoding alternativo (latin-1) aplicado.")
            except Exception as e:
                print(f"  -> Erro ao ler {arquivo}: {e}")
                continue
        except Exception as e:
            print(f"  -> Erro ao ler {arquivo}: {e}")
            continue

        if df is not None:
            lista_dataframes.append(df)
            print(f"  -> {df.shape[0]:,} linhas, {df.shape[1]} colunas")

    if not lista_dataframes:
        print("\nNenhum arquivo pôde ser lido com sucesso.")
        return None

    # Concatena todos os DataFrames
    df_consolidado = pd.concat(lista_dataframes, ignore_index=True)

    print("\n--- Resumo dos dados brutos ---")
    print(f"Total de linhas:  {df_consolidado.shape[0]:,}")
    print(f"Total de colunas: {df_consolidado.shape[1]}")
    print("\nColunas disponíveis:")
    print(list(df_consolidado.columns))

    return df_consolidado


if __name__ == "__main__":
    df = carregar_dados_brutos()

    if df is not None:
        os.makedirs(PASTA_TRATADOS, exist_ok=True)
        df.to_csv(ARQ_SAIDA, index=False, sep=";", encoding="utf-8")
        print(f"\nDados brutos consolidados salvos em: {ARQ_SAIDA}")
