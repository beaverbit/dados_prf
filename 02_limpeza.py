# 02_limpeza.py
# Objetivo: Limpar e padronizar os dados consolidados brutos da PRF.

import pandas as pd
import numpy as np
import os

# Caminhos
ARQ_ENTRADA = os.path.join("dados_tratados", "dados_consolidados_brutos.csv")
ARQ_SAIDA = os.path.join("dados_tratados", "acidentes_limpos.csv")

# Colunas que serão mantidas (o restante é descartado)
COLUNAS_UTEIS = [
    # Tempo
    "data_inversa", "horario", "dia_semana",
    # Local
    "uf", "br", "km", "municipio", "latitude", "longitude", "regional",
    # Causa / tipo
    "causa_principal", "causa_acidente", "tipo_acidente", "classificacao_acidente",
    # Condições
    "fase_dia", "condicao_metereologica", "tipo_pista", "tracado_via", "uso_solo",
    # Vítimas
    "mortos", "feridos_graves", "feridos_leves", "ilesos",
    # Veículo / pessoa
    "tipo_veiculo", "idade", "sexo", "estado_fisico", "tipo_envolvido",
]


def limpar_dados():
    print("Carregando dados brutos consolidados...")

    # Leitura em chunks para não estourar a RAM (dataset ~1,2 GB)
    chunks = pd.read_csv(
        ARQ_ENTRADA,
        sep=";",
        encoding="utf-8",
        low_memory=False,
        chunksize=200_000,
    )

    lista_limpos = []
    total_linhas = 0

    for i, chunk in enumerate(chunks):
        total_linhas += len(chunk)
        print(f"  Processando chunk {i + 1} ({len(chunk):,} linhas)...")

        # 1. Mantém apenas as colunas úteis
        colunas_existentes = [c for c in COLUNAS_UTEIS if c in chunk.columns]
        df = chunk[colunas_existentes].copy()

        # 2. Remove linhas duplicadas dentro do chunk
        df = df.drop_duplicates()

        lista_limpos.append(df)

    print(f"\nTotal de linhas lidas: {total_linhas:,}")

    # Junta os chunks novamente
    df = pd.concat(lista_limpos, ignore_index=True)
    del lista_limpos  # libera memória

    print(f"Linhas após remoção de duplicadas: {len(df):,}")

    # 3. Conversão de tipos
    print("\nConvertendo tipos...")
    df["data_inversa"] = pd.to_datetime(df["data_inversa"], errors="coerce")
    df["horario"] = pd.to_datetime(
        df["horario"], format="%H:%M:%S", errors="coerce"
    ).dt.time

    # 4. Colunas derivadas
    print("Criando colunas derivadas...")
    df["ano"] = df["data_inversa"].dt.year
    df["mes"] = df["data_inversa"].dt.month
    df["hora"] = pd.to_datetime(
        df["horario"].astype(str), format="%H:%M:%S", errors="coerce"
    ).dt.hour

    # 5. Tratamento de valores nulos nas colunas de vítimas
    for col in ["mortos", "feridos_graves", "feridos_leves", "ilesos"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # 6. Coluna de gravidade (para classificação)
    df["gravidade"] = np.where(
        df["mortos"] > 0, "Fatal",
        np.where(
            df["feridos_graves"] > 0, "Grave",
            np.where(df["feridos_leves"] > 0, "Leve", "Sem Vítimas")
        )
    )

    # 7. Remove linhas sem data válida
    antes = len(df)
    df = df.dropna(subset=["data_inversa"])
    print(f"Removidas {antes - len(df):,} linhas sem data válida.")

    # 8. Filtro geográfico (opcional)
    # df = df[df["uf"] == "PR"].copy()
    # print(f"Filtrado para UF=PR: {len(df):,} linhas.")

    # 9. Reset do índice
    df = df.reset_index(drop=True)

    # 10. Salva o resultado
    os.makedirs("dados_tratados", exist_ok=True)
    print(f"\nSalvando dados limpos em: {ARQ_SAIDA}")
    df.to_csv(ARQ_SAIDA, index=False, sep=";", encoding="utf-8")

    # Resumo final
    print("\n--- Resumo final ---")
    print(f"Total de linhas:  {len(df):,}")
    print(f"Total de colunas: {df.shape[1]}")
    print(f"Anos cobertos:    {sorted(df['ano'].dropna().unique().tolist())}")
    print("\nDistribuição por gravidade:")
    print(df["gravidade"].value_counts())
    print("\nTop 5 UFs com mais acidentes:")
    print(df["uf"].value_counts().head())

    return df


if __name__ == "__main__":
    df = limpar_dados()
    print("\nLimpeza concluída com sucesso!")
