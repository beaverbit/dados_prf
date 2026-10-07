# 03_analise.py
# Objetivo: Gerar as análises estatísticas descritivas dos acidentes limpos.

import pandas as pd
import numpy as np
import os

ARQ_ENTRADA = os.path.join("dados_tratados", "acidentes_limpos.csv")
ARQ_SAIDA = os.path.join("dados_tratados", "resultados_analise.csv")

# Recorte geográfico
# Defina como None para analisar o Brasil inteiro, ou "PR" para apenas o Paraná
UF_FOCO = "PR"


def carregar_dados():
    print("Carregando dados limpos...")
    df = pd.read_csv(
        ARQ_ENTRADA,
        sep=";",
        encoding="utf-8",
        low_memory=False,
        parse_dates=["data_inversa"],
    )
    print(f"Total de linhas: {len(df):,}")

    if UF_FOCO:
        df = df[df["uf"] == UF_FOCO].copy()
        print(f"Filtrado para UF={UF_FOCO}: {len(df):,} linhas")

    return df


def analise_por_ano(df):
    print("\n" + "=" * 60)
    print("ANÁLISE 1: Acidentes por ano")
    print("=" * 60)
    resumo = df.groupby("ano").agg(
        total_acidentes=("gravidade", "count"),
        mortos=("mortos", "sum"),
        feridos_graves=("feridos_graves", "sum"),
        feridos_leves=("feridos_leves", "sum"),
    ).reset_index()
    resumo["taxa_fatalidade_%"] = (
        resumo["mortos"] / resumo["total_acidentes"] * 100
    ).round(2)
    print(resumo.to_string(index=False))
    return resumo


def analise_por_mes(df):
    print("\n" + "=" * 60)
    print("ANÁLISE 2: Acidentes por mês")
    print("=" * 60)
    resumo = df.groupby("mes").agg(
        total=("gravidade", "count"),
        mortos=("mortos", "sum"),
    ).reset_index()
    resumo["taxa_fatalidade_%"] = (resumo["mortos"] / resumo["total"] * 100).round(2)
    print(resumo.to_string(index=False))
    return resumo


def analise_por_dia_semana(df):
    print("\n" + "=" * 60)
    print("ANÁLISE 3: Acidentes por dia da semana")
    print("=" * 60)
    ordem = ["segunda-feira", "terça-feira", "quarta-feira", "quinta-feira",
             "sexta-feira", "sábado", "domingo"]
    resumo = df.groupby("dia_semana").agg(
        total=("gravidade", "count"),
        mortos=("mortos", "sum"),
    ).reset_index()
    resumo["taxa_fatalidade_%"] = (resumo["mortos"] / resumo["total"] * 100).round(2)
    resumo["dia_semana"] = pd.Categorical(
        resumo["dia_semana"], categories=ordem, ordered=True
    )
    resumo = resumo.sort_values("dia_semana")
    print(resumo.to_string(index=False))
    return resumo


def analise_por_hora(df):
    print("\n" + "=" * 60)
    print("ANÁLISE 4: Acidentes por hora do dia")
    print("=" * 60)
    resumo = df.groupby("hora").agg(
        total=("gravidade", "count"),
        mortos=("mortos", "sum"),
    ).reset_index()
    resumo["taxa_fatalidade_%"] = (resumo["mortos"] / resumo["total"] * 100).round(2)
    print(resumo.to_string(index=False))
    return resumo


def analise_por_condicao_meteorologica(df):
    print("\n" + "=" * 60)
    print("ANÁLISE 5: Acidentes por condição meteorológica")
    print("=" * 60)
    resumo = df.groupby("condicao_metereologica").agg(
        total=("gravidade", "count"),
        mortos=("mortos", "sum"),
    ).reset_index()
    resumo["taxa_fatalidade_%"] = (resumo["mortos"] / resumo["total"] * 100).round(2)
    resumo = resumo.sort_values("total", ascending=False)
    print(resumo.to_string(index=False))
    return resumo


def analise_por_tipo_pista(df):
    print("\n" + "=" * 60)
    print("ANÁLISE 6: Acidentes por tipo de pista")
    print("=" * 60)
    resumo = df.groupby("tipo_pista").agg(
        total=("gravidade", "count"),
        mortos=("mortos", "sum"),
    ).reset_index()
    resumo["taxa_fatalidade_%"] = (resumo["mortos"] / resumo["total"] * 100).round(2)
    resumo = resumo.sort_values("total", ascending=False)
    print(resumo.to_string(index=False))
    return resumo


def analise_top_causas(df):
    print("\n" + "=" * 60)
    print("ANÁLISE 7: 10 principais causas de acidentes")
    print("=" * 60)
    resumo = df.groupby("causa_acidente").agg(
        total=("gravidade", "count"),
        mortos=("mortos", "sum"),
    ).reset_index()
    resumo["taxa_fatalidade_%"] = (resumo["mortos"] / resumo["total"] * 100).round(2)
    resumo = resumo.sort_values("total", ascending=False).head(10)
    print(resumo.to_string(index=False))
    return resumo


def analise_por_tipo_acidente(df):
    print("\n" + "=" * 60)
    print("ANÁLISE 8: 10 principais tipos de acidente")
    print("=" * 60)
    resumo = df.groupby("tipo_acidente").agg(
        total=("gravidade", "count"),
        mortos=("mortos", "sum"),
    ).reset_index()
    resumo["taxa_fatalidade_%"] = (resumo["mortos"] / resumo["total"] * 100).round(2)
    resumo = resumo.sort_values("total", ascending=False).head(10)
    print(resumo.to_string(index=False))
    return resumo


def analise_por_fase_dia(df):
    print("\n" + "=" * 60)
    print("ANÁLISE 9: Acidentes por fase do dia")
    print("=" * 60)
    resumo = df.groupby("fase_dia").agg(
        total=("gravidade", "count"),
        mortos=("mortos", "sum"),
    ).reset_index()
    resumo["taxa_fatalidade_%"] = (resumo["mortos"] / resumo["total"] * 100).round(2)
    resumo = resumo.sort_values("total", ascending=False)
    print(resumo.to_string(index=False))
    return resumo


def analise_por_gravidade(df):
    print("\n" + "=" * 60)
    print("ANÁLISE 10: Distribuição de gravidade")
    print("=" * 60)
    resumo = df["gravidade"].value_counts().reset_index()
    resumo.columns = ["gravidade", "total"]
    resumo["percentual_%"] = (resumo["total"] / resumo["total"].sum() * 100).round(2)
    print(resumo.to_string(index=False))
    return resumo


def salvar_resultados(resultados):
    """Salva todas as tabelas em um único CSV."""
    os.makedirs("dados_tratados", exist_ok=True)

    with open(ARQ_SAIDA, "w", encoding="utf-8") as f:
        for nome, tabela in resultados.items():
            f.write(f"\n===== {nome} =====\n")
            f.write(tabela.to_csv(index=False, sep=";"))
            f.write("\n")

    print(f"\nResultados salvos em: {ARQ_SAIDA}")


if __name__ == "__main__":
    df = carregar_dados()

    resultados = {}
    resultados["Acidentes por ano"] = analise_por_ano(df)
    resultados["Acidentes por mes"] = analise_por_mes(df)
    resultados["Acidentes por dia da semana"] = analise_por_dia_semana(df)
    resultados["Acidentes por hora"] = analise_por_hora(df)
    resultados["Acidentes por condicao meteorologica"] = analise_por_condicao_meteorologica(df)
    resultados["Acidentes por tipo de pista"] = analise_por_tipo_pista(df)
    resultados["Principais causas"] = analise_top_causas(df)
    resultados["Principais tipos de acidente"] = analise_por_tipo_acidente(df)
    resultados["Acidentes por fase do dia"] = analise_por_fase_dia(df)
    resultados["Distribuicao de gravidade"] = analise_por_gravidade(df)

    salvar_resultados(resultados)

    print("\n" + "=" * 60)
    print("ANÁLISE CONCLUÍDA!")
    print("=" * 60)
