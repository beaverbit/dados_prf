# 04_visualizacao.py
# Objetivo: Gerar os gráficos do TCC e salvar em imagens/.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configurações visuais
sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 11

# Caminhos
ARQ_ENTRADA = os.path.join("dados_tratados", "acidentes_limpos.csv")
PASTA_IMAGENS = "imagens"

UF_FOCO = "PR"  # mude para None se quiser Brasil todo


def carregar_dados():
    print("Carregando dados limpos...")
    df = pd.read_csv(
        ARQ_ENTRADA,
        sep=";",
        encoding="utf-8",
        low_memory=False,
        parse_dates=["data_inversa"],
    )
    if UF_FOCO:
        df = df[df["uf"] == UF_FOCO].copy()
        print(f"Filtrado para UF={UF_FOCO}: {len(df)} linhas")
    return df


def salvar(nome):
    os.makedirs(PASTA_IMAGENS, exist_ok=True)
    caminho = os.path.join(PASTA_IMAGENS, nome)
    plt.tight_layout()
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Salvo: {caminho}")


# ============================================================
# GRÁFICO 1: Acidentes e mortos por ano
# ============================================================
def grafico_por_ano(df):
    print("\nGerando gráfico 1: Acidentes por ano...")
    resumo = df.groupby("ano").agg(
        total=("gravidade", "count"),
        mortos=("mortos", "sum"),
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(resumo["ano"], resumo["total"], color="steelblue", label="Total de acidentes")
    ax1.set_xlabel("Ano")
    ax1.set_ylabel("Total de acidentes", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    ax2 = ax1.twinx()
    ax2.plot(resumo["ano"], resumo["mortos"], color="crimson", marker="o",
             linewidth=2, label="Mortos")
    ax2.set_ylabel("Total de mortos", color="crimson")
    ax2.tick_params(axis="y", labelcolor="crimson")

    plt.title(f"Acidentes e mortos por ano — UF: {UF_FOCO}")
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
    salvar("01_acidentes_por_ano.png")


# ============================================================
# GRÁFICO 2: Acidentes por mês
# ============================================================
def grafico_por_mes(df):
    print("\nGerando gráfico 2: Acidentes por mês...")
    resumo = df.groupby("mes").agg(total=("gravidade", "count")).reset_index()

    plt.figure(figsize=(12, 6))
    sns.barplot(data=resumo, x="mes", y="total", palette="viridis")
    plt.title(f"Distribuição de acidentes por mês — UF: {UF_FOCO}")
    plt.xlabel("Mês")
    plt.ylabel("Total de acidentes")
    salvar("02_acidentes_por_mes.png")


# ============================================================
# GRÁFICO 3: Acidentes por dia da semana
# ============================================================
def grafico_por_dia_semana(df):
    print("\nGerando gráfico 3: Acidentes por dia da semana...")
    ordem = ["segunda-feira", "terça-feira", "quarta-feira", "quinta-feira",
             "sexta-feira", "sábado", "domingo"]
    resumo = df.groupby("dia_semana").agg(total=("gravidade", "count")).reset_index()
    resumo["dia_semana"] = pd.Categorical(resumo["dia_semana"], categories=ordem, ordered=True)
    resumo = resumo.sort_values("dia_semana")

    plt.figure(figsize=(12, 6))
    sns.barplot(data=resumo, x="dia_semana", y="total", palette="rocket")
    plt.title(f"Acidentes por dia da semana — UF: {UF_FOCO}")
    plt.xlabel("Dia da semana")
    plt.ylabel("Total de acidentes")
    plt.xticks(rotation=30)
    salvar("03_acidentes_por_dia_semana.png")


# ============================================================
# GRÁFICO 4: Acidentes e taxa de fatalidade por hora
# ============================================================
def grafico_por_hora(df):
    print("\nGerando gráfico 4: Acidentes por hora...")
    resumo = df.groupby("hora").agg(
        total=("gravidade", "count"),
        mortos=("mortos", "sum"),
    ).reset_index()
    resumo["taxa_fatalidade"] = resumo["mortos"] / resumo["total"] * 100

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.bar(resumo["hora"], resumo["total"], color="skyblue", label="Total de acidentes")
    ax1.set_xlabel("Hora do dia")
    ax1.set_ylabel("Total de acidentes", color="skyblue")
    ax1.tick_params(axis="y", labelcolor="skyblue")

    ax2 = ax1.twinx()
    ax2.plot(resumo["hora"], resumo["taxa_fatalidade"], color="darkred",
             marker="o", linewidth=2, label="Taxa de fatalidade (%)")
    ax2.set_ylabel("Taxa de fatalidade (%)", color="darkred")
    ax2.tick_params(axis="y", labelcolor="darkred")

    plt.title(f"Acidentes e taxa de fatalidade por hora — UF: {UF_FOCO}")
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
    salvar("04_acidentes_por_hora.png")


# ============================================================
# GRÁFICO 5: Acidentes por condição meteorológica
# ============================================================
def grafico_condicao_meteorologica(df):
    print("\nGerando gráfico 5: Condição meteorológica...")
    resumo = df.groupby("condicao_metereologica").agg(
        total=("gravidade", "count"),
    ).reset_index().sort_values("total", ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=resumo, x="total", y="condicao_metereologica", palette="coolwarm")
    plt.title(f"Acidentes por condição meteorológica — UF: {UF_FOCO}")
    plt.xlabel("Total de acidentes")
    plt.ylabel("Condição meteorológica")
    salvar("05_condicao_meteorologica.png")


# ============================================================
# GRÁFICO 6: Taxa de fatalidade por tipo de pista
# ============================================================
def grafico_tipo_pista(df):
    print("\nGerando gráfico 6: Tipo de pista...")
    resumo = df.groupby("tipo_pista").agg(
        total=("gravidade", "count"),
        mortos=("mortos", "sum"),
    ).reset_index()
    resumo["taxa_fatalidade"] = resumo["mortos"] / resumo["total"] * 100

    fig, ax1 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=resumo, x="tipo_pista", y="total", ax=ax1,
                palette="mako", alpha=0.7)
    ax1.set_ylabel("Total de acidentes")

    ax2 = ax1.twinx()
    sns.lineplot(data=resumo, x="tipo_pista", y="taxa_fatalidade",
                 ax=ax2, color="crimson", marker="o", linewidth=3)
    ax2.set_ylabel("Taxa de fatalidade (%)", color="crimson")
    ax2.tick_params(axis="y", labelcolor="crimson")

    plt.title(f"Acidentes e taxa de fatalidade por tipo de pista — UF: {UF_FOCO}")
    salvar("06_tipo_pista.png")


# ============================================================
# GRÁFICO 7: Top 10 causas de acidentes
# ============================================================
def grafico_top_causas(df):
    print("\nGerando gráfico 7: Top 10 causas...")
    resumo = df.groupby("causa_acidente").agg(
        total=("gravidade", "count"),
    ).reset_index().sort_values("total", ascending=False).head(10)

    plt.figure(figsize=(12, 7))
    sns.barplot(data=resumo, x="total", y="causa_acidente", palette="flare")
    plt.title(f"Top 10 causas de acidentes — UF: {UF_FOCO}")
    plt.xlabel("Total de acidentes")
    plt.ylabel("")
    salvar("07_top_causas.png")


# ============================================================
# GRÁFICO 8: Distribuição de gravidade
# ============================================================
def grafico_gravidade(df):
    print("\nGerando gráfico 8: Distribuição de gravidade...")
    resumo = df["gravidade"].value_counts().reset_index()
    resumo.columns = ["gravidade", "total"]

    cores = {"Sem Vítimas": "#2ecc71", "Leve": "#f1c40f",
             "Grave": "#e67e22", "Fatal": "#c0392b"}
    cores_lista = [cores[g] for g in resumo["gravidade"]]

    plt.figure(figsize=(10, 6))
    plt.pie(resumo["total"], labels=resumo["gravidade"], autopct="%1.1f%%",
            colors=cores_lista, startangle=90, textprops={"fontsize": 12})
    plt.title(f"Distribuição de gravidade dos acidentes — UF: {UF_FOCO}")
    salvar("08_gravidade.png")


if __name__ == "__main__":
    df = carregar_dados()

    grafico_por_ano(df)
    grafico_por_mes(df)
    grafico_por_dia_semana(df)
    grafico_por_hora(df)
    grafico_condicao_meteorologica(df)
    grafico_tipo_pista(df)
    grafico_top_causas(df)
    grafico_gravidade(df)

    print("\n" + "="*60)
    print("GRÁFICOS GERADOS COM SUCESSO!")
    print(f"Todos salvos em: {PASTA_IMAGENS}/")
    print("="*60)