# 05_dashboard.py
# Objetivo: Dashboard interativo dos acidentes da PRF com integração de ML.
# Para rodar: streamlit run 05_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import joblib

# CONFIGURAÇÃO DA PÁGINA
st.set_page_config(
    page_title="Acidentes PRF - Paraná",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# CONSTANTES
ARQ_DADOS = os.path.join("dados_tratados", "acidentes_limpos.csv")
ARQ_MODELO = os.path.join("modelo", "modelo_rf.pkl")
ARQ_ENCODERS = os.path.join("modelo", "encoders.pkl")

FEATURES = [
    "hora", "mes", "dia_semana", "fase_dia",
    "condicao_metereologica", "tipo_pista", "tracado_via",
    "uso_solo", "causa_acidente", "tipo_acidente",
]
TARGET = "gravidade"


# CARREGAMENTO DOS DADOS (com cache)
@st.cache_data
def carregar_dados():
    df = pd.read_csv(
        ARQ_DADOS,
        sep=";",
        encoding="utf-8",
        low_memory=False,
        parse_dates=["data_inversa"],
    )
    df = df[df["uf"] == "PR"].copy()
    return df


# CARREGAMENTO DO MODELO E ENCODERS (com cache)
@st.cache_resource
def carregar_modelo():
    if not os.path.exists(ARQ_MODELO) or not os.path.exists(ARQ_ENCODERS):
        return None, None
    modelo = joblib.load(ARQ_MODELO)
    encoders = joblib.load(ARQ_ENCODERS)
    return modelo, encoders


df = carregar_dados()
modelo, encoders = carregar_modelo()


# CABEÇALHO
st.title("Análise de Acidentes nas Rodovias Federais do Paraná")
st.markdown(
    """
    **Fonte:** Polícia Rodoviária Federal (PRF) — Dados Abertos  
    **Período:** 2021 a 2026  
    **Total de registros:** {:,.0f} acidentes  
    """.format(len(df))
)

st.divider()


# SIDEBAR - FILTROS
st.sidebar.header("Filtros")

anos_disponiveis = sorted(df["ano"].dropna().unique().tolist())
anos_selecionados = st.sidebar.multiselect(
    "Ano",
    options=anos_disponiveis,
    default=anos_disponiveis,
)

gravidades_disponiveis = df["gravidade"].unique().tolist()
gravidades_selecionadas = st.sidebar.multiselect(
    "Gravidade",
    options=gravidades_disponiveis,
    default=gravidades_disponiveis,
)

condicoes = df["condicao_metereologica"].dropna().unique().tolist()
condicoes_selecionadas = st.sidebar.multiselect(
    "Condição meteorológica",
    options=condicoes,
    default=condicoes,
)

df_filtrado = df[
    (df["ano"].isin(anos_selecionados))
    & (df["gravidade"].isin(gravidades_selecionadas))
    & (df["condicao_metereologica"].isin(condicoes_selecionadas))
].copy()

st.sidebar.markdown(f"**Registros filtrados:** {len(df_filtrado):,}")


# KPIs PRINCIPAIS
col1, col2, col3, col4 = st.columns(4)

total_acidentes = len(df_filtrado)
total_mortos = int(df_filtrado["mortos"].sum())
total_feridos_graves = int(df_filtrado["feridos_graves"].sum())
taxa_fatalidade = (total_mortos / total_acidentes * 100) if total_acidentes > 0 else 0

col1.metric("Total de Acidentes", f"{total_acidentes:,}")
col2.metric("Total de Mortos", f"{total_mortos:,}")
col3.metric("Feridos Graves", f"{total_feridos_graves:,}")
col4.metric("Taxa de Fatalidade", f"{taxa_fatalidade:.2f}%")

st.divider()


# GRÁFICO 1: Acidentes e mortos por ano
st.subheader("Distribuição Anual de Acidentes e Óbitos")

resumo_ano = df_filtrado.groupby("ano").agg(
    total=("gravidade", "count"),
    mortos=("mortos", "sum"),
).reset_index()

fig1 = go.Figure()
fig1.add_trace(go.Bar(
    x=resumo_ano["ano"],
    y=resumo_ano["total"],
    name="Total de acidentes",
    marker_color="steelblue",
    yaxis="y1",
))
fig1.add_trace(go.Scatter(
    x=resumo_ano["ano"],
    y=resumo_ano["mortos"],
    name="Mortos",
    mode="lines+markers",
    marker=dict(color="crimson", size=10),
    line=dict(width=3),
    yaxis="y2",
))
fig1.update_layout(
    yaxis=dict(title="Total de acidentes", side="left"),
    yaxis2=dict(title="Mortos", overlaying="y", side="right"),
    hovermode="x unified",
    height=400,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig1, width="stretch")


# GRÁFICO 2 e 3: Por mês e por dia da semana
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Distribuição Mensal de Acidentes")
    resumo_mes = df_filtrado.groupby("mes").agg(total=("gravidade", "count")).reset_index()
    fig2 = px.bar(
        resumo_mes, x="mes", y="total",
        color="total", color_continuous_scale="viridis",
        labels={"mes": "Mês", "total": "Acidentes"},
    )
    fig2.update_layout(height=350, showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig2, width="stretch")

with col_b:
    st.subheader("Distribuição por Dia da Semana")
    ordem = ["segunda-feira", "terça-feira", "quarta-feira", "quinta-feira",
             "sexta-feira", "sábado", "domingo"]
    resumo_dia = df_filtrado.groupby("dia_semana").agg(total=("gravidade", "count")).reset_index()
    resumo_dia["dia_semana"] = pd.Categorical(
        resumo_dia["dia_semana"], categories=ordem, ordered=True
    )
    resumo_dia = resumo_dia.sort_values("dia_semana")
    fig3 = px.bar(
        resumo_dia, x="dia_semana", y="total",
        color="total", color_continuous_scale="orrd",
        labels={"dia_semana": "Dia", "total": "Acidentes"},
    )
    fig3.update_layout(height=350, showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig3, width="stretch")


# GRÁFICO 4: Acidentes e taxa de fatalidade por hora
st.subheader("Distribuição Horária e Taxa de Fatalidade")

resumo_hora = df_filtrado.groupby("hora").agg(
    total=("gravidade", "count"),
    mortos=("mortos", "sum"),
).reset_index()
resumo_hora["taxa_fatalidade"] = resumo_hora["mortos"] / resumo_hora["total"] * 100

fig4 = go.Figure()
fig4.add_trace(go.Bar(
    x=resumo_hora["hora"],
    y=resumo_hora["total"],
    name="Total de acidentes",
    marker_color="skyblue",
    yaxis="y1",
))
fig4.add_trace(go.Scatter(
    x=resumo_hora["hora"],
    y=resumo_hora["taxa_fatalidade"],
    name="Taxa de fatalidade (%)",
    mode="lines+markers",
    marker=dict(color="darkred", size=8),
    line=dict(width=3),
    yaxis="y2",
))
fig4.update_layout(
    yaxis=dict(title="Total de acidentes", side="left"),
    yaxis2=dict(title="Taxa de fatalidade (%)", overlaying="y", side="right"),
    hovermode="x unified",
    height=400,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig4, width="stretch")


# GRÁFICO 5 e 6: Tipo de pista e fase do dia
col_c, col_d = st.columns(2)

with col_c:
    st.subheader("Taxa de Fatalidade por Tipo de Pista")
    resumo_pista = df_filtrado.groupby("tipo_pista").agg(
        total=("gravidade", "count"),
        mortos=("mortos", "sum"),
    ).reset_index()
    resumo_pista["taxa_fatalidade"] = resumo_pista["mortos"] / resumo_pista["total"] * 100
    fig5 = px.bar(
        resumo_pista, x="tipo_pista", y="taxa_fatalidade",
        color="taxa_fatalidade", color_continuous_scale="reds",
        labels={"tipo_pista": "Tipo de pista", "taxa_fatalidade": "Taxa de fatalidade (%)"},
        text="taxa_fatalidade",
    )
    fig5.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig5.update_layout(height=350, showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig5, width="stretch")

with col_d:
    st.subheader("Distribuição por Fase do Dia")
    resumo_fase = df_filtrado.groupby("fase_dia").agg(total=("gravidade", "count")).reset_index()
    fig6 = px.pie(
        resumo_fase, names="fase_dia", values="total",
        color_discrete_sequence=px.colors.sequential.Plasma,
        hole=0.4,
    )
    fig6.update_layout(height=350)
    st.plotly_chart(fig6, width="stretch")


# GRÁFICO 7: Causas de acidente
st.subheader("Principais Causas de Acidentes")

resumo_causas = df_filtrado.groupby("causa_acidente").agg(
    total=("gravidade", "count"),
    mortos=("mortos", "sum"),
).reset_index().sort_values("total", ascending=False).head(10)
resumo_causas["taxa_fatalidade"] = resumo_causas["mortos"] / resumo_causas["total"] * 100

fig7 = px.bar(
    resumo_causas, x="total", y="causa_acidente",
    orientation="h",
    color="taxa_fatalidade", color_continuous_scale="orrd",
    labels={"total": "Total de acidentes", "causa_acidente": "",
            "taxa_fatalidade": "Taxa de fatalidade (%)"},
    text="total",
)
fig7.update_traces(texttemplate="%{text:,}", textposition="outside")
fig7.update_layout(height=500, coloraxis_colorbar=dict(title="Taxa fatal. (%)"))
st.plotly_chart(fig7, width="stretch")


# GRÁFICO 8: Tipos de acidente
st.subheader("Principais Tipos de Acidente")

resumo_tipos = df_filtrado.groupby("tipo_acidente").agg(
    total=("gravidade", "count"),
    mortos=("mortos", "sum"),
).reset_index().sort_values("total", ascending=False).head(10)
resumo_tipos["taxa_fatalidade"] = resumo_tipos["mortos"] / resumo_tipos["total"] * 100

fig8 = px.bar(
    resumo_tipos, x="total", y="tipo_acidente",
    orientation="h",
    color="taxa_fatalidade", color_continuous_scale="reds",
    labels={"total": "Total de acidentes", "tipo_acidente": "",
            "taxa_fatalidade": "Taxa de fatalidade (%)"},
    text="total",
)
fig8.update_traces(texttemplate="%{text:,}", textposition="outside")
fig8.update_layout(height=500, coloraxis_colorbar=dict(title="Taxa fatal. (%)"))
st.plotly_chart(fig8, width="stretch")


# SEÇÃO DE MACHINE LEARNING - PREVISÃO DE GRAVIDADE
st.divider()
st.header("Previsão de Gravidade com Machine Learning")
st.markdown(
    """
    Utilize o formulário abaixo para simular as condições de um acidente.
    O modelo **Random Forest** (treinado com 262.788 amostras) irá prever
    a gravidade esperada com base nas variáveis selecionadas.
    """
)

if modelo is None or encoders is None:
    st.warning(
        "Modelo não encontrado. Rode `python3 06_machine_learning.py` "
        "para gerar os arquivos `modelo/modelo_rf.pkl` e `modelo/encoders.pkl`."
    )
else:
    with st.form("form_previsao"):
        col_e, col_f, col_g = st.columns(3)

        with col_e:
            hora_input = st.slider("Hora do dia", 0, 23, 18)
            mes_input = st.slider("Mês", 1, 12, 12)
            dia_input = st.selectbox(
                "Dia da semana",
                options=[
                    "segunda-feira", "terça-feira", "quarta-feira",
                    "quinta-feira", "sexta-feira", "sábado", "domingo",
                ],
                index=4,
            )
            fase_input = st.selectbox(
                "Fase do dia",
                options=["Pleno dia", "Plena Noite", "Anoitecer", "Amanhecer"],
                index=0,
            )

        with col_f:
            condicao_input = st.selectbox(
                "Condição meteorológica",
                options=df["condicao_metereologica"].dropna().unique().tolist(),
            )
            tipo_pista_input = st.selectbox(
                "Tipo de pista",
                options=df["tipo_pista"].dropna().unique().tolist(),
            )
            tracado_input = st.selectbox(
                "Traçado da via",
                options=df["tracado_via"].dropna().unique().tolist(),
            )

        with col_g:
            uso_solo_input = st.selectbox(
                "Uso do solo",
                options=df["uso_solo"].dropna().unique().tolist(),
            )
            causa_input = st.selectbox(
                "Causa do acidente",
                options=df["causa_acidente"].dropna().unique().tolist(),
            )
            tipo_acidente_input = st.selectbox(
                "Tipo de acidente",
                options=df["tipo_acidente"].dropna().unique().tolist(),
            )

        submit = st.form_submit_button("Prever Gravidade", use_container_width=True)

    if submit:
        try:
            entrada = {
                "hora": hora_input,
                "mes": mes_input,
                "dia_semana": dia_input,
                "fase_dia": fase_input,
                "condicao_metereologica": condicao_input,
                "tipo_pista": tipo_pista_input,
                "tracado_via": tracado_input,
                "uso_solo": uso_solo_input,
                "causa_acidente": causa_input,
                "tipo_acidente": tipo_acidente_input,
            }

            linha_codificada = {}
            for feat in FEATURES:
                valor = str(entrada[feat])
                le = encoders[feat]
                if valor in le.classes_:
                    linha_codificada[feat] = int(le.transform([valor])[0])
                else:
                    linha_codificada[feat] = 0
                    st.warning(
                        f"Valor '{valor}' não visto no treino para '{feat}'. "
                        f"Usando valor padrão."
                    )

            X_input = pd.DataFrame([linha_codificada])[FEATURES].astype(float)

            pred = modelo.predict(X_input)[0]
            proba = modelo.predict_proba(X_input)[0]

            le_target = encoders[TARGET]
            classe_prevista = le_target.inverse_transform([pred])[0]

            st.markdown("### Resultado da Previsão")

            cores = {
                "Fatal": "#c0392b",
                "Grave": "#e67e22",
                "Leve": "#f1c40f",
                "Sem Vítimas": "#2ecc71",
            }

            cor = cores.get(classe_prevista, "#888")

            st.markdown(
                f"""
                <div style="
                    background-color: {cor}20;
                    border-left: 8px solid {cor};
                    padding: 20px;
                    border-radius: 8px;
                    margin: 10px 0;
                ">
                    <h2 style="color: {cor}; margin: 0;">
                        Gravidade prevista: {classe_prevista}
                    </h2>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("#### Probabilidade por Classe")
            classes_originais = list(le_target.classes_)
            df_proba = pd.DataFrame({
                "Gravidade": classes_originais,
                "Probabilidade (%)": [round(p * 100, 2) for p in proba],
            }).sort_values("Probabilidade (%)", ascending=True)

            fig_pred = px.bar(
                df_proba,
                x="Probabilidade (%)",
                y="Gravidade",
                orientation="h",
                color="Probabilidade (%)",
                color_continuous_scale="reds",
                text="Probabilidade (%)",
            )
            fig_pred.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_pred.update_layout(
                height=300,
                showlegend=False,
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_pred, width="stretch")

            st.caption(
                "Esta previsão é baseada em padrões estatísticos históricos e "
                "não deve ser utilizada como única fonte para decisões de segurança viária."
            )

        except Exception as e:
            st.error(f"Erro ao fazer a previsão: {e}")


# RODAPÉ
st.divider()
st.caption(
    "Dashboard analítico de acidentes rodoviários — dados públicos da PRF (2021-2026)."
)
