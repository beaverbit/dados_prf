# 06_machine_learning.py
# Objetivo: Treinar um modelo de ML para prever a gravidade de acidentes.
# Algoritmo: Random Forest Classifier
# Saída: métricas, matriz de confusão e importância das features.

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score,
)
import joblib

# CONFIGURAÇÃO
ARQ_ENTRADA = os.path.join("dados_tratados", "acidentes_limpos.csv")
PASTA_MODELO = "modelo"
PASTA_IMAGENS = "imagens"
UF_FOCO = "PR"

# Features que vou usar para prever a gravidade
FEATURES = [
    "hora", "mes", "dia_semana", "fase_dia",
    "condicao_metereologica", "tipo_pista", "tracado_via",
    "uso_solo", "causa_acidente", "tipo_acidente",
]

# Variável alvo
TARGET = "gravidade"

def carregar_dados():
    print("Carregando dados...")
    df = pd.read_csv(
        ARQ_ENTRADA, sep=";", encoding="utf-8",
        low_memory=False, parse_dates=["data_inversa"],
    )
    if UF_FOCO:
        df = df[df["uf"] == UF_FOCO].copy()
        print(f"Filtrado para UF={UF_FOCO}: {len(df)} linhas")
    return df

def preparar_dados(df):
    """Seleciona features, remove nulos e codifica variáveis categóricas."""
    print("\nPreparando dados para o modelo...")

    # Mantém só as colunas necessárias
    colunas = FEATURES + [TARGET]
    df = df[colunas].copy()

    # Remove linhas com valores nulos nas features
    antes = len(df)
    df = df.dropna()
    print(f"Removidas {antes - len(df)} linhas com valores nulos.")

    # Codificação: tudo vira string, aplica LabelEncoder, força int
    encoders = {}
    for col in FEATURES:
        serie_str = df[col].astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(serie_str)
        df[col] = df[col].astype(int)
        encoders[col] = le
        print(f"  {col}: codificada ({len(le.classes_)} valores únicos)")

    # Codifica a variável alvo também
    le_target = LabelEncoder()
    df[TARGET] = le_target.fit_transform(df[TARGET].astype(str))
    df[TARGET] = df[TARGET].astype(int)
    encoders[TARGET] = le_target

    # Validação final: garante que X é todo numérico
    X = df[FEATURES]
    tipos_unicos = X.dtypes.unique()
    print(f"\nTipos das features após codificação: {tipos_unicos}")
    if not all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns):
        raise ValueError("Ainda existem colunas não-numéricas em X!")

    print(f"Total de amostras: {len(df)}")
    print(f"Classes do target: {list(le_target.classes_)}")
    print(f"Distribuição:\n{df[TARGET].value_counts()}")

    return df, encoders

def treinar_modelo(df):
    """Treina o Random Forest e avalia."""
    print("\n" + "="*60)
    print("TREINANDO MODELO RANDOM FOREST")
    print("="*60)

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    X = X.astype(float)
    y = y.astype(int)

    # Split treino/teste (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    print(f"Treino: {len(X_train)} amostras")
    print(f"Teste:  {len(X_test)} amostras")

    # Modelo
    print("\nTreinando Random Forest (pode demorar 1-2 minutos)...")
    modelo = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
    )
    modelo.fit(X_train, y_train)

    # Previsões
    y_pred = modelo.predict(X_test)

    # Métricas
    print("\n" + "="*60)
    print("RESULTADOS")
    print("="*60)
    print(f"\nAcurácia: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1-Score (weighted): {f1_score(y_test, y_pred, average='weighted'):.4f}")
    print("\nRelatório por classe:")
    print(classification_report(y_test, y_pred))

    return modelo, X_test, y_test, y_pred

def plotar_matriz_confusao(y_test, y_pred, encoders):
    print("\nGerando matriz de confusão...")
    cm = confusion_matrix(y_test, y_pred)
    labels = encoders[TARGET].classes_

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
    )
    plt.title("Matriz de Confusão - Random Forest")
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.tight_layout()
    os.makedirs(PASTA_IMAGENS, exist_ok=True)
    caminho = os.path.join(PASTA_IMAGENS, "09_matriz_confusao.png")
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Salvo: {caminho}")

def plotar_importancia_features(modelo):
    print("\nGerando gráfico de importância das features...")
    importancias = pd.DataFrame({
        "feature": FEATURES,
        "importancia": modelo.feature_importances_,
    }).sort_values("importancia", ascending=True)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=importancias, x="importancia", y="feature", palette="viridis",
                hue="feature", legend=False)
    plt.title("Importância das Features - Random Forest")
    plt.xlabel("Importância")
    plt.ylabel("")
    plt.tight_layout()
    caminho = os.path.join(PASTA_IMAGENS, "10_importancia_features.png")
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Salvo: {caminho}")

    print("\nTop 5 features mais importantes:")
    print(importancias.tail(5).to_string(index=False))

def salvar_modelo(modelo, encoders):
    print("\nSalvando modelo e encoders...")
    os.makedirs(PASTA_MODELO, exist_ok=True)
    joblib.dump(modelo, os.path.join(PASTA_MODELO, "modelo_rf.pkl"))
    joblib.dump(encoders, os.path.join(PASTA_MODELO, "encoders.pkl"))
    print(f"  -> Modelo salvo em: {PASTA_MODELO}/modelo_rf.pkl")
    print(f"  -> Encoders salvos em: {PASTA_MODELO}/encoders.pkl")

if __name__ == "__main__":
    df = carregar_dados()
    df_prep, encoders = preparar_dados(df)
    modelo, X_test, y_test, y_pred = treinar_modelo(df_prep)

    # ORDEM CORRIGIDA: salvar primeiro, plotar depois
    salvar_modelo(modelo, encoders)
    plotar_matriz_confusao(y_test, y_pred, encoders)
    plotar_importancia_features(modelo)

    print("\n" + "="*60)
    print("MACHINE LEARNING CONCLUÍDO!")
    print("="*60)