"""Submissão HistGradientBoosting (hgb_submission.csv).

HGB com os hiperparâmetros da procura 2, treinado em todo o treino tratado.
Macro F1 em validação cruzada (5 folds, random_state=42): 0.470 nos dados sem WGT.

Executar a partir da pasta diana/ depois de correr notebooks/tratamento.ipynb:
    python scripts/submissao_hgb.py
"""
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

BASE = Path(__file__).resolve().parent.parent
DADOS = BASE / "dados"
SAIDA = BASE / "submissions"
SAIDA.mkdir(exist_ok=True)

train = pd.read_parquet(DADOS / "train_tratado.parquet")
test = pd.read_parquet(DADOS / "test_tratado.parquet")
X, y = train.drop(columns="AGE_RANGE"), train["AGE_RANGE"]
assert list(test.columns) == list(X.columns)

modelo = HistGradientBoostingClassifier(
    categorical_features="from_dtype", class_weight="balanced", early_stopping=False, random_state=42,
    learning_rate=0.15, max_iter=200, max_leaf_nodes=15, min_samples_leaf=10,
    max_features=1.0, l2_regularization=0.5,
).fit(X, y)

sub = pd.DataFrame({"RowId": range(len(test)), "AGE_RANGE": modelo.predict(test).astype(int)})
exemplo = pd.read_csv(DADOS / "example_submission_10.csv")
assert list(sub.columns) == list(exemplo.columns) and len(sub) == len(exemplo)
sub.to_csv(SAIDA / "hgb_submission.csv", index=False)
print(sub["AGE_RANGE"].value_counts().sort_index().to_dict())
