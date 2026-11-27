"""Stacking HGB afinado + Random Forest + regressor ordinal -> submissão.

Uso (a partir de uma pasta ao lado de dados/, p.ex. notebooks/ ou submissions/):
    python submissao_stacking.py <treino.parquet> <teste.parquet> <nome_submissao.csv>
ex.: python submissao_stacking.py ../dados/train_tratado.parquet ../dados/test_tratado.parquet stacking_dados_originais.csv
     python submissao_stacking.py ../dados/train_tratado_v2.parquet ../dados/test_tratado_v2.parquet stacking_dados_tratados.csv
Os parquets vêm de tratamento.ipynb (dados originais) e notebooks/tratamento_v2.ipynb (dados tratados).
Macro F1 em validação cruzada: 0.4825 (dados originais) e 0.4849 (dados tratados); Kaggle público: 0.50 nas duas.

Todos os modelos usam os mesmos 5 folds (StratifiedKFold, random_state=42). As previsões out-of-fold
(cada pessoa prevista por modelos que não a viram) treinam a regressão logística que combina os três.
"""
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

treino_path, teste_path, nome = sys.argv[1:4]
inicio = time.time()
train = pd.read_parquet(treino_path)
test = pd.read_parquet(teste_path)
X, y = train.drop(columns="AGE_RANGE"), train["AGE_RANGE"].values
X_test = test[X.columns]
folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
pesos = compute_sample_weight("balanced", y)


def para_random_forest(df):
    """O RandomForest só aceita números: categorias -> código, valores em falta -> -1."""
    df = df.copy()
    for c in df.select_dtypes("category"):
        df[c] = df[c].cat.codes
    return df.fillna(-1)


# HGB afinado sem WGT (mais lento e mais regularizado do que a procura 2)
hgb = HistGradientBoostingClassifier(
    categorical_features="from_dtype", class_weight="balanced", early_stopping=False, random_state=42,
    learning_rate=0.05, max_iter=500, max_leaf_nodes=15, min_samples_leaf=20,
    max_features=0.5, l2_regularization=1.0)
rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", max_features=0.85, min_samples_leaf=3,
                            random_state=42, n_jobs=-1)
reg = HistGradientBoostingRegressor(
    categorical_features="from_dtype", early_stopping=False, random_state=42, loss="squared_error",
    learning_rate=0.03, max_iter=400, max_leaf_nodes=63, min_samples_leaf=10,
    max_features=0.7, l2_regularization=0)

oof_hgb = cross_val_predict(hgb, X, y, cv=folds, method="predict_proba", n_jobs=5)
oof_rf = cross_val_predict(rf, para_random_forest(X), y, cv=folds, method="predict_proba")
oof_reg = cross_val_predict(reg, X, y, cv=folds, params={"sample_weight": pesos}, n_jobs=5)
test_hgb = hgb.fit(X, y).predict_proba(X_test)
test_rf = rf.fit(para_random_forest(X), y).predict_proba(para_random_forest(X_test))
test_reg = reg.fit(X, y, sample_weight=pesos).predict(X_test)


def entradas(p_hgb, p_rf, r):
    """Probabilidades em logaritmo (combinar evidência) + previsão contínua do regressor."""
    return np.hstack([np.log(np.clip(p_hgb, 1e-6, 1)), np.log(np.clip(p_rf, 1e-6, 1)), r[:, None]])


def criar_stacking():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000))


Z = entradas(oof_hgb, oof_rf, oof_reg)
f1 = {"HGB": [], "RF": [], "stacking": []}
for a, b in folds:
    f1["HGB"].append(f1_score(y[b], oof_hgb[b].argmax(1), average="macro"))
    f1["RF"].append(f1_score(y[b], oof_rf[b].argmax(1), average="macro"))
    f1["stacking"].append(f1_score(y[b], criar_stacking().fit(Z[a], y[a]).predict(Z[b]), average="macro"))
for k, v in f1.items():
    print(f"{k:9s} {np.mean(v):.4f} ± {np.std(v):.4f}  {np.round(v, 4).tolist()}")
print("stacking melhor que o HGB em", (np.array(f1["stacking"]) > np.array(f1["HGB"])).sum(), "/5 folds")

sub = pd.read_csv("../dados/example_submission_10.csv")
sub["AGE_RANGE"] = criar_stacking().fit(Z, y).predict(entradas(test_hgb, test_rf, test_reg)).astype(int)
assert len(sub) == len(X_test) and sub["RowId"].tolist() == list(range(len(X_test)))
Path("../submissions").mkdir(exist_ok=True)
sub.to_csv(Path("../submissions") / nome, index=False)
print("gravado ../submissions/" + nome, "|", sub["AGE_RANGE"].value_counts().sort_index().to_dict(),
      f"| {(time.time() - inicio) / 60:.1f} min")
