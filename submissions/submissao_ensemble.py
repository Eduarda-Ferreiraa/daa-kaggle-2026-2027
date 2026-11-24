"""Submissão do ensemble por stacking (ensemble_submission.csv).

Três modelos, todos com os mesmos 5 folds (StratifiedKFold, random_state=42):
- HistGradientBoosting (probabilidades das 16 faixas)
- Random Forest (probabilidades das 16 faixas)
- Regressor ordinal HGB (previsão contínua da faixa)

O modelo final (regressão logística) aprende a combiná-los a partir das previsões out-of-fold
(cada pessoa prevista por modelos que não a viram no treino). Nos dados sem WGT:
HGB sozinho 0.470 | RF 0.463 | stacking 0.479 (melhor que o HGB em 4/5 folds).

Executar a partir da pasta diana/ depois de correr notebooks/tratamento.ipynb:
    python scripts/submissao_ensemble.py
Demora cerca de 7 minutos.
"""
import time
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

BASE = Path(__file__).resolve().parent.parent
DADOS = BASE / "dados"
SAIDA = BASE / "submissions"
SAIDA.mkdir(exist_ok=True)

inicio = time.time()
train = pd.read_parquet(DADOS / "train_tratado.parquet")
test = pd.read_parquet(DADOS / "test_tratado.parquet")
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


hgb = HistGradientBoostingClassifier(
    categorical_features="from_dtype", class_weight="balanced", early_stopping=False, random_state=42,
    learning_rate=0.15, max_iter=200, max_leaf_nodes=15, min_samples_leaf=10,
    max_features=1.0, l2_regularization=0.5)
rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", max_features=0.85, min_samples_leaf=3,
                            random_state=42, n_jobs=-1)
reg = HistGradientBoostingRegressor(
    categorical_features="from_dtype", early_stopping=False, random_state=42, loss="squared_error",
    learning_rate=0.03, max_iter=400, max_leaf_nodes=63, min_samples_leaf=10,
    max_features=0.7, l2_regularization=0)

# Previsões out-of-fold (treino) e previsões do teste (modelos treinados com todo o treino)
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
f1_hgb, f1_stack = [], []
for a, b in folds:
    f1_hgb.append(f1_score(y[b], oof_hgb[b].argmax(1), average="macro"))
    f1_stack.append(f1_score(y[b], criar_stacking().fit(Z[a], y[a]).predict(Z[b]), average="macro"))
print(f"HGB sozinho {np.mean(f1_hgb):.4f} ± {np.std(f1_hgb):.4f} | "
      f"stacking {np.mean(f1_stack):.4f} ± {np.std(f1_stack):.4f} | "
      f"stacking melhor em {(np.array(f1_stack) > np.array(f1_hgb)).sum()}/5 folds")

sub = pd.read_csv(BASE / "example_submission_10.csv")
sub["AGE_RANGE"] = criar_stacking().fit(Z, y).predict(entradas(test_hgb, test_rf, test_reg)).astype(int)
assert len(sub) == len(X_test) and sub["RowId"].tolist() == list(range(len(X_test)))
sub.to_csv(SAIDA / "ensemble_submission.csv", index=False)
print(sub["AGE_RANGE"].value_counts().sort_index().to_dict(), f"| {(time.time() - inicio) / 60:.1f} min")
