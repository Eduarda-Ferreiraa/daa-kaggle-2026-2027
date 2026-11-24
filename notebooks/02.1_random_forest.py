"""
DAA 2026/2027 - Competição Kaggle (previsão de AGE_RANGE)
Modelo 2: Random Forest (versão corrigida)
Métrica: Macro F1 (validação cruzada estratificada)

Correções face à versão anterior
--------------------------------
1. Dados: usa os dados de tratamento.ipynb (../dados/*_tratado.parquet), os mesmos de todos os outros
   modelos. O pré-processamento próprio foi retirado: dava o mesmo F1 (0,437 vs 0,435), não convertia
   PE7/PE9 (HHMM) para minutos, dava códigos arbitrários às colunas nominais e decidia a codificação
   com treino + teste juntos.
2. Removida a variável HH_MEMBER (HHID % 100): foi analisada e rejeitada (não representa a posição
   no agregado); retirá-la muda o F1 em 0,003, dentro do ruído.
3. Tuning centrado no hiperparâmetro que importa: max_features. Com "sqrt" cada divisão só vê ~17 das
   ~300 colunas e quase nunca as poucas colunas fortes (bloco PC/HA/IN15, situação profissional...).
   Medido nos mesmos 5 folds (300 árvores, min_samples_leaf=3):
       sqrt 0,439 | 0,3 0,483 | 0,5 0,493 | 0,7 0,506 | 0,85 0,512 | 1,0 0,511
   A grelha anterior ia só até 0,5. 300 árvores chegam (mais árvores só estabilizam, não sobem o F1).
4. Mesmos 5 folds dos outros modelos (StratifiedKFold, random_state=42) no tuning e nas previsões
   out-of-fold: permite comparar fold a fold e juntar as probabilidades no ensemble.
   (O F1 do melhor de 12 combinações é ligeiramente otimista; com 12 combinações o efeito é pequeno.)
5. Mantém-se: diagnóstico de overfitting por fold, probabilidades OOF, ajuste da decisão com pesos
   por classe (avaliado sem leakage), importâncias por permutação e submissão.

Executar a partir da pasta notebooks/:  python 02.1_random_forest.py
Para um teste rápido (poucas árvores e combinações): QUICK=1 python 02.1_random_forest.py
"""
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

DATA_DIR = "../dados"
OUT_DIR = "../resultados"              # ficheiros com previsões: não vão para o GitHub (.gitignore)
SUB_DIR = "../submissions"
SEED = 42
N_SPLITS = 5
N_JOBS = -1
QUICK = os.environ.get("QUICK") == "1"
N_ARVORES = 50 if QUICK else 300
N_PERM_REPEATS = 1 if QUICK else 3     # repetições da importância por permutação
TARGET = "AGE_RANGE"
os.makedirs(OUT_DIR, exist_ok=True)


# ================================================================ 1. Dados
def para_random_forest(df):
    """O RandomForest do scikit-learn só aceita números: categorias -> código (pela ordem do treino).

    As categorias do teste já foram alinhadas com as do treino no tratamento, por isso o mesmo código
    significa a mesma categoria nos dois. Valores em falta -> -1 (abaixo de qualquer valor real,
    a árvore pode separá-los com um só corte)."""
    df = df.copy()
    for c in df.select_dtypes("category").columns:
        df[c] = df[c].cat.codes                 # NaN -> -1
    return df.fillna(-1)


train = pd.read_parquet(f"{DATA_DIR}/train_tratado.parquet")
test = pd.read_parquet(f"{DATA_DIR}/test_tratado.parquet")
X = para_random_forest(train.drop(columns=TARGET))
y = train[TARGET].values
X_test = para_random_forest(test)[X.columns]
sub = pd.read_csv("../example_submission_10.csv")
n_classes = len(np.unique(y))
print(f"Treino: {X.shape} | Teste: {X_test.shape}")


# ================================================================ 2. Ajuste da decisão para Macro F1
def fit_class_weights(proba, y, n_rounds=3, grid=np.linspace(0.5, 2.5, 21)):
    """Pesos por classe w: previsão = argmax(proba * w). Subida coordenada sobre o Macro F1."""
    w = np.ones(proba.shape[1])
    best = f1_score(y, (proba * w).argmax(1), average="macro")
    for _ in range(n_rounds):
        for k in range(len(w)):
            for g in grid:
                w2 = w.copy(); w2[k] = g
                s = f1_score(y, (proba * w2).argmax(1), average="macro")
                if s > best + 1e-6:
                    best, w = s, w2
    return w


def honest_score(fit_fn, apply_fn, X, y, folds):
    """Ajusta em k-1 folds das previsões OOF e avalia no fold que ficou de fora."""
    pred = np.zeros(len(y), dtype=int)
    for a, b in folds:
        params = fit_fn(X[a], y[a])
        pred[b] = apply_fn(X[b], params)
    return f1_score(y, pred, average="macro"), pred


# ================================================================ 3. Tuning
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)   # os mesmos folds de todos os modelos
folds = list(cv.split(X, y))

param_grid = {
    "max_features": [0.5, 0.7, 0.85, 1.0],
    "min_samples_leaf": [1, 3, 5],
}
if QUICK:
    param_grid = {"max_features": [0.5, 0.85], "min_samples_leaf": [3]}

search = GridSearchCV(
    RandomForestClassifier(n_estimators=N_ARVORES, class_weight="balanced", random_state=SEED, n_jobs=N_JOBS),
    param_grid=param_grid,
    scoring="f1_macro",
    cv=folds,
    n_jobs=1,          # a paralelização já é feita dentro da Random Forest
    verbose=1,
    refit=False,
    return_train_score=True,
)
t0 = time.time()
search.fit(X, y)
print(f"\nTuning concluído em {(time.time() - t0) / 60:.1f} min")

res = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
cols = ["mean_test_score", "std_test_score", "mean_train_score"] + [c for c in res.columns if c.startswith("param_")]
res[cols].to_csv(f"{OUT_DIR}/resultados_rf_cv.csv", index=False)
print(res[cols].round(4).to_string(index=False))
best_params = search.best_params_
print("\nMelhores hiperparâmetros:", best_params)
print(f"Macro F1 no tuning: {search.best_score_:.4f}")


def build_model():
    return RandomForestClassifier(n_estimators=N_ARVORES, class_weight="balanced", random_state=SEED,
                                  n_jobs=N_JOBS, **best_params)


# ================================================================ 4. Previsões out-of-fold (mesmos folds)
oof_proba = np.zeros((len(y), n_classes))
rows = []
for k, (a, b) in enumerate(folds, 1):
    m = build_model().fit(X.iloc[a], y[a])
    oof_proba[b] = m.predict_proba(X.iloc[b])
    f_tr = f1_score(y[a], m.predict(X.iloc[a]), average="macro")
    f_va = f1_score(y[b], oof_proba[b].argmax(1), average="macro")
    rows.append({"fold": k, "f1_treino": f_tr, "f1_validacao": f_va})
    if k == N_SPLITS:  # importância por permutação no último fold de validação
        perm = permutation_importance(m, X.iloc[b], y[b], scoring="f1_macro",
                                      n_repeats=N_PERM_REPEATS, random_state=SEED, n_jobs=1)

folds_df = pd.DataFrame(rows)
print("\nDiagnóstico de overfitting (por fold):")
print(folds_df.round(4).to_string(index=False))
print(f"Média: treino {folds_df.f1_treino.mean():.4f} | validação {folds_df.f1_validacao.mean():.4f}"
      f" ± {folds_df.f1_validacao.std(ddof=0):.4f}")
print("Nota: F1 de treino ≈ 1 é normal numa Random Forest (cada árvore decora a sua amostra; a floresta"
      " generaliza pela média). Subir min_samples_leaf baixa o treino mas quase não mexe na validação.")

pred = oof_proba.argmax(1)
print(f"\nMacro F1 (argmax): {f1_score(y, pred, average='macro'):.4f} | Accuracy: {(pred == y).mean():.4f}"
      f" | MAE (classes): {np.abs(pred - y).mean():.3f} | ≤1 classe: {(np.abs(pred - y) <= 1).mean():.3f}")

# ================================================================ 5. Ajuste da decisão (pesos por classe)
# Pesos aprendidos em 4 folds das OOF e avaliados no 5.º (mesmos folds: nenhuma pessoa avalia os pesos que ajudou a escolher)
f1_adj, pred_adj = honest_score(fit_class_weights, lambda p, w: (p * w).argmax(1), oof_proba, y, folds)
class_w = fit_class_weights(oof_proba, y)  # pesos finais, aprendidos com todas as OOF
print(f"Macro F1 com pesos por classe (avaliação honesta): {f1_adj:.4f}")
print("Pesos por classe:", np.round(class_w, 2).tolist())

per_class = pd.DataFrame({
    "f1_argmax": f1_score(y, pred, average=None),
    "f1_ajustado": f1_score(y, pred_adj, average=None),
    "n": np.bincount(y),
})
print("\nF1 por classe:")
print(per_class.round(3).to_string())
pd.DataFrame(confusion_matrix(y, pred_adj)).to_csv(f"{OUT_DIR}/matriz_confusao_rf.csv")
blk = y >= 9
print(f"Erros entre os blocos 0–8 e 9–15 (65+): {((pred_adj >= 9) != blk).sum()}")

# Probabilidades OOF + diferença real/prevista (base para o ensemble e para a análise de anomalias)
oof_df = pd.DataFrame(oof_proba, columns=[f"p_{c}" for c in range(n_classes)])
oof_df.insert(0, "PID", pd.read_csv("dataset_train.csv", usecols=["PID"]).PID.values)
oof_df["real"] = y
oof_df["prevista"] = pred_adj
oof_df["idade_esperada"] = (oof_proba * np.arange(n_classes)).sum(1)
oof_df["diferenca"] = oof_df["prevista"] - oof_df["real"]
oof_df.to_csv(f"{OUT_DIR}/oof_rf.csv", index=False)

# ================================================================ 6. Importâncias por permutação
imp = (pd.DataFrame({"importance": perm.importances_mean, "std": perm.importances_std}, index=X.columns)
       .sort_values("importance", ascending=False))
imp.to_csv(f"{OUT_DIR}/importancias_rf.csv")
print("\nTop 20 variáveis (permutação, queda no Macro F1):")
print(imp.head(20).round(4).to_string())

# ================================================================ 7. Treino final e submissão
final = build_model().fit(X, y)
test_proba = final.predict_proba(X_test)
pd.DataFrame(test_proba, columns=[f"p_{c}" for c in range(n_classes)]).to_csv(
    f"{OUT_DIR}/test_proba_rf.csv", index=False)

sub[TARGET] = (test_proba * class_w).argmax(1)
assert len(sub) == len(X_test)
sub.to_csv(f"{SUB_DIR}/rf_submission.csv", index=False)
print(f"\nSubmissão guardada em {SUB_DIR}/rf_submission.csv")
print(sub[TARGET].value_counts().sort_index().to_dict())
