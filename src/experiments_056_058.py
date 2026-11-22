"""DAA 056–058. Seleção só por CV dentro do treino; importação sem treinos."""
from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from src.preparar_dados import preparar_dados, make_stratified_split, build_preprocessor

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports"
OUTPUT = ROOT / "outputs/experiments_056_058"
RESULTS = REPORT / ".experiments_056_058_results.json"
CLASSES = np.arange(16)
SEED = 42

# Configurações confirmadas nas fontes de 007, 025 e no resultado real de 041.
LGB_CONTROL = dict(num_leaves=31, learning_rate=0.03, n_estimators=126,
                   min_child_samples=20, reg_alpha=0.0, reg_lambda=0.0)
XGB_CONTROL = dict(n_estimators=200, max_depth=5, learning_rate=0.1,
                   min_child_weight=1, subsample=1.0, colsample_bytree=1.0)
MLP_CONTROL = dict(epochs=85, learning_rate=0.0001, weighting="inverse_sqrt", dropout=0.1)


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _indices(frame):
    return _digest(frame.index.tolist())


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _versions():
    out = {}
    for name in ["numpy", "pandas", "scipy", "scikit-learn", "torch", "xgboost", "lightgbm"]:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def _reference():
    paths = [ROOT / "example_submission_10.csv", ROOT / "notebooks/example_submission_10.csv",
             ROOT / "submissions/example_submission_10.csv"]
    path = next((p for p in paths if p.is_file()), None)
    if path is None:
        raise FileNotFoundError("Falta o exemplo oficial example_submission_10.csv; não inventar RowIds.")
    reference = pd.read_csv(path)
    if not {"RowId", "AGE_RANGE"}.issubset(reference.columns):
        raise ValueError("O exemplo oficial precisa de RowId e AGE_RANGE.")
    if len(reference) != 3655 or reference.RowId.isna().any() or not reference.RowId.is_unique:
        raise ValueError("RowIds oficiais inválidos ou número de linhas diferente de 3655.")
    return path, reference[["RowId"]].copy()


def context():
    reference, _ = _reference()
    files = [ROOT / "notebooks/dataset_train.csv", ROOT / "notebooks/dataset_test_X.csv",
             ROOT / "src/preparar_dados.py", ROOT / "src/experiments_016_018.py",
             ROOT / "src/experiments_022_027.py", reference]
    shared = [_hash, _digest, _indices, _data, _metrics, _smooth_weights,
              _estimator, _probabilities, _check_probabilities, _dense, _fit_cached]
    code = "\n".join(inspect.getsource(fn) for fn in shared)
    return {"files": {str(p.relative_to(ROOT)): _hash(p) for p in files},
            "versions": _versions(), "seed": SEED, "cv": 3,
            "shared_code_sha256": hashlib.sha256(code.encode()).hexdigest(),
            "controls": {"lgb": LGB_CONTROL, "xgb": XGB_CONTROL, "mlp": MLP_CONTROL},
            "runtime": {"python": platform.python_version(), "system": platform.system(), "machine": platform.machine()}}


def result_context(k):
    functions = [_ensemble] if k == "058" else [_trees, _cv, _grid]
    functions += [_folds, _paired, _export, _plots, _reference]
    code = "\n".join(inspect.getsource(fn) for fn in functions)
    return {**context(), "experiment": k, "procedure_code_sha256": hashlib.sha256(code.encode()).hexdigest()}


def _data():
    train = pd.read_csv(ROOT / "notebooks/dataset_train.csv")
    test = pd.read_csv(ROOT / "notebooks/dataset_test_X.csv")
    if train.shape != (10962, 237) or test.shape != (3655, 236):
        raise ValueError("Os CSVs já não têm as dimensões do protocolo original.")
    fixed = preparar_dados(train, test)
    X, V, y, v = make_stratified_split(fixed["X_train"], fixed["y_train"],
                                      test_size=0.2, random_state=SEED)
    forbidden = {"AGE_RANGE", "PID", "HHID", "PROXY", "COUNTRY"}
    for name, frame in [("treino", X), ("validação", V), ("teste", fixed["X_test"])]:
        if forbidden.intersection(frame.columns):
            raise ValueError(f"{name}: há colunas proibidas nas entradas.")
    if not X.index.equals(y.index) or not V.index.equals(v.index):
        raise ValueError("Índices das entradas e alvo desalinhados.")
    if X.index.intersection(V.index).size or len(X) != 8769 or len(V) != 2193:
        raise ValueError("Divisão fixa inválida.")
    if not np.isin(y, CLASSES).all() or not np.array_equal(np.sort(y.unique()), CLASSES):
        raise ValueError("AGE_RANGE precisa das 16 classes inteiras de 0 a 15.")
    if list(X.columns) != list(V.columns) or list(X.columns) != list(fixed["X_test"].columns):
        raise ValueError("Esquemas diferentes depois da preparação fixa.")
    return dict(X=X, y=y, V=V, v=v, full=fixed)


def _metrics(y, prediction):
    y = np.asarray(y, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)
    detail = classification_report(y, prediction, labels=CLASSES, output_dict=True, zero_division=0)
    return {"macro_f1": float(f1_score(y, prediction, labels=CLASSES, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y, prediction)),
            "per_class": [{"classe": int(c), "precision": detail[str(c)]["precision"],
                           "recall": detail[str(c)]["recall"], "F1": detail[str(c)]["f1-score"],
                           "support": int(detail[str(c)]["support"])} for c in CLASSES]}


def _smooth_weights(y):
    targets = np.asarray(y, dtype=np.int64)
    counts = np.bincount(targets, minlength=16)
    if np.any(counts == 0):
        raise ValueError("Falta uma classe na parcela de treino.")
    weights = (1.0 / np.sqrt(counts))[targets]
    return weights / weights.mean()


def _grid(k):
    if k == "056":
        changes = [("controlo_041", {}, "none"),
                   ("folhas_min_50", {"min_child_samples": 50}, "none"),
                   ("folhas_min_100", {"min_child_samples": 100}, "none"),
                   ("regularizacao_moderada", {"reg_alpha": 0.5, "reg_lambda": 1.0}, "none"),
                   ("regularizacao_forte", {"reg_alpha": 1.0, "reg_lambda": 5.0}, "none"),
                   ("pesos_suaves", {}, "inverse_sqrt"),
                   ("combinado", {"min_child_samples": 50, "reg_alpha": 0.5, "reg_lambda": 1.0}, "inverse_sqrt")]
        return [dict(label=label, kind="lgb", params={**LGB_CONTROL, **patch}, weighting=w)
                for label, patch, w in changes]
    changes = [("controlo_007", {}), ("min_child_5", {"min_child_weight": 5}),
               ("min_child_10", {"min_child_weight": 10}),
               ("subsample_08", {"subsample": 0.8}),
               ("colsample_08", {"colsample_bytree": 0.8}),
               ("combinado", {"min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8})]
    return [dict(label=label, kind="xgb", params={**XGB_CONTROL, **patch}, weighting="none")
            for label, patch in changes]


def _estimator(candidate):
    if candidate["kind"] == "lgb":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(objective="multiclass", num_class=16, random_state=SEED,
                              n_jobs=1, verbosity=-1, **candidate["params"])
    from xgboost import XGBClassifier
    return XGBClassifier(objective="multi:softprob", num_class=16, tree_method="hist",
                         random_state=SEED, n_jobs=1, verbosity=0, **candidate["params"])


def _probabilities(model, matrix):
    raw = np.asarray(model.predict_proba(matrix), dtype=np.float32)
    result = np.zeros((len(raw), 16), dtype=np.float32)
    result[:, np.asarray(model.classes_, dtype=int)] = raw
    _check_probabilities(result, len(raw))
    return result


def _check_probabilities(probs, rows):
    if probs.shape != (rows, 16) or not np.isfinite(probs).all():
        raise ValueError("Probabilidades com forma ou valores inválidos.")
    if np.any(probs < -1e-6) or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("As probabilidades não formam distribuições válidas.")


def _dense(matrix):
    estimated = int(matrix.shape[0]) * int(matrix.shape[1]) * 4 * 3
    if estimated > 512 * 2**20:
        raise MemoryError(f"Conversão MLP: estimativa das matrizes {estimated/2**20:.0f} MiB > 512 MiB.")
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def _fit_cached(candidate, X, y, Z, signature, y_monitor=None):
    # O alvo de monitorização não decide épocas nem parâmetros: MLP usa 85 épocas fixas.
    payload = dict(context=signature, candidate=candidate, train_indices=_indices(X),
                   evaluation_indices=_indices(Z), train_size=len(X), evaluation_size=len(Z),
                   monitored=y_monitor is not None)
    key = _digest(payload)
    base = OUTPUT / "cache" / key
    manifest, arrays = base.with_suffix(".json"), base.with_suffix(".npz")
    if manifest.exists() and arrays.exists():
        record = json.loads(manifest.read_text())
        if record["signature"] != payload:
            raise ValueError("Assinatura do cache incompatível.")
        with np.load(arrays, allow_pickle=False) as saved:
            train_probs, eval_probs = saved["train"], saved["evaluation"]
        _check_probabilities(train_probs, len(X))
        _check_probabilities(eval_probs, len(Z))
        return train_probs, eval_probs, record["diagnostics"]

    start = time.perf_counter()
    import joblib
    prep = build_preprocessor(X, normalize_numeric=candidate["kind"] == "mlp")
    A = prep.transform(X)
    before = joblib.hash(prep)
    B = prep.transform(Z)
    if before != joblib.hash(prep):
        raise ValueError("O transform da validação/teste alterou o estado aprendido do pré-processador.")
    if candidate["kind"] == "mlp":
        from src.experiments_022_027 import _fit_mlp, _predict_probabilities
        A, B = _dense(A), _dense(B)
        model, history, _ = _fit_mlp(
            A, np.asarray(y, dtype=np.int64), MLP_CONTROL["epochs"], MLP_CONTROL["learning_rate"],
            weighting=MLP_CONTROL["weighting"], dropout=MLP_CONTROL["dropout"], seed=SEED,
            X_monitor=B if y_monitor is not None else None,
            y_monitor=np.asarray(y_monitor, dtype=np.int64) if y_monitor is not None else None)
        train_probs = _predict_probabilities(model, A).astype(np.float32)
        eval_probs = _predict_probabilities(model, B).astype(np.float32)
    else:
        model = _estimator(candidate)
        weights = _smooth_weights(y) if candidate["weighting"] == "inverse_sqrt" else None
        model.fit(A, y, sample_weight=weights)
        train_probs, eval_probs = _probabilities(model, A), _probabilities(model, B)
        history = None
    _check_probabilities(train_probs, len(X))
    _check_probabilities(eval_probs, len(Z))
    diagnostics = dict(seconds=time.perf_counter() - start, encoded_features=int(A.shape[1]), history=history)
    arrays.parent.mkdir(parents=True, exist_ok=True)
    temporary = arrays.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, train=train_probs, evaluation=eval_probs)
    temporary.replace(arrays)
    _atomic_json(manifest, dict(signature=payload, diagnostics=diagnostics))
    del model, prep, A, B
    gc.collect()
    return train_probs, eval_probs, diagnostics


def _folds(data):
    return list(StratifiedKFold(3, shuffle=True, random_state=SEED).split(data["X"], data["y"]))


def _cv(candidate, data, signature, k):
    values, train_values = [], []
    for fold, (a, b) in enumerate(_folds(data), 1):
        print(f"{k}: {candidate['label']}, fold {fold}/3", flush=True)
        tr, va, _ = _fit_cached(candidate, data["X"].iloc[a], data["y"].iloc[a],
                               data["X"].iloc[b], signature)
        train_values.append(_metrics(data["y"].iloc[a], tr.argmax(axis=1))["macro_f1"])
        values.append(_metrics(data["y"].iloc[b], va.argmax(axis=1))["macro_f1"])
    return dict(candidate=candidate, folds=values, mean=float(np.mean(values)),
                std=float(np.std(values, ddof=1)), train_mean=float(np.mean(train_values)))


def _export(prediction, k):
    _, reference = _reference()
    prediction = np.asarray(prediction)
    if prediction.shape != (3655,) or not np.isfinite(prediction).all():
        raise ValueError("Previsões com forma ou valores inválidos.")
    if not np.array_equal(prediction, prediction.astype(np.int64)) or not np.isin(prediction, CLASSES).all():
        raise ValueError("As classes têm de ser inteiras entre 0 e 15.")
    directory = ROOT / "submissions"
    directory.mkdir(exist_ok=True)
    differences = {}
    for p in sorted(directory.glob("submission_*.csv")):
        old = pd.read_csv(p)
        if list(old) != ["RowId", "AGE_RANGE"] or len(old) != 3655 or not old.RowId.is_unique:
            continue
        aligned = old.set_index("RowId").reindex(reference.RowId)
        if aligned.AGE_RANGE.notna().all():
            differences[p.name] = int(np.sum(aligned.AGE_RANGE.to_numpy() != prediction))
    duplicates = [name for name, n in differences.items() if n == 0]
    if duplicates:
        return dict(path=None, duplicate_of=duplicates, differences=differences)
    labels = {"056": "lightgbm_afinacao", "057": "xgboost_afinacao", "058": "ensemble_xgboost_mlp_refinado"}
    path = directory / f"submission_{k}_{labels[k]}.csv"
    if path.exists():
        raise FileExistsError(f"Não substituir silenciosamente {path}")
    out = reference.copy()
    out["AGE_RANGE"] = prediction.astype(np.int64)
    out.to_csv(path, index=False)
    reopened = pd.read_csv(path)
    if not reopened.equals(out) or list(reopened) != ["RowId", "AGE_RANGE"]:
        raise ValueError("A releitura do CSV não coincide com os dados exportados.")
    return dict(path=str(path.relative_to(ROOT)), duplicate_of=[], differences=differences,
                sha256=_hash(path), rows=3655)


def _paired(selected, control):
    delta = np.asarray(selected["folds"]) - np.asarray(control["folds"])
    return dict(differences=delta.tolist(), mean=float(delta.mean()), std=float(delta.std(ddof=1)))


def _plots(k, rows, fixed, actual, prediction):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    REPORT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.errorbar(np.arange(len(rows)), [r["mean"] for r in rows],
                yerr=[r["std"] for r in rows], marker="o", capsize=4)
    ax.set(xticks=np.arange(len(rows)), xticklabels=[r["candidate"]["label"] for r in rows],
           ylabel="Macro F1 CV: média ± DP", title=f"{k} — seleção dentro do treino")
    ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    fig.savefig(REPORT / f"experiencia_{k}_cv.png", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(confusion_matrix(actual, prediction, labels=CLASSES), cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(xticks=CLASSES, yticks=CLASSES, xlabel="Previsto", ylabel="Real",
           title=f"{k} — validação de desenvolvimento")
    fig.tight_layout()
    fig.savefig(REPORT / f"experiencia_{k}_matriz_confusao.png", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(fixed))
    ax.bar(x - .18, [r["train"]["macro_f1"] for r in fixed.values()], .36, label="Treino")
    ax.bar(x + .18, [r["validation"]["macro_f1"] for r in fixed.values()], .36, label="Desenvolvimento")
    ax.set(xticks=x, xticklabels=list(fixed), ylabel="Macro F1", title=f"{k} — diferença treino/validação")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT / f"experiencia_{k}_treino_validacao.png", dpi=130)
    plt.close(fig)


def _trees(k, data, signature):
    rows = [_cv(c, data, signature, k) for c in _grid(k)]
    selected = max(rows, key=lambda r: r["mean"])
    fixed = {}
    predictions = {}
    for label, row in [("controlo", rows[0]), ("selecionado", selected)]:
        tr, va, _ = _fit_cached(row["candidate"], data["X"], data["y"], data["V"], signature)
        fixed[label] = dict(train=_metrics(data["y"], tr.argmax(axis=1)),
                            validation=_metrics(data["v"], va.argmax(axis=1)))
        predictions[label] = va.argmax(axis=1)
    submission = dict(path=None, reason="O controlo venceu a seleção interna.")
    if selected["candidate"] != rows[0]["candidate"]:
        full = data["full"]
        _, test, _ = _fit_cached(selected["candidate"], full["X_train"], full["y_train"], full["X_test"], signature)
        submission = _export(test.argmax(axis=1), k)
    _plots(k, rows, fixed, data["v"], predictions["selecionado"])
    return dict(candidates=rows, selected=selected, paired_cv=_paired(selected, rows[0]),
                fixed=fixed, submission=submission,
                protocol="3 folds dentro de 8769 linhas; 2193 apenas para desenvolvimento. Sem early stopping: número de árvores histórico fixo.")


def _ensemble(k, data, signature):
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    if "057" not in results or results["057"]["context"] != result_context("057"):
        raise ValueError("Executar 057 com estes dados/código/versões antes do 058.")
    original = _grid("057")[0]
    tuned = results["057"]["selected"]["candidate"]
    mlp = dict(label="mlp_020_fixo", kind="mlp", params=MLP_CONTROL, weighting="inverse_sqrt")
    options = [dict(label="controlo_025_75_25", xgb=original, weight=.75)]
    options += [dict(label=f"xgb057_{w:g}_mlp_{1-w:g}", xgb=tuned, weight=w)
                for w in [0., .25, .5, .6, .75, .9, 1.]]
    scores = [[] for _ in options]
    histories = []
    split_hashes = []
    for fold, (a, b) in enumerate(_folds(data), 1):
        print(f"058: componentes do fold {fold}/3", flush=True)
        X, y, V, v = data["X"].iloc[a], data["y"].iloc[a], data["X"].iloc[b], data["y"].iloc[b]
        _, old_probs, _ = _fit_cached(original, X, y, V, signature)
        _, new_probs, _ = _fit_cached(tuned, X, y, V, signature)
        _, mlp_probs, diagnostics = _fit_cached(mlp, X, y, V, signature, y_monitor=v)
        histories.append(diagnostics["history"])
        split_hashes.append(dict(train=_indices(X), validation=_indices(V)))
        for i, option in enumerate(options):
            probs = old_probs if i == 0 else new_probs
            mix = option["weight"] * probs + (1 - option["weight"]) * mlp_probs
            scores[i].append(_metrics(v, mix.argmax(axis=1))["macro_f1"])
    rows = [dict(candidate=o, folds=s, mean=float(np.mean(s)), std=float(np.std(s, ddof=1)))
            for o, s in zip(options, scores)]
    selected = max(rows, key=lambda r: r["mean"])
    fixed, predictions = {}, {}
    mt, mv, _ = _fit_cached(mlp, data["X"], data["y"], data["V"], signature, y_monitor=data["v"])
    for label, row in [("controlo_025", rows[0]), ("selecionado", selected)]:
        option = row["candidate"]
        xt, xv, _ = _fit_cached(option["xgb"], data["X"], data["y"], data["V"], signature)
        w = option["weight"]
        tp, vp = (w * xt + (1-w) * mt).argmax(axis=1), (w * xv + (1-w) * mv).argmax(axis=1)
        fixed[label] = dict(train=_metrics(data["y"], tp), validation=_metrics(data["v"], vp))
        predictions[label] = vp
    submission = dict(path=None, reason="O controlo 025 venceu a seleção interna.")
    if selected is not rows[0]:
        full, option = data["full"], selected["candidate"]
        _, xp, _ = _fit_cached(option["xgb"], full["X_train"], full["y_train"], full["X_test"], signature)
        if option["weight"] == 1:
            mixed = xp
        else:
            _, mp, _ = _fit_cached(mlp, full["X_train"], full["y_train"], full["X_test"], signature)
            mixed = option["weight"] * xp + (1-option["weight"]) * mp
        submission = _export(mixed.argmax(axis=1), k)
    _plots(k, rows, fixed, data["v"], predictions["selecionado"])
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, h in enumerate(histories, 1):
        ax.plot(h["epoch"], h["train_loss"], "--", label=f"Treino fold {i}")
        ax.plot(h["epoch"], h["validation_loss"], label=f"Validação fold {i}")
    ax.set(xlabel="Época", ylabel="Perda", title="058 — MLP: 85 épocas históricas fixas")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(REPORT / "experiencia_058_mlp_curvas.png", dpi=130)
    plt.close(fig)
    return dict(candidates=rows, selected=selected, paired_cv=_paired(selected, rows[0]),
                fixed=fixed, submission=submission, cv_split_hashes=split_hashes,
                mlp_configuration=dict(hidden_layers=[64, 32], outputs=16, optimizer="Adam", **MLP_CONTROL),
                protocol="CV de desenvolvimento para escolher pesos. A configuração XGBoost já foi selecionada em 057 usando o mesmo treino; estas médias não são uma estimativa independente pós-seleção. A validação fixa não escolhe pesos nem épocas.")


def run_experiment(k):
    k = str(k).zfill(3)
    if k not in ["056", "057", "058"]:
        raise ValueError("Usar 056, 057 ou 058.")
    required = ["lightgbm"] if k == "056" else ["xgboost"]
    if k == "058":
        required.append("torch")
    missing = [p for p in required if importlib.util.find_spec(p) is None]
    if missing:
        raise ModuleNotFoundError("Dependências ausentes: " + ", ".join(missing))
    signature = context()
    expected = result_context(k)
    old = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    if k in old:
        if old[k]["context"] != expected:
            raise ValueError(f"{k}: resultado de outros dados/código/versões; não o substituir silenciosamente.")
        print(k, "resultado completo reutilizado após verificar contexto.", flush=True)
        return old[k]
    data = _data()
    start = time.perf_counter()
    with threadpool_limits(limits=1):
        result = _ensemble(k, data, signature) if k == "058" else _trees(k, data, signature)
    result.update(experiment=k, context=expected, module_sha256=_hash(Path(__file__)), seconds=time.perf_counter()-start, kaggle_score=None,
                  selection="Macro F1 média dos mesmos três folds, apenas no treino; em empate, primeiro candidato (controlo).",
                  cv_split_hashes=[dict(train=_indices(data["X"].iloc[a]), validation=_indices(data["X"].iloc[b]))
                                   for a,b in _folds(data)])
    old[k] = result
    _atomic_json(RESULTS, old)
    write_report()
    return result


def write_report():
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    lines = ["# Experiências 056–058", "", "Validação fixa de desenvolvimento; seleção apenas dentro do treino. Kaggle por preencher.", "",
             "| ID | Macro F1 CV | Controlo fixo | Selecionado fixo | CSV |", "|---|---:|---:|---:|---|"]
    for k in ["056", "057", "058"]:
        if k not in results:
            lines.append(f"| {k} | — | — | — | Por executar |")
            continue
        r = results[k]
        control = next(iter(r["fixed"].values()))["validation"]["macro_f1"]
        score = r["fixed"]["selecionado"]["validation"]["macro_f1"]
        lines.append(f"| {k} | {r['selected']['mean']:.4f} | {control:.4f} | {score:.4f} | {r['submission'].get('path') or 'Não gerado; consultar motivo'} |")
    lines += ["", "## Protocolo e limites", "", "- 056 mantém 126 árvores, 31 folhas e taxa 0,03 do 041. 057 mantém 200 árvores, profundidade 5 e taxa 0,1 do 007.",
              "- 058 reutiliza MLP 64→32, Adam 0,0001, dropout 0,1, pesos 1/√frequência e 85 épocas do 020/025. A monitorização não escolhe épocas.",
              "- Todas as imputações, categorias, normalizações e pesos de classes são ajustados/calculados apenas na parcela de treino correspondente.",
              "- O 058 reutiliza a configuração escolhida em 057. As médias CV são desenvolvimento, com seleção sequencial, não um teste independente.",
              "- A validação fixa já foi observada em experiências anteriores; o relatório não a chama avaliação final independente.",
              "- Um CSV novo e distinto é exploratório; não garante melhoria no Kaggle. Duplicados só são comparados com CSVs existentes em submissions/.",
              "- Nenhuma submissão, commit ou push é realizado automaticamente."]
    path = REPORT / "experiments_056_058.md"
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")
    return path
