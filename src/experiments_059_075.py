"""DAA 059–075: SVM, SVR, misturas e análise. Importar não inicia treinos."""
from __future__ import annotations

import copy
import gc
import hashlib
import inspect
import json
import time
import unicodedata
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import confusion_matrix, log_loss, mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from threadpoolctl import threadpool_limits

from src import experiments_056_058 as previous
from src.preparar_dados import DataPreprocessor

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports"
OUTPUT = ROOT / "outputs/experiments_059_075"
RESULTS = REPORT / ".experiments_059_075_results.json"
CLASSES = np.arange(16)
SEED = 42
NAMES = {
    "059": "svc_linear", "060": "svc_rbf_baseline", "061": "svc_rbf_afinacao",
    "062": "svc_polinomial", "063": "svc_sigmoidal", "064": "svc_pesos_classes",
    "065": "svc_svd", "066": "svc_ovo_ovr", "067": "linear_svr",
    "068": "svr_rbf", "069": "svr_limiares_oof", "070": "svm_calibracao",
    "071": "ensemble_058_svm", "072": "ensemble_058_lightgbm",
    "073": "estabilidade_cv", "074": "erros_perfis_oof", "075": "caracterizacao_escaloes",
}
DESCRIPTIONS = {
    "059": "SVC linear versus configuração LinearSVC do 037 (C=0,01, balanced).",
    "060": "SVC RBF C=1/gamma=scale versus a referência linear de 059.",
    "061": "Pesquisa curta de C/gamma em RBF; controlo C=1/gamma=scale.",
    "062": "Kernel polinomial graus 2/3 versus RBF escolhido em 061.",
    "063": "Kernel sigmoidal exploratório versus RBF escolhido em 061.",
    "064": "Pesos None/balanced na melhor configuração SVM anterior, mantida fixa.",
    "065": "Entradas completas versus SVD32/SVD64, mantendo o SVM escolhido em 064.",
    "066": "Treino OvO nativo de SVC versus OneVsRestClassifier(SVC), com o mesmo RBF. Não altera apenas decision_function_shape.",
    "067": "LinearSVR; estima o código ordinal, com arredondamento .5 para cima e clipping a 0–15.",
    "068": "SVR RBF: pequena pesquisa C/gamma/epsilon versus referência linear de 067.",
    "069": "Arredondamento versus 15 limiares monótonos aprendidos com previsões OOF apenas do treino de cada fold.",
    "070": "Calibração sigmoid do melhor SVM; a pipeline inteira fica dentro da CV de calibração.",
    "071": "Ensemble 058 com SVM calibrado; pesos 0/0,1/0,25/0,5/1 do SVM, incluindo os modelos individuais.",
    "072": "Ensemble 058 com LightGBM escolhido em 056; mesmos pesos, incluindo os extremos.",
    "073": "Estabilidade de 025, 058 e melhor SVM: três partições CV (42/73/123), mesmos folds por modelo; sementes dos modelos mantidas em 42. Sem CSV Kaggle.",
    "074": "Erros OOF e perfis com desvio de pelo menos três códigos de escalão. Hipóteses de perfil, não diagnósticos clínicos. Sem CSV Kaggle.",
    "075": "Caracterização socioeconómica/saúde dos escalões só nas 8.769 linhas de treino. Respostas especiais distintas. Sem CSV Kaggle.",
}
DEPENDENCIES = {
    "059": [], "060": ["059"], "061": ["060"], "062": ["061"], "063": ["061"],
    "064": ["059", "061", "062", "063"], "065": ["064"], "066": ["061"],
    "067": [], "068": ["067"], "069": ["068"],
    "070": ["059", "061", "062", "063", "064", "065", "066"],
    "071": ["070"], "072": [], "073": ["070"], "074": ["073"], "075": [],
}
_BASE = None


def _json(value):
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_json(v) for v in value]
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(_json(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def _all():
    return json.loads(RESULTS.read_text()) if RESULTS.exists() else {}


def _hash_code(functions):
    return hashlib.sha256("\n".join(inspect.getsource(f) for f in functions).encode()).hexdigest()


def _base_context():
    global _BASE
    if _BASE is None:
        _BASE = previous.context()
    return _BASE


def _canonical(spec):
    return json.dumps(spec, sort_keys=True, ensure_ascii=False)


def _svc(**changes):
    return dict(family="svc", kernel="rbf", C=1.0, gamma="scale", degree=3,
                coef0=0.0, class_weight=None, strategy="ovo", svd=0, **{}) | changes


def _pipeline(estimator, spec):
    steps = [("prep", DataPreprocessor(normalize_numeric=True))]
    if spec.get("svd", 0):
        steps += [("svd", TruncatedSVD(n_components=spec["svd"], random_state=SEED)),
                  ("scale_components", StandardScaler())]
    return Pipeline(steps + [("model", estimator)])


def _classifier(spec):
    if spec["family"] == "linear_svc":
        return _pipeline(LinearSVC(C=spec["C"], class_weight=spec.get("class_weight"),
                                   random_state=SEED, dual="auto", max_iter=50000), spec)
    # Native SVC always trains OvO. 'ovr' below controls score shape only.
    # OvR TRAINING is implemented explicitly by the wrapper.
    model = SVC(C=spec["C"], kernel=spec["kernel"], gamma=spec.get("gamma", "scale"),
                degree=spec.get("degree", 3), coef0=spec.get("coef0", 0),
                class_weight=spec.get("class_weight"), cache_size=256,
                decision_function_shape="ovr", random_state=SEED, max_iter=-1)
    if spec.get("strategy", "ovo") == "ovr":
        model = OneVsRestClassifier(model, n_jobs=1)
    return _pipeline(model, spec)


def _regressor(spec):
    if spec["family"] == "linear_svr":
        model = LinearSVR(C=spec["C"], epsilon=spec["epsilon"], random_state=SEED,
                          max_iter=50000, dual="auto", tol=1e-3)
    else:
        model = SVR(C=spec["C"], epsilon=spec["epsilon"], kernel=spec["kernel"],
                    gamma=spec.get("gamma", "scale"), cache_size=256, max_iter=-1)
    return _pipeline(model, spec)


def _calibrated(spec):
    # Entire RAW-DATA pipeline is cloned/refitted inside each calibration fold.
    return CalibratedClassifierCV(estimator=_classifier(spec["base"]), method="sigmoid",
                                  cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
                                  n_jobs=1, ensemble=True)


def _signature(spec):
    family = spec["family"]
    shared = [_pipeline, _raw_fit, _support, _json, _save]
    if family in ["svc", "linear_svc"]:
        specific = [_classifier]
    elif family in ["svr", "linear_svr"]:
        specific = [_regressor]
        if spec.get("conversion"):
            specific += [_round, _prediction]
        if spec.get("conversion") == "learned":
            specific += [_cutpoints, _optimise_cuts]
    elif family == "calibrated":
        specific = [_classifier, _calibrated]
    elif family in ["legacy_058", "legacy_025", "legacy_lgb"]:
        specific = [_legacy]
    elif family == "mixture":
        specific = [_prediction]
    else:
        raise ValueError(f"Família desconhecida: {family}")
    result = dict(base=_base_context(), spec=spec, code=_hash_code(shared + specific))
    if family == "calibrated":
        result["base_model"] = _signature(spec["base"])
    if family == "mixture":
        result["components"] = [_signature(s) for s in spec["components"]]
    return result


def _support(model):
    if isinstance(model, Pipeline):
        return _support(model.steps[-1][1])
    if hasattr(model, "n_support_"):
        return {"total": int(np.sum(model.n_support_)), "per_class": np.asarray(model.n_support_).tolist()}
    if hasattr(model, "calibrated_classifiers_"):
        return {"calibration_models": [_support(c.estimator) for c in model.calibrated_classifiers_]}
    if hasattr(model, "estimators_"):
        return {"binary_models": [_support(m) for m in model.estimators_]}
    return None


def _raw_fit(spec, X, y, Z):
    raw_spec = {k: v for k, v in spec.items() if k != "conversion"}
    payload = dict(signature=_signature(raw_spec), train=previous._indices(X),
                   evaluation=previous._indices(Z), n_train=len(X), n_evaluation=len(Z))
    key = previous._digest(payload)
    path = OUTPUT / "cache" / key
    meta, values = path.with_suffix(".json"), path.with_suffix(".npz")
    if meta.exists() and values.exists():
        record = json.loads(meta.read_text())
        if record["context"] != _json(payload):
            raise ValueError("Cache com assinatura incompatível.")
        with np.load(values, allow_pickle=False) as f:
            arrays = {name: f[name] for name in f.files}
        return arrays, record["diagnostics"]

    model = (_calibrated(spec) if spec["family"] == "calibrated" else
             _regressor(spec) if spec["family"] in ["svr", "linear_svr"] else _classifier(spec))
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        model.fit(X, np.array(y, dtype=np.int64, copy=True))
    messages = [f"{w.category.__name__}: {w.message}" for w in captured]
    for message in messages:
        print(message, flush=True)
    if any(issubclass(w.category, ConvergenceWarning) for w in captured):
        raise RuntimeError("Solver não convergiu. Não pontuar este ajuste; rever o orçamento antes de retomar.")
    # Transform/predict must not modify fitted preprocessing/statistics.
    before = joblib.hash(model)
    arrays = {"train": np.asarray(model.predict(X)), "evaluation": np.asarray(model.predict(Z))}
    if spec["family"] == "calibrated":
        classes = np.asarray(model.classes_, dtype=int)
        for key_, frame in [("p_train", X), ("p_evaluation", Z)]:
            p = np.zeros((len(frame), 16), dtype=np.float32)
            p[:, classes] = model.predict_proba(frame)
            previous._check_probabilities(p, len(frame))
            arrays[key_] = p
    if joblib.hash(model) != before:
        raise RuntimeError("O predict/transform alterou o estado aprendido do modelo.")
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("Previsões não finitas.")
    diagnostic = dict(seconds=time.perf_counter()-started, warnings=messages, support_vectors=_support(model))
    values.parent.mkdir(parents=True, exist_ok=True)
    temp = values.with_suffix(".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temp.replace(values)
    _save(meta, dict(context=payload, diagnostics=diagnostic))
    del model
    gc.collect()
    return arrays, diagnostic


def _round(values):
    return np.clip(np.floor(np.asarray(values) + 0.5), 0, 15).astype(np.int64)


def _optimise_cuts(prediction, y):
    cuts = np.arange(0.5, 15.0, 1.0)
    score = previous._metrics(y, np.searchsorted(cuts, prediction, side="right"))["macro_f1"]
    for _ in range(2):
        changed = False
        for j in range(15):
            best, value = score, cuts[j]
            for candidate in np.unique(np.r_[cuts[j], np.linspace(j, j+1.0, 11)]):
                lower = cuts[j-1]+1e-5 if j else -0.5
                upper = cuts[j+1]-1e-5 if j < 14 else 15.5
                if not lower < candidate < upper:
                    continue
                trial = cuts.copy()
                trial[j] = candidate
                s = previous._metrics(y, np.searchsorted(trial, prediction, side="right"))["macro_f1"]
                if s > best + 1e-12:
                    best, value = s, candidate
            if value != cuts[j]:
                cuts[j], score, changed = value, best, True
        if not changed:
            break
    if not np.all(np.diff(cuts) > 0):
        raise ValueError("Os limiares não são estritamente monótonos.")
    return cuts, score


def _cutpoints(spec, X, y):
    payload = dict(signature=_signature(spec), train=previous._indices(X), procedure="inner_3fold_oof_thresholds")
    file = OUTPUT / "thresholds" / (previous._digest(payload) + ".json")
    if file.exists():
        record = json.loads(file.read_text())
        if record["context"] != _json(payload):
            raise ValueError("Cache de limiares incompatível.")
        return np.asarray(record["cuts"]), record
    prediction = np.empty(len(X), dtype=float)
    raw_spec = {**spec, "conversion": "round"}
    partitions = []
    for fold, (a, b) in enumerate(StratifiedKFold(3, shuffle=True, random_state=SEED).split(X, y), 1):
        print(f"069: calibração OOF interna, fold {fold}/3", flush=True)
        raw, _ = _raw_fit(raw_spec, X.iloc[a], y.iloc[a], X.iloc[b])
        prediction[b] = raw["evaluation"]
        partitions.append(dict(train=previous._indices(X.iloc[a]), validation=previous._indices(X.iloc[b])))
    cuts, diagnostic_score = _optimise_cuts(prediction, y)
    record = dict(context=payload, cuts=cuts.tolist(), inner_calibration_macro_f1=diagnostic_score,
                  partitions=partitions, note="Métrica do ajuste dos limiares, não avaliação externa.")
    _save(file, record)
    return cuts, record


def _legacy(spec, X, y, Z, y_monitor=None):
    # Reuse 056–058 caches when context, hyperparameters and row order match.
    # Explicit copies protect tensors from pandas 3 read-only NumPy views.
    signature = previous.context()
    yy = np.array(y, dtype=np.int64, copy=True)
    monitor = np.array(y_monitor, dtype=np.int64, copy=True) if y_monitor is not None else None
    if spec["family"] == "legacy_lgb":
        tr, va, diagnostic = previous._fit_cached(spec["candidate"], X, yy, Z, signature)
    else:
        weight = spec["weight"]
        if weight != 0:
            xt, xv, _ = previous._fit_cached(spec["xgb"], X, yy, Z, signature)
        if weight != 1:
            mlp = dict(label="mlp_020_fixo", kind="mlp", params=previous.MLP_CONTROL, weighting="inverse_sqrt")
            mt, mv, _ = previous._fit_cached(mlp, X, yy, Z, signature, y_monitor=monitor)
        tr = xt if weight == 1 else mt if weight == 0 else weight*xt+(1-weight)*mt
        va = xv if weight == 1 else mv if weight == 0 else weight*xv+(1-weight)*mv
        diagnostic = {"note": "MLP histórico de 85 épocas fixas; monitorização não seleciona duração."}
    previous._check_probabilities(tr, len(X))
    previous._check_probabilities(va, len(Z))
    return {"train": tr.argmax(1), "evaluation": va.argmax(1), "p_train": tr, "p_evaluation": va}, diagnostic


def _prediction(spec, X, y, Z, y_monitor=None):
    family = spec["family"]
    if family.startswith("legacy_"):
        return _legacy(spec, X, y, Z, y_monitor)
    if family == "mixture":
        w = spec["second_weight"]
        if w == 0:
            return _prediction(spec["components"][0], X, y, Z, y_monitor)
        if w == 1:
            return _prediction(spec["components"][1], X, y, Z, y_monitor)
        first, _ = _prediction(spec["components"][0], X, y, Z, y_monitor)
        second, _ = _prediction(spec["components"][1], X, y, Z, y_monitor)
        result = {}
        for key in ["p_train", "p_evaluation"]:
            p = (1-w)*first[key] + w*second[key]
            previous._check_probabilities(p, len(X) if key == "p_train" else len(Z))
            result[key] = p
        result.update(train=result["p_train"].argmax(1), evaluation=result["p_evaluation"].argmax(1))
        return result, {"second_weight": w}
    raw, diagnostic = _raw_fit(spec, X, y, Z)
    if family in ["svr", "linear_svr"]:
        result = dict(raw)
        if spec.get("conversion", "round") == "learned":
            cuts, info = _cutpoints(spec, X, y)
            for key in ["train", "evaluation"]:
                result[key] = np.searchsorted(cuts, raw[key], side="right")
            diagnostic = {**diagnostic, "thresholds": info}
        else:
            result.update(train=_round(raw["train"]), evaluation=_round(raw["evaluation"]))
        diagnostic = {**diagnostic, "train_raw_mae_in_codes": float(mean_absolute_error(y, raw["train"]))}
        return result, diagnostic
    return raw, diagnostic


def _legacy_recipe(k):
    if k == "025":
        return dict(family="legacy_025", xgb=previous._grid("057")[0], weight=.75)
    file = ROOT / "reports/.experiments_056_058_results.json"
    if not file.exists():
        raise FileNotFoundError("Falta reports/.experiments_056_058_results.json do teu PC. Não reconstruir parâmetros a partir dos CSVs.")
    records = json.loads(file.read_text())
    if k not in records:
        raise ValueError(f"Falta o resultado {k} no JSON 056–058.")
    record = records[k]
    # Full old code/version contexts may differ after harmless maintenance.
    # The source-data/preparation hashes must match; new fitting uses current versions.
    old_files, current_files = record["context"]["files"], previous.context()["files"]
    for name in ["notebooks/dataset_train.csv", "notebooks/dataset_test_X.csv", "src/preparar_dados.py"]:
        if old_files.get(name) != current_files.get(name):
            raise ValueError(f"{k}: dados/preparação mudaram em {name}. Rever antes de reutilizar a configuração.")
    candidate = copy.deepcopy(record["selected"]["candidate"])
    if k == "056":
        return dict(family="legacy_lgb", candidate=candidate)
    if k != "058" or "xgb" not in candidate or "weight" not in candidate:
        raise ValueError("Configuração do ensemble 058 não reconhecida.")
    return dict(family="legacy_058", xgb=candidate["xgb"], weight=candidate["weight"])


def _best(records, ids):
    rows = [records[k]["selected"] for k in ids]
    return copy.deepcopy(max(rows, key=lambda r: r["mean"])["spec"])


def _recipe(k, records):
    linear = dict(family="linear_svc", C=.01, class_weight="balanced", svd=0)
    if k == "059":
        return [("controlo_037", linear), ("svc_linear", _svc(kernel="linear", C=.01, class_weight="balanced"))]
    if k == "060":
        return [("controlo_linear", _best(records, ["059"])), ("rbf_baseline", _svc())]
    if k == "061":
        return [("controlo_rbf", _svc())] + [(f"C_{c}_gamma_{g}", _svc(C=c, gamma=g))
                for c, g in [(.1, "scale"), (10., "scale"), (1., .001), (1., .01), (10., .001), (10., .01)]]
    if k in ["062", "063"]:
        ref = _best(records, ["061"])
        candidates = [("controlo_rbf", ref)]
        candidates += [(f"polinomial_{d}", _svc(kernel="poly", degree=d)) for d in [2, 3]] if k == "062" else [
            ("sigmoid_scale", _svc(kernel="sigmoid")), ("sigmoid_0001", _svc(kernel="sigmoid", gamma=.001))]
        return candidates
    if k == "064":
        ref = _best(records, ["059", "061", "062", "063"])
        other = {**ref, "class_weight": None if ref.get("class_weight") == "balanced" else "balanced"}
        return [("controlo", ref), ("pesos_alternativos", other)]
    if k == "065":
        ref = _best(records, ["064"])
        return [("controlo_sem_svd", ref)] + [(f"svd_{n}", {**ref, "svd": n}) for n in [32, 64]]
    if k == "066":
        ref = {**_best(records, ["061"]), "strategy": "ovo"}
        return [("controlo_ovo", ref), ("treino_ovr", {**ref, "strategy": "ovr"})]
    if k == "067":
        return [("linear_svr", dict(family="linear_svr", C=1., epsilon=.1, svd=0, conversion="round"))]
    if k == "068":
        return [("controlo_linear", _best(records, ["067"]))] + [
            (f"rbf_C{c}_epsilon{eps}_gamma{g}", dict(family="svr", kernel="rbf", C=c, epsilon=eps,
                                                   gamma=g, svd=0, conversion="round"))
            for c, eps, g in [(1., .1, "scale"), (10., .1, "scale"), (10., .5, "scale"), (10., .1, .001)]]
    if k == "069":
        ref = {**_best(records, ["068"]), "conversion": "round"}
        return [("controlo_arredondamento", ref), ("limiares_oof", {**ref, "conversion": "learned"})]
    if k == "070":
        ref = _best(records, ["059", "061", "062", "063", "064", "065", "066"])
        return [("controlo_sem_calibracao", ref), ("sigmoid_cv", dict(family="calibrated", base=ref))]
    if k in ["071", "072"]:
        ref = _legacy_recipe("058")
        extra = (copy.deepcopy(records["070"]["candidates"][1]["spec"]) if k == "071" else _legacy_recipe("056"))
        return [("controlo_058", ref)] + [(f"peso_extra_{w:g}", dict(family="mixture", components=[ref, extra], second_weight=w))
                                          for w in [.1, .25, .5, 1.]]
    raise ValueError(k)


def _context(k, recipe):
    procedural = [_stability] if k == "073" else [_errors] if k == "074" else [_profiles, _normal] if k == "075" else [_classification, _cv, _export]
    return dict(base=_base_context(), experiment=k, recipe=recipe,
                models=[_signature(s) for _, s in recipe] if recipe else [],
                procedure=_hash_code(procedural), split_seed=SEED)


def _cv(spec, data, k, fold_seed=SEED):
    rows, oof = [], np.empty(len(data["X"]), dtype=np.int64)
    probabilities = np.full((len(data["X"]), 16), np.nan, dtype=np.float32)
    partitions = []
    for fold, (a, b) in enumerate(StratifiedKFold(3, shuffle=True, random_state=fold_seed).split(data["X"], data["y"]), 1):
        print(f"{k}: fold {fold}/3, partição {fold_seed}", flush=True)
        result, diagnostic = _prediction(spec, data["X"].iloc[a], data["y"].iloc[a], data["X"].iloc[b], data["y"].iloc[b])
        oof[b] = result["evaluation"]
        if "p_evaluation" in result:
            probabilities[b] = result["p_evaluation"]
        metric = previous._metrics(data["y"].iloc[b], result["evaluation"])
        rows.append(dict(validation=metric, train=previous._metrics(data["y"].iloc[a], result["train"]), diagnostics=diagnostic))
        partitions.append(dict(train=previous._indices(data["X"].iloc[a]), validation=previous._indices(data["X"].iloc[b])))
    scores = [r["validation"]["macro_f1"] for r in rows]
    return dict(folds=scores, mean=float(np.mean(scores)), std=float(np.std(scores, ddof=1)),
                details=rows, partitions=partitions), oof, probabilities


def _export(k, prediction):
    _, ref = previous._reference()
    p = np.asarray(prediction)
    if p.shape != (3655,) or not np.isfinite(p).all() or not np.array_equal(p, p.astype(int)) or not np.isin(p, CLASSES).all():
        raise ValueError("Previsões de submissão inválidas; precisam de 3655 classes inteiras 0–15.")
    directory = ROOT / "submissions"
    directory.mkdir(exist_ok=True)
    differences = {}
    for file in sorted(directory.glob("submission_*.csv")):
        old = pd.read_csv(file)
        if list(old) != ["RowId", "AGE_RANGE"] or len(old) != 3655 or not old.RowId.is_unique:
            continue
        old = old.set_index("RowId").reindex(ref.RowId)
        if old.AGE_RANGE.notna().all():
            differences[file.name] = int(np.sum(old.AGE_RANGE.to_numpy() != p))
    duplicates = [name for name, n in differences.items() if n == 0]
    filename = f"submission_{k}_{NAMES[k]}.csv"
    if duplicates:
        return dict(path=str(Path("submissions")/filename) if filename in duplicates else None,
                    duplicate_of=duplicates, differences=differences, reason="Previsões já existentes; não foi escrito novo CSV.")
    path = directory / filename
    if path.exists():
        raise FileExistsError(f"Não substituir silenciosamente {path}.")
    out = ref.copy()
    out["AGE_RANGE"] = p.astype(np.int64)
    out.to_csv(path, index=False)
    if not pd.read_csv(path).equals(out):
        raise ValueError("O CSV relido não coincide com os dados exportados.")
    return dict(path=str(path.relative_to(ROOT)), duplicate_of=[], differences=differences,
                rows=3655, sha256=previous._hash(path))


def _figure(k, candidates, y, prediction):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    REPORT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.errorbar(range(len(candidates)), [r["mean"] for r in candidates],
                yerr=[r["std"] for r in candidates], marker="o", capsize=4)
    ax.set(xticks=range(len(candidates)), xticklabels=[r["label"] for r in candidates],
           ylabel="Macro F1 CV: média ± DP", title=f"{k} — seleção no treino")
    ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    fig.savefig(REPORT/f"experiencia_{k}_cv.png", dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    matrix = confusion_matrix(y, prediction, labels=CLASSES)
    for axis, values, label in [(axes[0], matrix, "Contagens"),
                                (axes[1], matrix/np.maximum(matrix.sum(1, keepdims=True), 1), "Proporção por classe real")]:
        image = axis.imshow(values, cmap="Blues")
        fig.colorbar(image, ax=axis)
        axis.set(xticks=CLASSES, yticks=CLASSES, xlabel="Previsto", ylabel="Real", title=label)
    fig.suptitle(f"{k} — validação de desenvolvimento")
    fig.tight_layout()
    fig.savefig(REPORT/f"experiencia_{k}_confusao.png", dpi=130)
    plt.close(fig)


def _classification(k, recipe, data):
    candidates = []
    for label, spec in recipe:
        print(k, label, flush=True)
        row, _, _ = _cv(spec, data, k)
        candidates.append(dict(label=label, spec=spec, **row))
    selected = max(candidates, key=lambda r: r["mean"])
    fixed = {}
    chosen_prediction = None
    for label, candidate in [("controlo", candidates[0]), ("selecionado", selected)]:
        result, diagnostic = _prediction(candidate["spec"], data["X"], data["y"], data["V"], data["v"])
        fixed[label] = dict(train=previous._metrics(data["y"], result["train"]),
                            validation=previous._metrics(data["v"], result["evaluation"]), diagnostics=diagnostic,
                            mae_in_class_codes=float(mean_absolute_error(data["v"], result["evaluation"])))
        if "p_evaluation" in result:
            fixed[label]["log_loss"] = float(log_loss(data["v"], result["p_evaluation"], labels=CLASSES))
        if label == "selecionado":
            chosen_prediction = result["evaluation"]
    # 067 is a new baseline; all other experiments have a previously tested control.
    submission = dict(path=None, reason="O controlo venceu a seleção interna; sem novo CSV.")
    if k == "067" or selected is not candidates[0]:
        full = data["full"]
        result, _ = _prediction(selected["spec"], full["X_train"], full["y_train"], full["X_test"])
        submission = _export(k, result["evaluation"])
    difference = np.asarray(selected["folds"])-np.asarray(candidates[0]["folds"])
    _figure(k, candidates, data["v"], chosen_prediction)
    return dict(candidates=candidates, selected=selected, fixed=fixed, submission=submission,
                paired_cv=dict(differences=difference.tolist(), mean=float(difference.mean()), std=float(difference.std(ddof=1))),
                note="Seleção só pela CV no treino. A validação fixa é desenvolvimento e não escolhe parâmetros. Configurações de experiências anteriores já foram selecionadas nestas mesmas linhas; CV pós-seleção não é avaliação independente.")


def _stability(data, records):
    models = [("controlo025", _legacy_recipe("025")), ("ensemble058", _legacy_recipe("058")),
              ("melhor_svm", _best(records, ["070"]))]
    rows = []
    for label, spec in models:
        scores, artifacts = [], {}
        for seed in [42, 73, 123]:
            print("073", label, "seed de partição", seed, flush=True)
            cv, oof, probabilities = _cv(spec, data, "073", fold_seed=seed)
            scores.extend(cv["folds"])
            path = OUTPUT/"oof"/f"073_{seed}_{previous._digest(_signature(spec))[:24]}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as stream:
                np.savez_compressed(stream, prediction=oof, probabilities=probabilities,
                                    indices=data["X"].index.to_numpy(), labels=data["y"].to_numpy())
            artifacts[str(seed)] = str(path.relative_to(ROOT))
        rows.append(dict(label=label, spec=spec, scores=scores, mean=float(np.mean(scores)),
                         std=float(np.std(scores, ddof=1)), artifacts=artifacts))
    selected = max(rows, key=lambda r: r["mean"])
    baseline = np.asarray(rows[0]["scores"])
    for row in rows:
        delta = np.asarray(row["scores"])-baseline
        row["paired_vs_025"] = dict(differences=delta.tolist(), mean=float(delta.mean()), std=float(delta.std(ddof=1)))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(range(len(rows)), [r["mean"] for r in rows], yerr=[r["std"] for r in rows], marker="o", capsize=4)
    ax.set(xticks=range(len(rows)), xticklabels=[r["label"] for r in rows], ylabel="Macro F1 média ± DP dos 9 folds", title="073 — estabilidade no desenvolvimento")
    fig.tight_layout()
    fig.savefig(REPORT/"experiencia_073_estabilidade.png", dpi=130)
    plt.close(fig)
    return dict(candidates=rows, selected=selected, submission=None,
                note="Sementes 42/73/123 variam as partições, não a inicialização dos modelos (42). Folds correlacionados: DP não é intervalo de confiança. Não foi consultada a validação fixa. Sem CSV Kaggle.")


def _errors(data, records):
    selected = records["073"]["selected"]
    path = ROOT/selected["artifacts"]["42"]
    with np.load(path, allow_pickle=False) as f:
        prediction, probs = f["prediction"], f["probabilities"]
        if not np.array_equal(f["indices"], data["X"].index) or not np.array_equal(f["labels"], data["y"]):
            raise ValueError("As previsões OOF não estão alinhadas com os dados atuais.")
    actual = data["y"].to_numpy()
    gap = prediction-actual
    confidence = np.max(probs, axis=1) if np.isfinite(probs).all() else np.full(len(actual), np.nan)
    table = pd.DataFrame(dict(original_index=data["X"].index, declared_class=actual,
                              predicted_class=prediction, code_gap=gap, absolute_code_gap=np.abs(gap),
                              profile_flag=np.abs(gap)>=3, confidence=confidence))
    table["direction"] = np.where(gap > 0, "previsto mais velho", np.where(gap < 0, "previsto mais novo", "mesmo escalao"))
    raw_ids = pd.read_csv(ROOT/"notebooks/dataset_train.csv", usecols=["PID", "HHID"])
    table["PID"] = raw_ids.loc[data["X"].index, "PID"].to_numpy()
    table["HHID"] = raw_ids.loc[data["X"].index, "HHID"].to_numpy()
    columns = [c for c in ["SEX", "REGION", "HATLEVEL", "MAINSTAT", "HHINCOME", "HS1", "HS2", "HS3", "BM1", "BM2", "BMI"] if c in data["X"]]
    for c in columns:
        table[c] = data["X"][c].to_numpy()
    file = REPORT/"074_perfis_oof.csv"
    table.to_csv(file, index=False)
    cases = table[table.profile_flag].sort_values(["absolute_code_gap", "confidence", "original_index"], ascending=[False, False, True]).head(10)
    cases.to_csv(REPORT/"074_casos_para_revisao.csv", index=False)
    by_class = table.groupby("declared_class").agg(n=("profile_flag", "size"), flags=("profile_flag", "sum"), mean_absolute_code_gap=("absolute_code_gap", "mean"))
    by_class["flag_rate"] = by_class["flags"] / by_class["n"]
    by_class.to_csv(REPORT/"074_resumo_por_escalao.csv")
    _, oof, _ = _cv(selected["spec"], data, "074", fold_seed=42)  # cache only for completed folds
    if not np.array_equal(oof, prediction):
        raise ValueError("As previsões OOF atuais diferem das guardadas em 073.")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(by_class.index, by_class.flag_rate)
    ax.set(xticks=CLASSES, xlabel="Escalão declarado", ylabel="Proporção com desvio ≥3 códigos", title="074 — sinalização exploratória OOF")
    fig.tight_layout()
    fig.savefig(REPORT/"experiencia_074_perfis.png", dpi=130)
    plt.close(fig)
    return dict(model=selected["label"], metrics_oof=previous._metrics(actual, prediction),
                flagged=int(table.profile_flag.sum()), total=len(table), code_threshold=3,
                selected_cases=json.loads(cases.to_json(orient="records")), by_class=json.loads(by_class.reset_index().to_json(orient="records")),
                report_csv=str(file.relative_to(ROOT)), submission=None,
                note="OOF com seed 42; modelo escolhido no desenvolvimento em 073. Desvio entre códigos não equivale a anos nem comprova idade biológica. Rever elegibilidade das perguntas, incerteza e contexto antes de interpretar um caso. PID/HHID apenas ligam os relatórios aos registos; não entram no modelo.")


def _normal(value):
    return "".join(c for c in unicodedata.normalize("NFKD", str(value).strip().casefold()) if not unicodedata.combining(c))


def _profiles(data):
    X, y = data["X"], data["y"]
    numeric = [c for c in ["BM1", "BM2", "BMI", "HH_ACT", "HH_INACT"] if c in X]
    categories = [c for c in ["SEX", "REGION", "DEG_URB", "HATLEVEL", "MAINSTAT", "HHINCOME", "HS1", "HS2", "HS3", "AM1", "IN16"] if c in X]
    categories += [c for c in X if c.startswith(("CD1", "PL", "PC1", "HA1")) and c not in categories]
    categorical_rows, numeric_rows, disease_rows = [], [], []
    for age in CLASSES:
        group = X.loc[y == age]
        for column in categories:
            values = group[column].astype("string").fillna("Ausente - motivo desconhecido")
            for answer, count in values.value_counts(dropna=False).items():
                categorical_rows.append(dict(age_class=int(age), variable=column, answer=str(answer), n=int(count), proportion=float(count/len(group))))
            if column.startswith("CD1"):
                norm = values.map(_normal)
                valid = norm.isin(["sim", "nao"])
                yes = int(norm.eq("sim").sum())
                disease_rows.append(dict(age_class=int(age), variable=column, yes=yes, yes_no_answers=int(valid.sum()),
                                         proportion_among_yes_no=float(yes/valid.sum()) if valid.any() else None,
                                         other_or_special_answers=int((~valid).sum())))
        for column in numeric:
            values = pd.to_numeric(group[column], errors="coerce")
            numeric_rows.append(dict(age_class=int(age), variable=column, n=int(values.notna().sum()),
                                     missing=int(values.isna().sum()), mean=values.mean(), median=values.median(),
                                     q25=values.quantile(.25), q75=values.quantile(.75)))
    cat, num, diseases = pd.DataFrame(categorical_rows), pd.DataFrame(numeric_rows), pd.DataFrame(disease_rows)
    cat.to_csv(REPORT/"075_perfis_categoricos.csv", index=False)
    num.to_csv(REPORT/"075_perfis_numericos.csv", index=False)
    diseases.to_csv(REPORT/"075_doencas_respostas_validas.csv", index=False)
    top = cat.sort_values("proportion", ascending=False).groupby(["age_class", "variable"], sort=False).head(3)
    text = ["# Caracterização dos escalões — treino de desenvolvimento", "", "Sem resposta, não aplicável e ausências são mantidos distintos. Padrões descritivos não são explicações causais ou diagnósticos. As unidades de BM1/BM2 continuam a depender de confirmação documental.", ""]
    for age in CLASSES:
        text += [f"## Classe {age} — n={int((y==age).sum())}", ""]
        for column in ["HATLEVEL", "MAINSTAT", "HHINCOME", "HS1", "HS2"]:
            rows = top[(top.age_class == age) & (top.variable == column)].head(2)
            if len(rows):
                summary = "; ".join(f"{r.answer}: {r.proportion:.1%} (n={r.n})" for r in rows.itertuples())
                text.append(f"- {column}: {summary}")
        text.append("")
    (REPORT/"075_caracterizacao_escaloes.md").write_text("\n".join(text), encoding="utf-8")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(CLASSES, [int((y==age).sum()) for age in CLASSES])
    ax.set(xticks=CLASSES, xlabel="Escalão", ylabel="Indivíduos no treino", title="075 — suporte da caracterização")
    fig.tight_layout()
    fig.savefig(REPORT/"experiencia_075_suporte.png", dpi=130)
    plt.close(fig)
    return dict(n=len(X), categorical_profiles=json.loads(top.to_json(orient="records")),
                numeric_profiles=json.loads(num.to_json(orient="records")),
                chronic_diseases=json.loads(diseases.to_json(orient="records")), submission=None,
                note="Só 8769 linhas de treino. Prevalências CD1 usam apenas respostas explícitas Sim/Não; restantes respostas são reportadas separadamente. Não inferir escalas clínicas nem converter Não aplicável em ausência de doença.")


def run_experiment(k):
    k = str(k).zfill(3)
    if k not in NAMES:
        raise ValueError("Usar 059–075.")
    REPORT.mkdir(exist_ok=True)
    records = _all()
    for dep in DEPENDENCIES[k]:
        if dep not in records:
            raise ValueError(f"{k} depende de {dep}; executar primeiro essa experiência.")
        if records[dep]["context"]["base"] != _base_context():
            raise ValueError(f"{dep}: dados/preparação/versões mudaram; rever os resultados antes de continuar.")
    recipe = (_recipe(k, records) if int(k) <= 72 else
              [("025", _legacy_recipe("025")), ("058", _legacy_recipe("058")), ("svm", _best(records, ["070"]))] if k == "073" else
              [("selecionado_073", records["073"]["selected"]["spec"])] if k == "074" else [])
    signature = _context(k, recipe)
    if k in records:
        if records[k]["context"] != _json(signature):
            raise ValueError(f"{k}: contexto mudou; não substituir resultados de outra versão silenciosamente.")
        print(k, "resultado completo reutilizado após verificar o contexto.", flush=True)
        return records[k]
    data = previous._data()
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        result = (_stability(data, records) if k == "073" else _errors(data, records) if k == "074" else
                  _profiles(data) if k == "075" else _classification(k, recipe, data))
    result.update(experiment=k, context=signature, seconds=time.perf_counter()-started,
                  kaggle_score=None, module_sha256=previous._hash(Path(__file__)))
    records[k] = result
    _save(RESULTS, records)
    write_report()
    return result


def write_report():
    records = _all()
    lines = ["# Experiências 059–075", "", "Seleção interna/CV apenas nas 8.769 linhas de treino. Validação fixa de desenvolvimento, já observada anteriormente. Pontuações Kaggle por preencher.", "",
             "| ID | Tema | Macro F1 CV | Macro F1 fixa | CSV Kaggle | Estado |", "|---|---|---:|---:|---|---|"]
    for k, name in NAMES.items():
        r = records.get(k)
        if r is None:
            lines.append(f"| {k} | {name} | — | — | — | Por executar |")
        else:
            cv = r.get("selected", {}).get("mean")
            fixed = r.get("fixed", {}).get("selecionado", {}).get("validation", {}).get("macro_f1")
            csv = r.get("submission") or {}
            if cv is not None and fixed is not None:
                lines.append(f"| {k} | {name} | {cv:.4f} | {fixed:.4f} | {csv.get('path') or 'Não gerado'} | Concluído |")
            else:
                lines.append(f"| {k} | {name} | {format(cv,'.4f') if cv is not None else '—'} | — | — | Análise concluída |")
    lines += ["", "## Ler os resultados", "", "- 059–072 podem gerar CSV; 073–075 são análises e não geram submissões.",
              "- Vencer um controlo interno não garante melhoria pública. Não há obrigação de preencher as três submissões diárias.",
              "- As etapas aprendidas (incluindo SVD, calibração e limiares) são ajustadas dentro do treino correspondente. Os limiares SVR usam OOF interno.",
              "- Configurações são escolhidas sequencialmente usando estas mesmas linhas; métricas CV pós-seleção são desenvolvimento, não avaliação final independente.",
              "- SVR estima códigos ordinais, não anos de idade. Desvio de códigos e ruído de perfil não são diagnósticos clínicos.",
              "- Resultados anteriores 001–058 são preservados. Não há submissão, commit ou push automático."]
    path = REPORT/"experiments_059_075.md"
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")
    return path
