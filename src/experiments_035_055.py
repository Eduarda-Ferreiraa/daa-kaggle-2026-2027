"""Reproducible, sequential runners for DAA methods 035–055."""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import os
import time
import warnings
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, pairwise_distances
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA, TruncatedSVD
from xgboost import XGBClassifier

from src.experiments_016_018 import (
    AGE_BANDS,
    CLASS_IDS,
    NUM_CLASSES,
    _classification_metrics,
    _export_predictions,
    _official_row_reference,
    align_probabilities,
    fit_preprocessor,
    load_experiment_data,
    make_model_pipeline,
    make_xgb,
    to_dense_float32,
)
# Importações de experiências posteriores retiradas: os métodos 035–041
# só dependem da preparação e dos utilitários originais de 016–018.
from src.preparar_dados import DataPreprocessor, build_preprocessor, preparar_dados
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = ROOT / "reports" / ".experiments_035_055_retoma_checkpoint.json"
RESULTS_PATH = ROOT / "reports" / ".experiments_035_055_retoma_results.json"
SEED = 42
CV_FOLDS = 3
BEST_CONTROL_FIXED_F1 = 0.4614702838167389
MAX_DENSE_MEMORY_FRACTION = 0.40
torch.set_num_threads(1)


def _raw_frames():
    return (pd.read_csv(ROOT/'notebooks/dataset_train.csv'),
            pd.read_csv(ROOT/'notebooks/dataset_test_X.csv'))


def _compare_and_export(data, predictions, filename):
    ref=_official_row_reference(data)['RowId'].reset_index(drop=True)
    pred=np.asarray(predictions)
    if len(pred)!=3655 or not np.isin(pred,CLASS_IDS).all():
        raise ValueError('Previsões inválidas; não exportar.')
    out=pd.DataFrame({'RowId':ref,'AGE_RANGE':pred.astype(np.int64)})
    differences={}
    for old in sorted((ROOT/'submissions').glob('submission_*.csv')):
        f=pd.read_csv(old)
        if list(f)!=['RowId','AGE_RANGE'] or len(f)!=3655 or not f.RowId.is_unique:
            continue
        f=f.set_index('RowId').reindex(ref)
        if not f.AGE_RANGE.isna().any():
            differences[old.name]=int((f.AGE_RANGE.to_numpy()!=pred).sum())
    duplicates=[name for name,count in differences.items() if count==0]
    if duplicates:return {'path':None,'duplicate_of':duplicates,'differences':differences}
    path=ROOT/'submissions'/filename
    if path.exists():raise FileExistsError(f'Não substituir {path}')
    out.to_csv(path,index=False)
    check=pd.read_csv(path)
    if not check.equals(out):raise ValueError('CSV diferente após releitura.')
    return {'path':str(path.relative_to(ROOT)),'differences':differences,'duplicate_of':[]}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _versions() -> dict[str, str | None]:
    versions = {}
    for package in (
        "numpy",
        "pandas",
        "scikit-learn",
        "torch",
        "xgboost",
        "catboost",
        "lightgbm",
        "shap",
        "scikit-learn-extra",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _context(experiment: str, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "config": config,
        "train_sha256": _sha256(ROOT / "notebooks" / "dataset_train.csv"),
        "test_sha256": _sha256(ROOT / "notebooks" / "dataset_test_X.csv"),
        "preparation_sha256": _sha256(ROOT / "src" / "preparar_dados.py"),
        "runner_sha256": _sha256(Path(__file__)),
        "versions": _versions(),
    }


def _load_checkpoint(context: dict[str, Any]) -> dict[str, Any]:
    if not CHECKPOINT_PATH.is_file():
        return {"contexts": {context["experiment"]: context}, "experiments": {}}
    checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    experiment = context["experiment"]
    previous = checkpoint.setdefault("contexts", {}).get(experiment)
    if previous is not None and previous != context:
        raise ValueError(
            f"Checkpoint {experiment} incompatível com dados, preparação, parâmetros, "
            "código ou versões; não será reutilizado."
        )
    checkpoint["contexts"][experiment] = context
    checkpoint.setdefault("experiments", {})
    return checkpoint


def _save_checkpoint(checkpoint: dict[str, Any]) -> None:
    temporary = CHECKPOINT_PATH.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, CHECKPOINT_PATH)


def _load_results() -> dict[str, Any]:
    if not RESULTS_PATH.is_file():
        return {}
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


def _save_result(experiment: str, result: dict[str, Any]) -> None:
    results = _load_results()
    results[experiment] = result
    temporary = RESULTS_PATH.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, RESULTS_PATH)
    write_overall_report()


def _validate_features(X: pd.DataFrame) -> None:
    forbidden = {"AGE_RANGE", "PID", "HHID", "PROXY", "COUNTRY"}
    leaked = sorted(forbidden.intersection(X.columns))
    if leaked:
        raise ValueError(f"Colunas proibidas nas features: {leaked}")


def _indices_hash(index: Any) -> str:
    return hashlib.sha256(np.asarray(index).tobytes()).hexdigest()


def _folds(X: pd.DataFrame, y: pd.Series) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    return list(splitter.split(X, y))


def _scores(y: Any, predictions: Any) -> dict[str, Any]:
    return _classification_metrics(
        np.asarray(y, dtype=np.int64), np.asarray(predictions, dtype=np.int64)
    )


def _pipeline(estimator: Any, *, normalize_numeric: bool = False) -> Pipeline:
    return make_model_pipeline(estimator, normalize_numeric=normalize_numeric)


def _model(experiment: str, params: dict[str, Any]) -> Any:
    if experiment == "036":
        return KNeighborsClassifier(
            n_neighbors=params["n_neighbors"],
            weights=params["weights"],
            n_jobs=1,
        )
    if experiment == "037":
        return LinearSVC(
            C=params["C"],
            class_weight=params["class_weight"],
            random_state=SEED,
            dual="auto",
            max_iter=params.get(
                "max_iter_used_after_warning", params.get("max_iter", 10000)
            ),
            tol=1e-4,
        )
    if experiment == "038":
        tree = DecisionTreeClassifier(
            max_depth=params["max_depth"], random_state=SEED
        )
        if params["kind"] == "tree":
            return tree
        return BaggingClassifier(
            estimator=tree,
            n_estimators=100,
            max_samples=1.0,
            bootstrap=True,
            random_state=SEED,
            n_jobs=1,
        )
    if experiment == "039":
        estimator = DecisionTreeClassifier(
            max_depth=params["base_depth"], random_state=SEED
        )
        return AdaBoostClassifier(
            estimator=estimator,
            n_estimators=200,
            learning_rate=params["learning_rate"],
            random_state=SEED,
        )
    if experiment == "040":
        return GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            random_state=SEED,
            max_features=None,
        )
    raise ValueError(f"Modelo de CV não definido para {experiment}.")


def _dense_memory_preflight(rows: int, columns: int, *, dtype_bytes: int = 8) -> dict[str, Any]:
    bytes_required = int(rows) * int(columns) * int(dtype_bytes)
    available = int(psutil.virtual_memory().available)
    conservative = bytes_required * 3
    enough = conservative <= available * MAX_DENSE_MEMORY_FRACTION
    return {
        "rows": int(rows),
        "columns": int(columns),
        "dtype": "float64",
        "one_dense_matrix_mib": bytes_required / 2**20,
        "three_matrix_working_estimate_mib": conservative / 2**20,
        "available_memory_mib": available / 2**20,
        "memory_fraction_limit": MAX_DENSE_MEMORY_FRACTION,
        "approved": enough,
    }


def _fit_gb_fold(
    params: dict[str, Any],
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_eval: pd.DataFrame,
) -> tuple[Any, Any, np.ndarray, np.ndarray, dict[str, Any], float]:
    preprocessor = build_preprocessor(X_fit, normalize_numeric=False)
    fit_sparse = preprocessor.transform(X_fit)
    eval_sparse = preprocessor.transform(X_eval)
    # A API instalada aceita CSR e converte para float32. Evitar uma
    # densificação que não faz parte do algoritmo nem dos hiperparâmetros.
    from scipy import sparse
    fit_dense = sparse.csr_matrix(fit_sparse, dtype=np.float32)
    eval_dense = sparse.csr_matrix(eval_sparse, dtype=np.float32)
    estimate = {
        "representation": "CSR float32 (sem densificação)",
        "rows": int(fit_dense.shape[0]), "columns": int(fit_dense.shape[1]),
        "nnz": int(fit_dense.nnz),
        "fit_matrix_mib": (fit_dense.data.nbytes + fit_dense.indices.nbytes + fit_dense.indptr.nbytes) / 2**20,
        "eval_matrix_mib": (eval_dense.data.nbytes + eval_dense.indices.nbytes + eval_dense.indptr.nbytes) / 2**20,
    }
    started = time.perf_counter()
    model = _model("040", params)
    with threadpool_limits(limits=1):
        model.fit(fit_dense, y_fit)
    seconds = time.perf_counter() - started
    return model, preprocessor, fit_dense, eval_dense, estimate, seconds


def _fit_pipeline_fold(
    experiment: str,
    params: dict[str, Any],
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_eval: pd.DataFrame,
) -> tuple[Any, np.ndarray, float, list[str], dict[str, Any] | None]:
    pipeline = _pipeline(
        _model(experiment, params),
        normalize_numeric=experiment in {"036", "037"},
    )
    warnings_found: list[str] = []
    started = time.perf_counter()
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline.fit(X_fit, y_fit)
        warnings_found.extend(
            f"{item.category.__name__}: {item.message}" for item in caught
        )
    if experiment == "037" and any(
        item.startswith("ConvergenceWarning:") for item in warnings_found
    ):
        retry_params = dict(params, max_iter=50000)
        retry = _pipeline(
            _model(experiment, retry_params), normalize_numeric=True
        )
        with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            retry.fit(X_fit, y_fit)
            retry_warnings = [f"{x.category.__name__}: {x.message}" for x in caught]
        warnings_found.extend(
            ["Retry with max_iter=50000 after convergence warning."]
            + retry_warnings
        )
        pipeline = retry
        params["max_iter_used_after_warning"] = 50000
    predictions = pipeline.predict(X_eval)
    return (
        pipeline,
        np.asarray(predictions, dtype=np.int64),
        time.perf_counter() - started,
        warnings_found,
        None,
    )


def _plot_cv(experiment: str, candidates: list[dict[str, Any]]) -> str:
    path = ROOT / "reports" / f"experiencia_{experiment}_curvas.png"
    labels = [row["label"] for row in candidates]
    means = [row["cv_macro_f1_mean"] for row in candidates]
    errors = [row["cv_macro_f1_std"] for row in candidates]
    figure, axis = plt.subplots(figsize=(max(8, len(labels) * 1.1), 4.5))
    axis.errorbar(np.arange(len(labels)), means, yerr=errors, marker="o", capsize=4)
    axis.set(
        title=f"{experiment} — Macro F1 CV estratificada",
        xlabel="Configuração",
        ylabel="Macro F1 média ± DP",
        xticks=np.arange(len(labels)),
        xticklabels=labels,
    )
    axis.tick_params(axis="x", labelrotation=35)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return str(path.relative_to(ROOT))


def _plot_confusion(experiment: str, y: Any, prediction: Any) -> str:
    from sklearn.metrics import confusion_matrix

    path = ROOT / "reports" / f"experiencia_{experiment}_matriz_confusao.png"
    matrix = confusion_matrix(y, prediction, labels=CLASS_IDS)
    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        title=f"{experiment} — validação fixa",
        xlabel="Previsto",
        ylabel="Real",
        xticks=CLASS_IDS,
        yticks=CLASS_IDS,
    )
    axis.set_xticklabels(CLASS_IDS, rotation=45)
    axis.set_yticklabels(CLASS_IDS)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return str(path.relative_to(ROOT))


def _candidate_grid(experiment: str) -> list[dict[str, Any]]:
    if experiment == "036":
        return [
            {"n_neighbors": k, "weights": weight}
            for k in (5, 15, 31)
            for weight in ("uniform", "distance")
        ]
    if experiment == "037":
        return [
            {"C": c, "class_weight": weight, "max_iter": 10000}
            for c in (0.01, 0.1, 1.0)
            for weight in (None, "balanced")
        ]
    if experiment == "038":
        return [
            {"kind": "tree", "max_depth": 5},
            {"kind": "bagging", "max_depth": 5},
            {"kind": "bagging", "max_depth": 10},
        ]
    if experiment == "039":
        return [
            {"base_depth": depth, "learning_rate": rate}
            for depth in (1, 2)
            for rate in (0.05, 0.5)
        ]
    if experiment == "040":
        return [
            {"max_depth": depth, "learning_rate": rate}
            for depth in (2, 3)
            for rate in (0.05, 0.1)
        ]
    raise ValueError(f"Sem grelha para {experiment}.")


def _label_params(experiment: str, params: dict[str, Any]) -> str:
    parts = []
    for key, value in params.items():
        if key == "max_iter":
            continue
        parts.append(f"{key}={value}")
    return ", ".join(parts)


def _run_supervised_cv(experiment: str) -> dict[str, Any]:
    data = load_experiment_data()
    _validate_features(data["X_train"])
    configs = _candidate_grid(experiment)
    config = {
        "grid": configs,
        "cv": {"n_splits": CV_FOLDS, "shuffle": True, "random_state": SEED},
        "selection": "mean unweighted sklearn Macro F1 on training CV only",
        "normalization": experiment in {"036", "037"},
        "thread_count": 1,
    }
    if experiment == "038":
        config["bagging"] = {"n_estimators": 100, "n_jobs": 1}
    if experiment == "039":
        config["n_estimators"] = 200
    if experiment == "040":
        config["n_estimators"] = 200
    context = _context(experiment, config)
    checkpoint = _load_checkpoint(context)
    stored = checkpoint["experiments"].setdefault(
        experiment, {"folds": {}, "result": None}
    )
    if stored.get("complete") and stored.get("result"):
        print(f"{experiment}: resultado reutilizado após validar checkpoint.")
        return stored["result"]

    folds = _folds(data["X_train"], data["y_train"])
    split_hashes = [
        {
            "train": _indices_hash(data["X_train"].index[train]),
            "validation": _indices_hash(data["X_train"].index[valid]),
        }
        for train, valid in folds
    ]
    cv_rows = []
    for candidate_index, params in enumerate(configs):
        key = str(candidate_index)
        fold_results = stored["folds"].setdefault(key, {})
        for fold_index, (train_indices, valid_indices) in enumerate(folds):
            fold_key = str(fold_index)
            if fold_key in fold_results:
                continue
            print(
                f"{experiment}: candidato {candidate_index + 1}/{len(configs)}, "
                f"fold {fold_index + 1}/{CV_FOLDS}"
            )
            X_fit = data["X_train"].iloc[train_indices]
            y_fit = data["y_train"].iloc[train_indices]
            X_valid = data["X_train"].iloc[valid_indices]
            y_valid = data["y_train"].iloc[valid_indices]
            if experiment == "040":
                model, preprocessor, fit_features, valid_features, memory, seconds = _fit_gb_fold(
                    params, X_fit, y_fit, X_valid
                )
                predictions = model.predict(valid_features)
                train_predictions = model.predict(fit_features)
                warning_rows: list[str] = []
                fit_metric = _scores(y_fit, train_predictions)
                valid_metric = _scores(y_valid, predictions)
                del preprocessor, fit_features, valid_features
                memory_note = memory
            else:
                model, predictions, seconds, warning_rows, _ = _fit_pipeline_fold(
                    experiment, params, X_fit, y_fit, X_valid
                )
                fit_metric = _scores(y_fit, model.predict(X_fit))
                valid_metric = _scores(y_valid, predictions)
                memory_note = None
            fold_results[fold_key] = {
                "train": fit_metric,
                "validation": valid_metric,
                "seconds": seconds,
                "warnings": warning_rows,
                "memory_preflight": memory_note,
            }
            _save_checkpoint(checkpoint)
            del model
            gc.collect()
        rows = [fold_results[str(i)] for i in range(CV_FOLDS)]
        scores = [row["validation"]["macro_f1"] for row in rows]
        cv_rows.append(
            {
                "params": params,
                "label": _label_params(experiment, params),
                "cv_macro_f1": scores,
                "cv_macro_f1_mean": float(np.mean(scores)),
                "cv_macro_f1_std": float(np.std(scores, ddof=1)),
                "cv_accuracy_mean": float(
                    np.mean([row["validation"]["accuracy"] for row in rows])
                ),
                "cv_train_macro_f1_mean": float(
                    np.mean([row["train"]["macro_f1"] for row in rows])
                ),
                "folds": rows,
            }
        )
    selected = max(cv_rows, key=lambda row: row["cv_macro_f1_mean"])

    if experiment == "040":
        final_model, prep, train_features, val_features, memory, fit_seconds = _fit_gb_fold(
            selected["params"],
            data["X_train"],
            data["y_train"],
            data["X_val"],
        )
        train_predictions = final_model.predict(train_features)
        val_predictions = final_model.predict(val_features)
        final_memory = memory
        del prep, train_features, val_features
    else:
        final_model, val_predictions, fit_seconds, warnings_final, _ = _fit_pipeline_fold(
            experiment,
            dict(selected["params"]),
            data["X_train"],
            data["y_train"],
            data["X_val"],
        )
        train_predictions = final_model.predict(data["X_train"])
        final_memory = None
    fixed_metrics = {
        "train": _scores(data["y_train"], train_predictions),
        "validation": _scores(data["y_val"], val_predictions),
        "fit_seconds": fit_seconds,
    }
    export_result: dict[str, Any] = {
        "path": None,
        "reason": "Candidato sem nova submissão; consultar comparação e duplicados.",
        "differences": {},
    }
    if (
        experiment != "038" or selected["params"].get("kind") == "bagging"
    ):
        raw_train, raw_test = _raw_frames()
        full = preparar_dados(raw_train, raw_test)
        if experiment == "040":
            final_model, _, full_features, test_features, full_memory, export_seconds = _fit_gb_fold(
                selected["params"], full["X_train"], full["y_train"], full["X_test"]
            )
            test_predictions = final_model.predict(test_features)
            final_memory = {"fixed": final_memory, "full": full_memory}
        else:
            final_model, test_predictions, export_seconds, _, _ = _fit_pipeline_fold(
                experiment,
                dict(selected["params"]),
                full["X_train"],
                full["y_train"],
                full["X_test"],
            )
        filename = f"submission_{experiment}_novo_modelo.csv"
        export_result = _compare_and_export(data, test_predictions, filename)
        export_result["fit_seconds"] = export_seconds
    chart = _plot_cv(experiment, cv_rows)
    confusion_path = _plot_confusion(
        experiment, data["y_val"], val_predictions
    )
    result = {
        "experiment": experiment,
        "configuration": config,
        "versions": _versions(),
        "seed": SEED,
        "cv_split_index_sha256": split_hashes,
        "candidates": cv_rows,
        "selected_internally": selected,
        "fixed_metrics": fixed_metrics,
        "memory_preflight": final_memory,
        "warnings": (
            warnings_final
            if experiment != "040"
            else [
                warning
                for candidate in cv_rows
                for fold in candidate["folds"]
                for warning in fold["warnings"]
            ]
        ),
        "curves": chart,
        "confusion_matrix": confusion_path,
        "submission": export_result,
        "kaggle_score": None,
        "elapsed_model_fit_seconds": float(
            sum(row["seconds"] for candidate in cv_rows for row in candidate["folds"])
            + fit_seconds
        ),
        "control_reference_fixed_macro_f1": BEST_CONTROL_FIXED_F1,
        "exploratory": fixed_metrics["validation"]["macro_f1"]
        <= BEST_CONTROL_FIXED_F1,
    }
    _save_result(experiment, result)
    stored["result"] = result
    stored["complete"] = True
    _save_checkpoint(checkpoint)
    return result


def run_035() -> dict[str, Any]:
    experiment = "035"
    config = {
        "strategies": ["most_frequent", "stratified"],
        "seed": SEED,
        "submission": False,
    }
    data = load_experiment_data()
    _validate_features(data["X_train"])
    context = _context(experiment, config)
    checkpoint = _load_checkpoint(context)
    stored = checkpoint["experiments"].setdefault(experiment, {})
    if stored.get("complete") and stored.get("result"):
        return stored["result"]
    results = []
    for strategy in config["strategies"]:
        model = DummyClassifier(strategy=strategy, random_state=SEED)
        model.fit(data["X_train"], data["y_train"])
        results.append(
            {
                "strategy": strategy,
                "train": _scores(
                    data["y_train"], model.predict(data["X_train"])
                ),
                "validation": _scores(data["y_val"], model.predict(data["X_val"])),
            }
        )
    result = {
        "experiment": experiment,
        "configuration": config,
        "versions": _versions(),
        "candidates": results,
        "submission": None,
        "kaggle_score": None,
        "elapsed_model_fit_seconds": 0.0,
        "note": "Baseline mínima; serve apenas como referência, sem CSV.",
    }
    _save_result(experiment, result)
    stored["result"] = result
    stored["complete"] = True
    _save_checkpoint(checkpoint)
    return result


def run_041() -> dict[str, Any]:
    """Seleção interna; a validação fixa não participa no early stopping."""
    import lightgbm as lgb
    data=load_experiment_data()
    config={'grid':[{'num_leaves':leaves,'learning_rate':lr} for leaves in [15,31] for lr in [.03,.1]],
            'iterations_max':1000,'early_stopping':50,'metric':'Macro F1 não ponderada',
            'seed':SEED,'num_threads':1,'protocol':'7453 treino interno / 1316 seleção; validação fixa 2193 só para desenvolvimento'}
    context=_context('041',config);checkpoint=_load_checkpoint(context)
    stored=checkpoint['experiments'].setdefault('041',{'candidates':[]})
    if stored.get('complete'):return stored['result']
    prep=build_preprocessor(data['X_inner_train'],normalize_numeric=False)
    A=prep.transform(data['X_inner_train']);V=prep.transform(data['X_inner_val'])
    def metric(y_true,prob):
        from sklearn.metrics import f1_score
        return 'macro_f1',f1_score(y_true,prob.argmax(axis=1),labels=CLASS_IDS,average='macro',zero_division=0),True
    for params in config['grid'][len(stored['candidates']):]:
        print('041 candidato:',params,flush=True);started=time.perf_counter()
        model=lgb.LGBMClassifier(objective='multiclass',num_class=16,n_estimators=1000,random_state=SEED,n_jobs=1,verbosity=-1,metric='None',**params)
        model.fit(A,data['y_inner_train'],eval_set=[(A,data['y_inner_train']),(V,data['y_inner_val'])],eval_names=['train','validation'],eval_metric=metric,
                  callbacks=[lgb.early_stopping(50,first_metric_only=True,verbose=True),lgb.log_evaluation(100)])
        scores=_scores(data['y_inner_val'],model.predict(V))
        stored['candidates'].append({'params':params,'label':str(params),'iterations':int(model.best_iteration_),
            'internal_macro_f1':scores['macro_f1'],'seconds':time.perf_counter()-started,'history':model.evals_result_})
        _save_checkpoint(checkpoint);del model;gc.collect()
    chosen=max(stored['candidates'],key=lambda x:x['internal_macro_f1'])
    def fit(X,y,Z):
        pp=build_preprocessor(X,normalize_numeric=False)
        m=lgb.LGBMClassifier(objective='multiclass',num_class=16,n_estimators=chosen['iterations'],random_state=SEED,n_jobs=1,verbosity=-1,**chosen['params'])
        m.fit(pp.transform(X),y)
        return m.predict(pp.transform(X)),m.predict(pp.transform(Z))
    train_pred,val_pred=fit(data['X_train'],data['y_train'],data['X_val'])
    fixed={'train':_scores(data['y_train'],train_pred),'validation':_scores(data['y_val'],val_pred)}
    full=data['prepared'];_,test_pred=fit(full['X_train'],full['y_train'],full['X_test'])
    exported=_compare_and_export(data,test_pred,'submission_041_lightgbm.csv')
    chart=ROOT/'reports/experiencia_041_curvas.png'
    fig,ax=plt.subplots(figsize=(11,5))
    for c in stored['candidates']:
        for split,style in [('train','--'),('validation','-')]:
            values=c['history'][split]['macro_f1'];ax.plot(range(1,len(values)+1),values,style,label=f"{c['label']} {split}")
    ax.set(xlabel='Iteração',ylabel='Macro F1',title='041 — treino e validação interna');ax.legend(fontsize=7);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(chart,dpi=130);plt.close(fig)
    result={'experiment':'041','configuration':config,'versions':_versions(),'candidates':stored['candidates'],
            'selected_internally':chosen,'fixed_metrics':fixed,'submission':exported,'curves':str(chart.relative_to(ROOT)),
            'confusion_matrix':_plot_confusion('041',data['y_val'],val_pred),'kaggle_score':None,
            'exploratory':fixed['validation']['macro_f1']<=BEST_CONTROL_FIXED_F1}
    stored.update(complete=True,result=result);_save_checkpoint(checkpoint);_save_result('041',result)
    return result


def run_experiment(experiment: str) -> dict[str, Any]:
    dispatch: dict[str, Callable[[], dict[str, Any]]] = {
        "035": run_035,
        "036": lambda: _run_supervised_cv("036"),
        "037": lambda: _run_supervised_cv("037"),
        "038": lambda: _run_supervised_cv("038"),
        "039": lambda: _run_supervised_cv("039"),
        "040": lambda: _run_supervised_cv("040"),
        "041": run_041,
    }
    if experiment not in dispatch:
        raise ValueError(f"Experiência ainda não implementada: {experiment}")
    return dispatch[experiment]()


def write_overall_report() -> Path:
    results = _load_results()
    names = {
        "035": "DummyClassifier",
        "036": "KNN",
        "037": "LinearSVC",
        "038": "Árvore / Bagging",
        "039": "AdaBoost",
        "040": "GradientBoosting sklearn",
        "041": "LightGBM",
        "042": "Stacking OOF",
        "043": "SelectPercentile",
        "044": "TruncatedSVD",
        "045": "PCA",
        "046": "Autoencoder latente",
        "047": "Erro AE cross-fit",
        "048": "ID categórico K-Means",
        "049": "K-Medoids",
        "050": "DBSCAN descritivo",
        "051": "Early stopping com restauração",
        "052": "Ablação de normalização MLP",
        "053": "Importância por permutação",
        "054": "TreeSHAP XGBoost",
        "055": "PDP / ICE",
    }
    lines = [
        "# Experiências DAA 035–055 — tabela cumulativa",
        "",
        "Auditoria: [checklist de métodos](./method_checklist_035_055.md). Os notebooks existentes até "
        "034 e os respetivos CSVs permanecem preservados. Os números Kaggle são contexto histórico "
        "fornecido no projeto; não foram usados para seleção. A validação fixa é desenvolvimento, "
        "não teste independente.",
        "",
        "| ID | Método | Macro F1 interna/CV | Macro F1 fixa | Accuracy fixa | Kaggle | CSV | Estado |",
        "|---:|---|---:|---:|---:|---:|---|---|",
    ]
    for experiment, name in names.items():
        result = results.get(experiment)
        if not result:
            status = "em fila" if experiment not in ("041", "049", "054") else "a verificar"
            lines.append(
                f"| {experiment} | {name} | — | — | — | — | — | {status} |"
            )
            continue
        if result.get("status") == "skipped":
            lines.append(
                f"| {experiment} | {name} | — | — | — | — | — | "
                f"não executado: {result['reason']} |"
            )
            continue
        candidates = result.get("candidates", [])
        internal = "—"
        if "selected_internally" in result:
            internal = f"{result['selected_internally'].get('cv_macro_f1_mean', result['selected_internally'].get('internal_macro_f1', 0.0)):.4f}"
        elif candidates:
            internal = "—"
        fixed = result.get("fixed_metrics", {})
        fixed_val = fixed.get("validation", {})
        macro = f"{fixed_val['macro_f1']:.4f}" if fixed_val else "—"
        accuracy = f"{fixed_val['accuracy']:.4f}" if fixed_val else "—"
        submission = result.get("submission")
        csv = submission.get("path") if isinstance(submission, dict) else None
        status = "exploratório" if result.get("exploratory") else "avaliado"
        lines.append(
            f"| {experiment} | {name} | {internal} | {macro} | {accuracy} | — | "
            f"{csv or '—'} | {status} |"
        )
    lines.extend(
        [
            "",
            "## Baselines contemporâneos e recomendações",
            "",
            "| Controlo | Configuração resumida | Macro F1 fixa | Kaggle histórico |",
            "|---|---|---:|---:|",
            "| 025 | XGBoost + MLP, mistura 75/25 | 0,4615 | 0,48 |",
            "| 028 | Ensemble 025 + agregados de domínio | consultar resultado guardado | 0,47 |",
            "| 029 | CatBoost categorias nativas | 0,4483 | 0,46 |",
            "| 031 | CatBoost inverse-sqrt | 0,4587 | 0,47 |",
            "| 034 | Ensemble 025 + CatBoost ponderado, 50/50 | 0,4607 | 0,46 |",
            "",
            "Candidatos para Kaggle: ainda não promover nenhum sem superar controles locais ou "
            "apresentar diversidade de erros justificada. Não serão preenchidos lugares à força; "
            "não haverá submissão automática.",
        ]
    )
    path = ROOT / "reports" / "experiments_035_055.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
