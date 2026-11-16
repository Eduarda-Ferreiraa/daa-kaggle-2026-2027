"""Reproducible runners for DAA experiments 030–032."""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from src.experiments_016_018 import (
    CLASS_IDS,
    NUM_CLASSES,
    _classification_metrics,
    _export_predictions,
    _official_row_reference,
    align_probabilities,
    fit_preprocessor,
    load_experiment_data,
    make_xgb,
)
from src.experiments_022_027 import _best_epoch, _fit_mlp, _predict_probabilities
from src.preparar_dados import preparar_dados, preparar_dados_fixos


ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "reports" / ".experiments_030_032_results.json"
CHECKPOINT_PATH = ROOT / "reports" / ".experiments_030_032_checkpoint_v2.json"
SEED = 42
MAX_EPOCHS = 100
CATBOOST_MAX_ITERATIONS = 1000
CATBOOST_EARLY_STOPPING = 50
CATBOOST_SELECTED = {"depth": 4, "learning_rate": 0.1, "iterations": 262}
CATBOOST_RESOURCE_PARAMETERS = {
    "boosting_type": "Plain",
    "max_ctr_complexity": 1,
    "ctr_leaf_count_limit": 256,
    "border_count": 32,
    "one_hot_max_size": 10,
    "counter_calc_method": "SkipTest",
    "rsm": 0.2,
    "leaf_estimation_iterations": 1,
    "used_ram_limit": "512mb",
}
AGGREGATE_GROUPS: dict[str, dict[str, dict[str, Any]]] = {
    "doencas": {
        "CD1": {
            "columns": [f"CD1{chr(65 + i)}" for i in range(16)],
            "positive_values": ["Sim"],
            "negative_values": ["Não"],
        }
    },
    "itens_funcionais_PL": {
        "PL": {
            "columns": ["PL6", "PL7", "PL8"],
            "positive_values": [
                "Alguma dificuldade",
                "Muita dificuldade",
                "Não consegue de todo",
            ],
            "negative_values": ["Nenhuma dificuldade"],
        }
    },
    "autocuidado": {
        "PC1": {
            "columns": [f"PC1{chr(65 + i)}" for i in range(5)],
            "positive_values": [
                "Alguma dificuldade",
                "Muita dificuldade",
                "Não consegue de todo",
            ],
            "negative_values": ["Nenhuma dificuldade"],
        }
    },
    "tarefas": {
        "HA1": {
            "columns": [f"HA1{chr(65 + i)}" for i in range(7)],
            "positive_values": [
                "Alguma dificuldade",
                "Muita dificuldade",
                "Não consegue de todo",
            ],
            "negative_values": ["Nenhuma dificuldade"],
        }
    },
}
VARIANTS_030 = {
    "sem_agregados": [],
    "apenas_doencas": ["doencas"],
    "apenas_itens_funcionais_PL": ["itens_funcionais_PL"],
    "apenas_autocuidado": ["autocuidado"],
    "apenas_tarefas": ["tarefas"],
    "todos_os_agregados": list(AGGREGATE_GROUPS),
}
BLEND_WEIGHTS_032 = (1.0, 0.0), (0.9, 0.1), (0.75, 0.25), (0.5, 0.5), (0.0, 1.0)
torch.set_num_threads(1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_versions() -> dict[str, str]:
    return {
        name: importlib.metadata.version(name)
        for name in ("numpy", "pandas", "scikit-learn", "torch", "xgboost", "catboost")
    }


def _checkpoint_context(experiment: str, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "config": config,
        "train_sha256": _sha256(ROOT / "notebooks" / "dataset_train.csv"),
        "test_sha256": _sha256(ROOT / "notebooks" / "dataset_test_X.csv"),
        "preparation_sha256": _sha256(ROOT / "src" / "preparar_dados.py"),
        "runner_sha256": _sha256(Path(__file__)),
        "versions": _package_versions(),
    }


def _load_checkpoint(context: dict[str, Any]) -> dict[str, Any]:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CHECKPOINT_PATH.is_file():
        return {"contexts": {context["experiment"]: context}, "experiments": {}}
    checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    experiment = context["experiment"]
    stored_context = checkpoint.setdefault("contexts", {}).get(experiment)
    if stored_context is not None and stored_context != context:
        raise ValueError(
            "O checkpoint existente não corresponde aos dados, preparação, "
            "configuração, código ou versões atuais; não será reutilizado."
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


def _record_result(experiment: str, result: dict[str, Any]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    stored = json.loads(RESULTS_PATH.read_text(encoding="utf-8")) if RESULTS_PATH.is_file() else {}
    stored[experiment] = result
    temporary = RESULTS_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(stored, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, RESULTS_PATH)


def _metrics(y_true: Any, probabilities: np.ndarray) -> dict[str, Any]:
    predictions = np.asarray(probabilities).argmax(axis=1)
    return _classification_metrics(np.asarray(y_true, dtype=np.int64), predictions)


def _score(y_true: Any, probabilities: np.ndarray) -> float:
    return float(
        f1_score(
            np.asarray(y_true, dtype=np.int64),
            np.asarray(probabilities).argmax(axis=1),
            labels=CLASS_IDS,
            average="macro",
            zero_division=0,
        )
    )


def _aggregate_mappings(groups: list[str]) -> dict[str, dict[str, Any]] | None:
    mappings: dict[str, dict[str, Any]] = {}
    for group in groups:
        mappings.update(AGGREGATE_GROUPS[group])
    return mappings or None


def _prepared_variant(
    train_frame: pd.DataFrame, reference_index: pd.Index, groups: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    prepared = preparar_dados_fixos(
        train_frame,
        domain_aggregates=_aggregate_mappings(groups),
    )
    X = prepared["X"].loc[reference_index]
    y = prepared["y"].loc[reference_index]
    excluded = {"AGE_RANGE", "PID", "HHID", "PROXY", "COUNTRY"}
    leaked = excluded.intersection(X.columns)
    if leaked:
        raise ValueError(f"Colunas proibidas nas entradas: {sorted(leaked)}")
    if len(X) != len(reference_index):
        raise ValueError("A preparação alterou o número ou alinhamento de linhas.")
    return X, y


def _fold_metrics(
    variant: str,
    fold: int,
    train_metrics: dict[str, Any],
    validation_metrics: dict[str, Any],
    epoch: int,
    seconds: float,
) -> dict[str, Any]:
    return {
        "variant": variant,
        "fold": fold,
        "epoch": epoch,
        "train": train_metrics,
        "validation": validation_metrics,
        "seconds": seconds,
    }


def _plot_confusion(
    targets: Any, predictions: Any, path: Path, title: str
) -> None:
    matrix = confusion_matrix(targets, predictions, labels=CLASS_IDS)
    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        title=title,
        xlabel="Previsto",
        ylabel="Real",
        xticks=CLASS_IDS,
        yticks=CLASS_IDS,
    )
    axis.set_xticklabels(CLASS_IDS, rotation=45)
    axis.set_yticklabels(CLASS_IDS)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _plot_030_histories(
    histories: dict[str, dict[str, dict[str, list[float]]]], variant: str
) -> Path:
    path = ROOT / "reports" / "experiencia_030_curvas.png"
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for fold, history in histories.get(variant, {}).items():
        epochs = history["epoch"]
        label = f"Fold {fold}"
        axes[0].plot(epochs, history["train_loss"], label=f"{label} treino")
        axes[0].plot(
            epochs,
            history["validation_loss"],
            linestyle="--",
            label=f"{label} interno",
        )
        axes[1].plot(epochs, history["train_macro_f1"], label=f"{label} treino")
        axes[1].plot(
            epochs,
            history["validation_macro_f1"],
            linestyle="--",
            label=f"{label} interno",
        )
    axes[0].set(title="MLP — perda ponderada", xlabel="Época", ylabel="Perda")
    axes[1].set(title="MLP — Macro F1", xlabel="Época", ylabel="Macro F1")
    for axis in axes:
        axis.legend(fontsize=7)
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def _plot_031_curves(histories: dict[str, Any]) -> Path:
    path = ROOT / "reports" / "experiencia_031_curvas.png"
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for label, history in histories.items():
        for split, style in (("learn", "-"), ("validation", "--")):
            metrics = history.get(split, {})
            metric_key = next(
                (key for key in metrics if "TotalF1" in key and "Macro" in key),
                None,
            )
            scores = metrics.get(metric_key, []) if metric_key else []
            if scores:
                axes[0].plot(
                    np.arange(1, len(scores) + 1),
                    scores,
                    linestyle=style,
                    label=f"{label} {split}",
                )
        losses = history.get("learn", {}).get("MultiClass", [])
        if losses:
            axes[1].plot(
                np.arange(1, len(losses) + 1),
                losses,
                label=f"{label} treino",
            )
    axes[0].set(title="CatBoost — Macro F1 não ponderada", xlabel="Iteração", ylabel="Macro F1")
    axes[1].set(title="CatBoost — MultiClass", xlabel="Iteração", ylabel="Perda")
    for axis in axes:
        axis.legend(fontsize=8)
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def _plot_032_candidates(
    candidates: list[dict[str, Any]], mlp_history: dict[str, list[float]]
) -> Path:
    path = ROOT / "reports" / "experiencia_032_curvas.png"
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    labels = [row["weight_label"] for row in candidates]
    scores = [row["internal_metrics"]["macro_f1"] for row in candidates]
    axes[0].plot(labels, scores, marker="o")
    axes[0].set(title="Macro F1 interno por mistura", xlabel="Peso ensemble/CatBoost", ylabel="Macro F1")
    axes[0].grid(alpha=0.25)
    axes[1].plot(
        mlp_history["epoch"],
        mlp_history["train_loss"],
        label="MLP 025 — perda treino",
    )
    axes[1].plot(
        mlp_history["epoch"],
        mlp_history["train_macro_f1"],
        label="MLP 025 — Macro F1 treino",
    )
    axes[1].set(title="Curvas MLP do componente 025", xlabel="Época")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def _cat_frame(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    frame = X.copy()
    categorical = [
        column
        for column in frame.columns
        if pd.api.types.is_object_dtype(frame[column])
        or isinstance(frame[column].dtype, pd.CategoricalDtype)
        or pd.api.types.is_string_dtype(frame[column])
    ]
    for column in categorical:
        frame[column] = frame[column].fillna("__MISSING__").astype(str)
    return frame, categorical


def _catboost_model(
    iterations: int,
    *,
    class_weights_list: list[float] | None = None,
    eval_metric: str | None = None,
) -> CatBoostClassifier:
    parameters: dict[str, Any] = {
        "loss_function": "MultiClass",
        "iterations": iterations,
        "depth": CATBOOST_SELECTED["depth"],
        "learning_rate": CATBOOST_SELECTED["learning_rate"],
        "random_seed": SEED,
        "thread_count": 1,
        "verbose": False,
        "allow_writing_files": False,
        **CATBOOST_RESOURCE_PARAMETERS,
    }
    if class_weights_list is not None:
        parameters["class_weights"] = class_weights_list
    if eval_metric:
        parameters["eval_metric"] = eval_metric
    return CatBoostClassifier(**parameters)


def _catboost_fit(
    X_fit: pd.DataFrame,
    y_fit: pd.Series | np.ndarray,
    X_eval: pd.DataFrame | None = None,
    y_eval: pd.Series | np.ndarray | None = None,
    *,
    iterations: int,
    class_weights_list: list[float] | None = None,
    early_stopping: bool = False,
) -> tuple[CatBoostClassifier, dict[str, Any]]:
    fit_frame, categorical = _cat_frame(X_fit)
    model = _catboost_model(
        iterations,
        class_weights_list=class_weights_list,
        eval_metric=(
            (
                "TotalF1:average=Macro"
                if class_weights_list is None
                else "TotalF1:average=Macro;use_weights=false"
            )
            if early_stopping
            else None
        ),
    )
    started = time.perf_counter()
    fit_args: dict[str, Any] = {
        "X": fit_frame,
        "y": np.asarray(y_fit, dtype=np.int64),
        "cat_features": categorical,
        "verbose": False,
    }
    if X_eval is not None and y_eval is not None:
        eval_frame, eval_categorical = _cat_frame(X_eval)
        if categorical != eval_categorical:
            raise ValueError("As colunas categóricas CatBoost diferem entre partições.")
        fit_args["eval_set"] = (eval_frame, np.asarray(y_eval, dtype=np.int64))
    if early_stopping:
        fit_args["early_stopping_rounds"] = CATBOOST_EARLY_STOPPING
        fit_args["use_best_model"] = True
    model.fit(**fit_args)
    seconds = time.perf_counter() - started
    detail: dict[str, Any] = {"seconds": seconds}
    if early_stopping:
        detail["best_iteration"] = max(int(model.get_best_iteration()) + 1, 1)
        detail["evals_result"] = model.get_evals_result()
        detail["best_score"] = model.get_best_score()
    return model, detail


def _catboost_probabilities(
    model: CatBoostClassifier, X: pd.DataFrame
) -> np.ndarray:
    frame, categorical = _cat_frame(X)
    categorical_indices = [frame.columns.get_loc(column) for column in categorical]
    if categorical_indices != model.get_cat_feature_indices():
        raise ValueError("A tipagem das categorias CatBoost mudou entre fit e predict.")
    return align_probabilities(model.predict_proba(frame), np.asarray(model.classes_))


def _soft_class_weights(y: Any) -> list[float]:
    labels = np.asarray(y, dtype=np.int64)
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    if np.any(counts == 0):
        raise ValueError(
            "Não é possível calcular pesos: classes sem observações no treino "
            f"{np.flatnonzero(counts == 0).tolist()}."
        )
    weights = 1.0 / np.sqrt(counts.astype(np.float64))
    weights /= weights.mean()
    return weights.tolist()


def _compare_to_submissions(
    data: dict[str, Any], predictions: np.ndarray
) -> dict[str, int]:
    reference = _official_row_reference(data)
    if len(predictions) != len(reference):
        raise ValueError("Previsões não correspondem aos RowIds oficiais.")
    expected_names = [
        "submission_016_xgboost_mlp015.csv",
        "submission_019_ensemble_pesos.csv",
        "submission_025_xgboost_mlp_dropout.csv",
        "submission_028_ensemble_features_dominio.csv",
        "submission_029_catboost.csv",
    ]
    differences: dict[str, int] = {}
    new = pd.DataFrame({"RowId": reference.RowId, "AGE_RANGE": predictions})
    for name in expected_names:
        path = ROOT / "submissions" / name
        if not path.is_file():
            continue
        previous = pd.read_csv(path)
        if previous.columns.tolist() != ["RowId", "AGE_RANGE"]:
            raise ValueError(f"Cabeçalho inválido em {path}.")
        if previous.RowId.isna().any() or not previous.RowId.is_unique:
            raise ValueError(f"RowIds ausentes/duplicados em {path}.")
        if not pd.api.types.is_integer_dtype(previous.AGE_RANGE) or not previous.AGE_RANGE.between(0, 15).all():
            raise ValueError(f"Classes inválidas em {path}.")
        aligned = previous.set_index("RowId").reindex(new.RowId)
        if aligned.AGE_RANGE.isna().any():
            raise ValueError(f"RowIds incompatíveis em {path}.")
        differences[name] = int(
            (new.AGE_RANGE.to_numpy() != aligned.AGE_RANGE.to_numpy()).sum()
        )
    return differences


def _maybe_export(
    data: dict[str, Any],
    predictions: np.ndarray,
    filename: str,
) -> dict[str, Any]:
    differences = _compare_to_submissions(data, predictions)
    duplicate = [name for name, count in differences.items() if count == 0]
    if duplicate:
        return {
            "path": None,
            "duplicate_of": duplicate,
            "differences": differences,
            "reason": "As previsões reproduzem uma submissão anterior.",
        }
    path = ROOT / "submissions" / filename
    if path.exists():
        existing = pd.read_csv(path)
        reference = _official_row_reference(data)
        if (
            existing.columns.tolist() != ["RowId", "AGE_RANGE"]
            or not existing.RowId.equals(reference.RowId)
            or not np.array_equal(existing.AGE_RANGE.to_numpy(), predictions)
        ):
            raise FileExistsError(
                f"Já existe uma submissão diferente em {path}; não será substituída."
            )
        return {
            "path": str(path.relative_to(ROOT)),
            "duplicate_of": [],
            "differences": differences,
            "reason": "CSV anterior desta execução validado sem substituição.",
        }
    path = _export_predictions(data, predictions, filename)
    reread = pd.read_csv(path)
    reference = _official_row_reference(data)
    if reread.shape != (3655, 2):
        raise ValueError(f"Dimensão inválida na releitura: {reread.shape}.")
    if reread.columns.tolist() != ["RowId", "AGE_RANGE"] or not reread.RowId.equals(reference.RowId):
        raise ValueError("Cabeçalho/RowIds diferem da referência oficial.")
    if not pd.api.types.is_integer_dtype(reread.AGE_RANGE) or not reread.AGE_RANGE.between(0, 15).all():
        raise ValueError("AGE_RANGE exportada não contém classes inteiras 0–15.")
    return {
        "path": str(path.relative_to(ROOT)),
        "duplicate_of": [],
        "differences": differences,
        "reason": None,
    }


def _raw_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(ROOT / "notebooks" / "dataset_train.csv", low_memory=False),
        pd.read_csv(ROOT / "notebooks" / "dataset_test_X.csv", low_memory=False),
    )


def _selected_epoch_for_split(
    X_fit: pd.DataFrame, y_fit: pd.Series, seed: int = SEED
) -> tuple[int, dict[str, list[float]], float]:
    from sklearn.model_selection import train_test_split

    X_inner_fit, X_monitor, y_inner_fit, y_monitor = train_test_split(
        X_fit,
        y_fit,
        test_size=0.15,
        random_state=seed,
        stratify=y_fit,
    )
    _, X_inner_array, X_monitor_array = fit_preprocessor(X_inner_fit, X_monitor)
    _, history, seconds = _fit_mlp(
        X_inner_array,
        y_inner_fit.to_numpy(dtype=np.int64),
        MAX_EPOCHS,
        0.0001,
        weighting="inverse_sqrt",
        dropout=0.1,
        seed=seed,
        X_monitor=X_monitor_array,
        y_monitor=y_monitor.to_numpy(dtype=np.int64),
    )
    epoch, _ = _best_epoch(history)
    return epoch, history, seconds


def _fit_ensemble_components(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_eval: pd.DataFrame,
    *,
    mlp_epochs: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray], float, dict[str, Any]]:
    started = time.perf_counter()
    xgb = make_xgb().fit(X_fit, y_fit)
    xgb_eval = align_probabilities(
        xgb.predict_proba(X_eval), xgb.named_steps["model"].classes_
    )
    xgb_train = align_probabilities(
        xgb.predict_proba(X_fit), xgb.named_steps["model"].classes_
    )
    _, X_fit_array, X_eval_array = fit_preprocessor(X_fit, X_eval)
    mlp, mlp_history, mlp_seconds = _fit_mlp(
        X_fit_array,
        y_fit.to_numpy(dtype=np.int64),
        mlp_epochs,
        0.0001,
        weighting="inverse_sqrt",
        dropout=0.1,
        seed=SEED,
    )
    mlp_eval = _predict_probabilities(mlp, X_eval_array)
    mlp_train = _predict_probabilities(mlp, X_fit_array)
    return (
        {"xgb": xgb, "mlp": mlp, "history": mlp_history},
        {
            "ensemble_eval": 0.75 * xgb_eval + 0.25 * mlp_eval,
            "ensemble_train": 0.75 * xgb_train + 0.25 * mlp_train,
        },
        time.perf_counter() - started,
        {"mlp_training_seconds": mlp_seconds},
    )


def run_030() -> dict[str, Any]:
    experiment = "030"
    config = {
        "variants": VARIANTS_030,
        "folds": 3,
        "split_seed": SEED,
        "xgboost": {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1, "n_jobs": 1},
        "mlp": {"hidden": [64, 32], "dropout": 0.1, "learning_rate": 0.0001, "batch": 128, "weighting": "inverse_sqrt", "max_epochs": MAX_EPOCHS},
        "blend": [0.75, 0.25],
        "aggregation_source": "mapeamentos/formulas documentados no notebook 028; fonte executável do runner original ausente",
    }
    context = _checkpoint_context(experiment, config)
    checkpoint = _load_checkpoint(context)
    stored = checkpoint["experiments"].setdefault(experiment, {"folds": {}, "final": None})
    if stored.get("complete") and stored.get("result"):
        print("030: resultados/checkpoints reutilizados após validação de dados, preparação, configuração e versões.")
        return stored["result"]

    data = load_experiment_data()
    X_base = data["X_train"]
    y_base = data["y_train"]
    raw_train, _ = _raw_frames()
    if len(X_base) != 8769:
        raise ValueError(f"Esperadas 8.769 linhas de treino; recebidas {len(X_base)}.")
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    folds = [
        (train_indices, validation_indices)
        for train_indices, validation_indices in splitter.split(X_base, y_base)
    ]
    if any(
        set(y_base.iloc[train_indices].unique()) != set(CLASS_IDS)
        or set(y_base.iloc[validation_indices].unique()) != set(CLASS_IDS)
        for train_indices, validation_indices in folds
    ):
        raise ValueError("Cada fold precisa conter todas as 16 classes.")

    summaries: dict[str, list[dict[str, Any]]] = {name: [] for name in VARIANTS_030}
    histories: dict[str, dict[str, dict[str, list[float]]]] = {
        name: {} for name in VARIANTS_030
    }
    all_fold_indices: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for fold_number, (train_indices, validation_indices) in enumerate(folds, start=1):
        all_fold_indices[fold_number] = (train_indices, validation_indices)
        X_fold_train = X_base.iloc[train_indices]
        X_fold_validation = X_base.iloc[validation_indices]
        y_fold_train = y_base.iloc[train_indices]
        y_fold_validation = y_base.iloc[validation_indices]
        for name, groups in VARIANTS_030.items():
            key = f"{name}:{fold_number}"
            if key in stored["folds"]:
                row = stored["folds"][key]
                summaries[name].append(row["metrics"])
                histories[name][str(fold_number)] = row["history"]
                continue
            variant_X, variant_y = _prepared_variant(raw_train, X_base.index, groups)
            X_outer_train = variant_X.iloc[train_indices]
            X_outer_validation = variant_X.iloc[validation_indices]
            y_outer_train = variant_y.iloc[train_indices]
            y_outer_validation = variant_y.iloc[validation_indices]
            if not y_outer_train.equals(y_fold_train) or not y_outer_validation.equals(y_fold_validation):
                raise ValueError("A preparação dos agregados desalinhou X e AGE_RANGE.")

            model_started = time.perf_counter()
            xgb = make_xgb().fit(X_outer_train, y_outer_train)
            xgb_validation = align_probabilities(
                xgb.predict_proba(X_outer_validation),
                xgb.named_steps["model"].classes_,
            )
            xgb_train = align_probabilities(
                xgb.predict_proba(X_outer_train), xgb.named_steps["model"].classes_
            )

            epoch, history, mlp_selection_seconds = _selected_epoch_for_split(
                X_outer_train, y_outer_train
            )
            _, X_mlp_train, X_mlp_validation = fit_preprocessor(
                X_outer_train, X_outer_validation
            )
            mlp, _, mlp_refit_seconds = _fit_mlp(
                X_mlp_train,
                y_outer_train.to_numpy(dtype=np.int64),
                epoch,
                0.0001,
                weighting="inverse_sqrt",
                dropout=0.1,
                seed=SEED,
            )
            mlp_validation = _predict_probabilities(mlp, X_mlp_validation)
            mlp_train = _predict_probabilities(mlp, X_mlp_train)
            validation_probabilities = 0.75 * xgb_validation + 0.25 * mlp_validation
            train_probabilities = 0.75 * xgb_train + 0.25 * mlp_train
            metrics = _fold_metrics(
                name,
                fold_number,
                _metrics(y_outer_train, train_probabilities),
                _metrics(y_outer_validation, validation_probabilities),
                epoch,
                time.perf_counter() - model_started,
            )
            history["selection_seconds"] = [mlp_selection_seconds]
            history["refit_seconds"] = [mlp_refit_seconds]
            stored["folds"][key] = {"metrics": metrics, "history": history}
            _save_checkpoint(checkpoint)
            summaries[name].append(metrics)
            histories[name][str(fold_number)] = history
            del variant_X, variant_y, xgb, mlp, X_outer_train, X_outer_validation
            del X_mlp_train, X_mlp_validation, train_probabilities, validation_probabilities
            gc.collect()

    baseline_scores = np.array(
        [row["validation"]["macro_f1"] for row in summaries["sem_agregados"]]
    )
    aggregate_rows = []
    for name, rows in summaries.items():
        scores = np.array([row["validation"]["macro_f1"] for row in rows])
        paired = scores - baseline_scores
        aggregate_rows.append(
            {
                "variant": name,
                "aggregate_count": len(VARIANTS_030[name]),
                "fold_macro_f1": scores.tolist(),
                "mean_macro_f1": float(scores.mean()),
                "std_macro_f1": float(scores.std(ddof=1)),
                "paired_differences_vs_control": paired.tolist(),
                "mean_paired_difference": float(paired.mean()),
                "std_paired_difference": float(paired.std(ddof=1)),
                "folds": rows,
            }
        )
    selected_row = sorted(
        aggregate_rows,
        key=lambda row: (-row["mean_macro_f1"], row["aggregate_count"]),
    )[0]
    selected_variant = selected_row["variant"]
    selected_epoch = int(
        np.median([row["epoch"] for row in summaries[selected_variant]])
    )

    fixed_X, fixed_y = _prepared_variant(raw_train, data["X_train"].index.append(data["X_val"].index), VARIANTS_030[selected_variant])
    X_train = fixed_X.loc[data["X_train"].index]
    X_validation = fixed_X.loc[data["X_val"].index]
    y_train = fixed_y.loc[data["y_train"].index]
    y_validation = fixed_y.loc[data["y_val"].index]
    started = time.perf_counter()
    xgb_fixed = make_xgb().fit(X_train, y_train)
    xgb_val_prob = align_probabilities(
        xgb_fixed.predict_proba(X_validation), xgb_fixed.named_steps["model"].classes_
    )
    xgb_train_prob = align_probabilities(
        xgb_fixed.predict_proba(X_train), xgb_fixed.named_steps["model"].classes_
    )
    _, X_train_arr, X_validation_arr = fit_preprocessor(X_train, X_validation)
    mlp_fixed, fixed_history, mlp_seconds = _fit_mlp(
        X_train_arr,
        y_train.to_numpy(dtype=np.int64),
        selected_epoch,
        0.0001,
        weighting="inverse_sqrt",
        dropout=0.1,
        seed=SEED,
    )
    val_prob = 0.75 * xgb_val_prob + 0.25 * _predict_probabilities(mlp_fixed, X_validation_arr)
    train_prob = 0.75 * xgb_train_prob + 0.25 * _predict_probabilities(mlp_fixed, X_train_arr)
    fixed = {
        "train": _metrics(y_train, train_prob),
        "validation": _metrics(y_validation, val_prob),
        "epoch": selected_epoch,
        "seconds": time.perf_counter() - started,
        "mlp_training_seconds": mlp_seconds,
    }
    _plot_confusion(
        y_validation,
        val_prob.argmax(axis=1),
        ROOT / "reports" / "experiencia_030_matriz_confusao.png",
        f"030 — {selected_variant} — validação fixa",
    )
    curves_path = _plot_030_histories(histories, selected_variant)
    result = {
        "experiment": experiment,
        "configuration": config,
        "seed": SEED,
        "folds_train_only": True,
        "fold_results": aggregate_rows,
        "selected_variant": selected_variant,
        "selected_epoch_for_full_fit": selected_epoch,
        "fixed_validation_once": fixed,
        "control_reference": {
            "025_fixed_macro_f1": 0.4614702838167389,
            "028_fixed_macro_f1": 0.4614702838167389,
            "control_notebook_execution_source_available": False,
        },
        "control_reconstruction_note": (
            "O notebook/runner executável original 028 não está presente. A configuração "
            "foi reconstruída a partir dos resultados guardados e do controlo 025: XGBoost "
            "007 + MLP 020 dropout 0,1, 75/25, pesos inverse_sqrt. Os agregados usam as "
            "fórmulas documentadas no notebook 028 e add_domain_aggregates; a fonte original "
            "dos mapeamentos não está presente. Os rótulos positivos/negativos e as respostas "
            "especiais estão declarados em config, não inferidos por treino."
        ),
        "training_seconds": float(sum(row["seconds"] for rows in summaries.values() for row in rows)),
        "fixed_training_seconds": fixed["seconds"],
        "curves": str(curves_path.relative_to(ROOT)),
        "confusion_matrix": "reports/experiencia_030_matriz_confusao.png",
        "kaggle_score": None,
    }
    if selected_variant == "sem_agregados":
        result["submission"] = {
            "path": None,
            "reason": "O controlo venceu; não foi criado um CSV repetido.",
            "differences": {},
        }
    else:
        raw_test = pd.read_csv(
            ROOT / "notebooks" / "dataset_test_X.csv", low_memory=False
        )
        full_prepared = preparar_dados(
            raw_train,
            raw_test,
            domain_aggregates=_aggregate_mappings(VARIANTS_030[selected_variant]),
        )
        full_X = full_prepared["X_train"]
        full_y = full_prepared["y_train"]
        full_test_X = full_prepared["X_test"]
        xgb_full = make_xgb().fit(full_X, full_y)
        xgb_test = align_probabilities(
            xgb_full.predict_proba(full_test_X), xgb_full.named_steps["model"].classes_
        )
        _, X_full_arr, X_test_arr = fit_preprocessor(full_X, full_test_X)
        mlp_full, _, export_seconds = _fit_mlp(
            X_full_arr,
            full_y.to_numpy(dtype=np.int64),
            selected_epoch,
            0.0001,
            weighting="inverse_sqrt",
            dropout=0.1,
            seed=SEED,
        )
        test_prob = 0.75 * xgb_test + 0.25 * _predict_probabilities(mlp_full, X_test_arr)
        result["export_training_seconds"] = export_seconds
        result["submission"] = _maybe_export(
            data,
            test_prob.argmax(axis=1),
            "submission_030_agregados_separados.csv",
        )
    _record_result(experiment, result)
    stored["result"] = result
    stored["complete"] = True
    _save_checkpoint(checkpoint)
    return result


def _select_catboost(
    X_fit: pd.DataFrame, y_fit: pd.Series, X_validation: pd.DataFrame, y_validation: pd.Series,
    weights: list[float] | None,
) -> tuple[CatBoostClassifier, dict[str, Any], float]:
    model, detail = _catboost_fit(
        X_fit,
        y_fit,
        X_validation,
        y_validation,
        iterations=CATBOOST_MAX_ITERATIONS,
        class_weights_list=weights,
        early_stopping=True,
    )
    probabilities = _catboost_probabilities(model, X_validation)
    sklearn_macro_f1 = _score(y_validation, probabilities)
    validation_scores = detail["best_score"].get("validation", {})
    native_scores = next(
        (
            score
            for name, score in validation_scores.items()
            if "TotalF1" in name and "Macro" in name
        ),
        None,
    )
    detail["sklearn_macro_f1"] = sklearn_macro_f1
    detail["catboost_best_macro_f1"] = native_scores
    detail["selected_iterations"] = detail["best_iteration"]
    detail["internal_metrics"] = _metrics(y_validation, probabilities)
    return model, detail, sklearn_macro_f1


def run_031() -> dict[str, Any]:
    experiment = "031"
    config = {
        "selected_029": CATBOOST_SELECTED,
        "resource_parameters": CATBOOST_RESOURCE_PARAMETERS,
        "weights": ["none", "inverse_sqrt_normalized_mean_1"],
        "max_iterations": CATBOOST_MAX_ITERATIONS,
        "early_stopping_rounds": CATBOOST_EARLY_STOPPING,
        "selection_metric": "sklearn f1_score(labels=0..15, average=macro, unweighted)",
        "seed": SEED,
    }
    context = _checkpoint_context(experiment, config)
    checkpoint = _load_checkpoint(context)
    stored = checkpoint["experiments"].setdefault(experiment, {"internal": {}, "result": None})
    if stored.get("complete") and stored.get("result"):
        print("031: resultados/checkpoints reutilizados após validação de dados, preparação, configuração e versões.")
        return stored["result"]

    data = load_experiment_data()
    X_fit, y_fit = data["X_inner_train"], data["y_inner_train"]
    X_internal_val, y_internal_val = data["X_inner_val"], data["y_inner_val"]
    candidates: list[dict[str, Any]] = []
    internal_histories: dict[str, Any] = {}
    internal_models: dict[str, CatBoostClassifier] = {}
    for label, weights in (
        ("sem_pesos", None),
        ("inverse_sqrt", _soft_class_weights(y_fit)),
    ):
        if label in stored["internal"]:
            detail = stored["internal"][label]
            candidates.append(detail["candidate"])
            internal_histories[label] = detail["evals_result"]
            continue
        model, detail, sklearn_macro_f1 = _select_catboost(
            X_fit, y_fit, X_internal_val, y_internal_val, weights
        )
        candidate = {
            "variant": label,
            "class_weights": weights,
            "selected_iterations": detail["selected_iterations"],
            "catboost_best_macro_f1": detail["catboost_best_macro_f1"],
            "sklearn_macro_f1": sklearn_macro_f1,
            "internal_metrics": detail["internal_metrics"],
            "selection_seconds": detail["seconds"],
        }
        stored["internal"][label] = {
            "candidate": candidate,
            "evals_result": detail["evals_result"],
        }
        _save_checkpoint(checkpoint)
        candidates.append(candidate)
        internal_histories[label] = detail["evals_result"]
        internal_models[label] = model
    selected = max(candidates, key=lambda row: row["sklearn_macro_f1"])

    fixed_metrics: dict[str, Any] = {}
    full_models: dict[str, CatBoostClassifier] = {}
    full_weights: dict[str, list[float] | None] = {}
    started = time.perf_counter()
    for candidate in candidates:
        label = candidate["variant"]
        weights = (
            None
            if label == "sem_pesos"
            else _soft_class_weights(data["y_train"])
        )
        model, detail = _catboost_fit(
            data["X_train"],
            data["y_train"],
            iterations=candidate["selected_iterations"],
            class_weights_list=weights,
        )
        train_prob = _catboost_probabilities(model, data["X_train"])
        validation_prob = _catboost_probabilities(model, data["X_val"])
        fixed_metrics[label] = {
            "train": _metrics(data["y_train"], train_prob),
            "validation": _metrics(data["y_val"], validation_prob),
            "iterations": candidate["selected_iterations"],
            "fit_seconds": detail["seconds"],
        }
        full_models[label] = model
        full_weights[label] = weights
        if label not in stored.get("fixed", {}):
            stored.setdefault("fixed", {})[label] = fixed_metrics[label]
            _save_checkpoint(checkpoint)

    selected_label = selected["variant"]
    validation_predictions = _catboost_probabilities(
        full_models[selected_label], data["X_val"]
    ).argmax(axis=1)
    confusion_path = ROOT / "reports" / "experiencia_031_matriz_confusao.png"
    _plot_confusion(
        data["y_val"],
        validation_predictions,
        confusion_path,
        f"031 — {selected_label} — validação fixa",
    )
    curves_path = _plot_031_curves(internal_histories)

    raw_train, _ = _raw_frames()
    raw_test = pd.read_csv(
        ROOT / "notebooks" / "dataset_test_X.csv", low_memory=False
    )
    full_prepared = preparar_dados(raw_train, raw_test)
    weights_full = (
        None
        if selected_label == "sem_pesos"
        else _soft_class_weights(full_prepared["y_train"])
    )
    export_model, export_detail = _catboost_fit(
        full_prepared["X_train"],
        full_prepared["y_train"],
        iterations=selected["selected_iterations"],
        class_weights_list=weights_full,
    )
    test_prob = _catboost_probabilities(export_model, full_prepared["X_test"])
    submission = _maybe_export(
        data,
        test_prob.argmax(axis=1),
        "submission_031_catboost_ponderacao_suave.csv",
    )
    result = {
        "experiment": experiment,
        "configuration": config,
        "candidates": candidates,
        "selected_internally": selected,
        "fixed_metrics_both_variants": fixed_metrics,
        "fixed_validation_selection_used": False,
        "key_class_comparison": {
            str(class_id): {
                variant: next(
                    row for row in fixed_metrics[variant]["validation"]["per_class"]
                    if row["classe"] == class_id
                )
                for variant in fixed_metrics
            }
            for class_id in (1, 6, 13)
        },
        "control_reconstruction_note": (
            "Configuração selecionada recuperada da saída guardada do notebook 029 "
            "(depth=4, learning_rate=0,1, 262 iterações); restrições de recursos "
            "recuperadas do Markdown do 029. O runner executável original não está presente."
        ),
        "resource_note": (
            "O parâmetro CatBoost used_ram_limit=512mb limita o uso indicativo de CTR, "
            "não a memória total do processo; o CatBoost avisou que o consumo de CPU "
            "RAM observado ultrapassou esse valor. Nenhum parâmetro foi reduzido."
        ),
        "control_reference": {
            "029_fixed_macro_f1": 0.448318,
            "029_kaggle_score": None,
        },
        "training_seconds": float(
            sum(row["selection_seconds"] for row in candidates)
            + sum(row["fit_seconds"] for row in fixed_metrics.values())
            + export_detail["seconds"]
        ),
        "curves": str(curves_path.relative_to(ROOT)),
        "confusion_matrix": str(confusion_path.relative_to(ROOT)),
        "submission": submission,
        "kaggle_score": None,
    }
    _record_result(experiment, result)
    stored["result"] = result
    stored["complete"] = True
    _save_checkpoint(checkpoint)
    return result


def run_032() -> dict[str, Any]:
    experiment = "032"
    config = {
        "ensemble_025": {
            "xgboost": {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1},
            "mlp": {"dropout": 0.1, "weighting": "inverse_sqrt", "epochs": 85, "learning_rate": 0.0001, "batch_size": 128},
            "ensemble_weights": [0.75, 0.25],
        },
        "catboost_029": {
            **CATBOOST_SELECTED,
            "class_weights": None,
            "resource_parameters": CATBOOST_RESOURCE_PARAMETERS,
        },
        "blend_weights": [list(weights) for weights in BLEND_WEIGHTS_032],
        "seed": SEED,
    }
    context = _checkpoint_context(experiment, config)
    checkpoint = _load_checkpoint(context)
    stored = checkpoint["experiments"].setdefault(experiment, {"internal": None, "result": None})
    if stored.get("complete") and stored.get("result"):
        print("032: resultados/checkpoints reutilizados após validação de dados, preparação, configuração e versões.")
        return stored["result"]

    data = load_experiment_data()
    if stored["internal"] is None:
        ensemble_components, ensemble_internal, ensemble_seconds, ensemble_timing = _fit_ensemble_components(
            data["X_inner_train"],
            data["y_inner_train"],
            data["X_inner_val"],
            mlp_epochs=85,
        )
        cat_internal_model, cat_internal_timing = _catboost_fit(
            data["X_inner_train"],
            data["y_inner_train"],
            iterations=CATBOOST_SELECTED["iterations"],
        )
        cat_internal = _catboost_probabilities(cat_internal_model, data["X_inner_val"])
        y_internal = data["y_inner_val"].to_numpy(dtype=np.int64)
        candidates: list[dict[str, Any]] = []
        for ensemble_weight, catboost_weight in BLEND_WEIGHTS_032:
            probabilities = (
                ensemble_weight * ensemble_internal["ensemble_eval"]
                + catboost_weight * cat_internal
            )
            weight_label = f"{ensemble_weight:.0%}/{catboost_weight:.0%}"
            candidates.append(
                {
                    "ensemble_weight": ensemble_weight,
                    "catboost_weight": catboost_weight,
                    "weight_label": weight_label,
                    "internal_metrics": _metrics(y_internal, probabilities),
                }
            )
        selected = max(candidates, key=lambda row: row["internal_metrics"]["macro_f1"])
        stored["internal"] = {
            "candidates": candidates,
            "selected": selected,
            "ensemble_training_seconds": ensemble_seconds,
            "ensemble_component_timing": ensemble_timing,
            "catboost_training_seconds": cat_internal_timing["seconds"],
            "mlp_history": ensemble_components["history"],
        }
        _save_checkpoint(checkpoint)
        del ensemble_components, cat_internal_model, cat_internal
        gc.collect()
    internal = stored["internal"]
    candidates = internal["candidates"]
    selected = internal["selected"]

    ensemble_models, ensemble_fixed, ensemble_seconds, ensemble_timing = _fit_ensemble_components(
        data["X_train"],
        data["y_train"],
        data["X_val"],
        mlp_epochs=85,
    )
    cat_fixed_model, cat_fixed_timing = _catboost_fit(
        data["X_train"],
        data["y_train"],
        iterations=CATBOOST_SELECTED["iterations"],
    )
    cat_fixed_val = _catboost_probabilities(cat_fixed_model, data["X_val"])
    cat_fixed_train = _catboost_probabilities(cat_fixed_model, data["X_train"])
    selected_val_prob = (
        selected["ensemble_weight"] * ensemble_fixed["ensemble_eval"]
        + selected["catboost_weight"] * cat_fixed_val
    )
    selected_train_prob = (
        selected["ensemble_weight"] * ensemble_fixed["ensemble_train"]
        + selected["catboost_weight"] * cat_fixed_train
    )
    fixed_metrics = {
        "train": _metrics(data["y_train"], selected_train_prob),
        "validation": _metrics(data["y_val"], selected_val_prob),
    }
    confusion_path = ROOT / "reports" / "experiencia_032_matriz_confusao.png"
    _plot_confusion(
        data["y_val"],
        selected_val_prob.argmax(axis=1),
        confusion_path,
        f"032 — {selected['weight_label']} ensemble/CatBoost — validação fixa",
    )
    curves_path = _plot_032_candidates(candidates, internal["mlp_history"])

    raw_train, _ = _raw_frames()
    raw_test = pd.read_csv(
        ROOT / "notebooks" / "dataset_test_X.csv", low_memory=False
    )
    full_prepared = preparar_dados(raw_train, raw_test)
    xgb_full = make_xgb().fit(full_prepared["X_train"], full_prepared["y_train"])
    xgb_test = align_probabilities(
        xgb_full.predict_proba(full_prepared["X_test"]),
        xgb_full.named_steps["model"].classes_,
    )
    _, X_all_arr, X_test_arr = fit_preprocessor(
        full_prepared["X_train"], full_prepared["X_test"]
    )
    mlp_full, _, mlp_export_seconds = _fit_mlp(
        X_all_arr,
        full_prepared["y_train"].to_numpy(dtype=np.int64),
        85,
        0.0001,
        weighting="inverse_sqrt",
        dropout=0.1,
        seed=SEED,
    )
    ensemble_test = 0.75 * xgb_test + 0.25 * _predict_probabilities(mlp_full, X_test_arr)
    cat_full_model, cat_export_timing = _catboost_fit(
        full_prepared["X_train"],
        full_prepared["y_train"],
        iterations=CATBOOST_SELECTED["iterations"],
    )
    cat_test = _catboost_probabilities(cat_full_model, full_prepared["X_test"])
    test_prob = (
        selected["ensemble_weight"] * ensemble_test
        + selected["catboost_weight"] * cat_test
    )
    submission = _maybe_export(
        data,
        test_prob.argmax(axis=1),
        "submission_032_ensemble_catboost.csv",
    )
    result = {
        "experiment": experiment,
        "configuration": config,
        "internal_candidates": candidates,
        "selected_internally": selected,
        "fixed_metrics_selected_blend": fixed_metrics,
        "fixed_validation_selection_used": False,
        "component_validation_metrics": {
            "ensemble_025": _metrics(data["y_val"], ensemble_fixed["ensemble_eval"]),
            "catboost_029": _metrics(data["y_val"], cat_fixed_val),
        },
        "control_reconstruction_note": (
            "Os componentes são reproduzidos com as configurações documentadas de 025 "
            "(85 épocas) e 029 (262 iterações, sem pesos). Não foi usada validação "
            "antecipada sobre a parcela que avalia a mistura: as épocas/iterações são fixas."
        ),
        "resource_note": (
            "O parâmetro CatBoost used_ram_limit=512mb não limita a memória total; "
            "não se alteraram as restrições de recursos documentadas no 029."
        ),
        "training_seconds": {
            "internal_ensemble": internal["ensemble_training_seconds"],
            "internal_catboost": internal["catboost_training_seconds"],
            "fixed_ensemble": ensemble_seconds,
            "fixed_catboost": cat_fixed_timing["seconds"],
            "export_mlp": mlp_export_seconds,
            "export_catboost": cat_export_timing["seconds"],
        },
        "curves": str(curves_path.relative_to(ROOT)),
        "confusion_matrix": str(confusion_path.relative_to(ROOT)),
        "submission": submission,
        "kaggle_score": None,
    }
    _record_result(experiment, result)
    stored["result"] = result
    stored["complete"] = True
    _save_checkpoint(checkpoint)
    return result


def write_final_report() -> Path:
    stored = json.loads(RESULTS_PATH.read_text(encoding="utf-8")) if RESULTS_PATH.is_file() else {}
    lines = [
        "# Experiências DAA 030–032",
        "",
        "A validação fixa é uma comparação de desenvolvimento, não um teste final independente. "
        "A pontuação Kaggle não foi usada para seleção e permanece em branco.",
        "",
        "As execuções foram sequenciais e com uma thread. Mantiveram-se todos os parâmetros "
        "CatBoost do 029; `used_ram_limit=512mb` é indicativo para CTRs, não um limite da "
        "memória total. O CatBoost avisou de consumo até 685 MiB, sem exigir redução de parâmetros.",
        "",
        "| Experiência | Macro F1 fixa | Accuracy fixa | Seleção interna | CSV | Recomendação |",
        "|---|---:|---:|---|---|---|",
    ]
    for experiment in ("030", "031", "032"):
        result = stored.get(experiment)
        if not result:
            lines.append(f"| {experiment} | — | — | Ainda não executada | — | — |")
            continue
        if experiment == "030":
            fixed = result["fixed_validation_once"]["validation"]
            selection = result["selected_variant"]
            baseline = result["control_reference"]["028_fixed_macro_f1"]
            recommendation = (
                "controlo CV venceu; sem CSV"
                if selection == "sem_agregados"
                else (
                    "exploratório"
                    if fixed["macro_f1"] <= baseline
                    else "candidato distinto"
                )
            )
        elif experiment == "031":
            selected = result["selected_internally"]
            fixed = result["fixed_metrics_both_variants"][selected["variant"]]["validation"]
            selection = selected["variant"]
            baseline = 0.4614702838167389
            recommendation = (
                "exploratório: abaixo do melhor controlo 025/028"
                if fixed["macro_f1"] <= baseline
                else "candidato distinto acima do melhor controlo"
            )
        else:
            fixed = result["fixed_metrics_selected_blend"]["validation"]
            selection = result["selected_internally"]["weight_label"]
            baseline = 0.4614702838167389
            recommendation = (
                "duplicado de 025; sem CSV"
                if result.get("submission", {}).get("duplicate_of")
                else (
                    "exploratório"
                    if fixed["macro_f1"] <= baseline
                    else "candidato distinto acima do controlo"
                )
            )
        submission = result.get("submission", {})
        csv = submission.get("path") or "—"
        lines.append(
            f"| {experiment} | {fixed['macro_f1']:.4f} | {fixed['accuracy']:.4f} | "
            f"{selection} | {csv} | {recommendation} |"
        )
    lines.extend(["", "## Conclusões e melhorias", ""])
    for experiment in ("030", "031", "032"):
        result = stored.get(experiment)
        if not result:
            continue
        if experiment == "030":
            rows = result["fold_results"]
            best = max(rows, key=lambda row: row["mean_macro_f1"])
            lines.append(
                f"- **030:** {best['variant']} obteve Macro F1 CV média "
                f"{best['mean_macro_f1']:.4f} (DP {best['std_macro_f1']:.4f}); "
                f"diferença emparelhada média {best['mean_paired_difference']:+.4f} "
                f"face ao controlo. A avaliação fixa usou {result['selected_epoch_for_full_fit']} "
                "épocas (mediana dos folds), enquanto o 028 guardado usou 85; a Macro F1 "
                "fixa não é diretamente comparável ao valor histórico. Melhorias: confirmar "
                "a estabilidade em novas divisões sem usar a validação fixa como teste final."
            )
        elif experiment == "031":
            selected = result["selected_internally"]
            fixed = result["fixed_metrics_both_variants"]
            one = fixed["sem_pesos"]["validation"]["macro_f1"]
            two = fixed["inverse_sqrt"]["validation"]["macro_f1"]
            key_differences = []
            for class_id in (1, 6, 13):
                no_weight = next(
                    row for row in fixed["sem_pesos"]["validation"]["per_class"]
                    if row["classe"] == class_id
                )
                weighted = next(
                    row for row in fixed["inverse_sqrt"]["validation"]["per_class"]
                    if row["classe"] == class_id
                )
                key_differences.append(
                    f"classe {class_id}: ΔF1 {weighted['F1'] - no_weight['F1']:+.3f}, "
                    f"Δrecall {weighted['recall'] - no_weight['recall']:+.3f}"
                )
            lines.append(
                f"- **031:** {selected['variant']} foi selecionado internamente; "
                f"Macro F1 fixa sem pesos={one:.4f}, ponderada={two:.4f} "
                f"(Δ={two - one:+.4f}); " + "; ".join(key_differences) + ". "
                "O controlo 029 guardado escolheu 262 iterações/F1 interna 0,4718; "
                "a reprodução sem pesos desta execução escolheu 181/F1 0,4625. Como o "
                "runner original está ausente, o ganho absoluto face ao 029 histórico não "
                "é causalmente atribuível à ponderação; a comparação emparelhada dentro "
                "de 031 é a referência. Melhorias: confirmar sobretudo os custos nas classes 6 e 13."
            )
        else:
            selected = result["selected_internally"]
            fixed = result["fixed_metrics_selected_blend"]["validation"]["macro_f1"]
            lines.append(
                f"- **032:** mistura {selected['weight_label']} selecionada internamente; "
                f"Macro F1 fixa={fixed:.4f}. Melhorias: validar a estabilidade da mistura "
                "sem reajustar os pesos na validação fixa."
            )
    lines.extend(["", "Pontuação Kaggle: ", ""])
    path = ROOT / "reports" / "experiments_030_032.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
