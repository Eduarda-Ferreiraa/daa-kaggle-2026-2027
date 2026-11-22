"""Reproducible runners for DAA experiments 033–034."""

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
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

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
    to_dense_float32,
)
from src.experiments_022_027 import (
    _best_epoch,
    _fit_mlp,
    _predict_probabilities,
)
from src.experiments_030_032 import (
    CATBOOST_EARLY_STOPPING,
    CATBOOST_MAX_ITERATIONS,
    CATBOOST_RESOURCE_PARAMETERS,
    CATBOOST_SELECTED,
    SEED,
    _catboost_fit,
    _catboost_probabilities,
    _metrics,
    _raw_frames,
    _score,
)
from src.preparar_dados import preparar_dados


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = ROOT / "reports" / ".experiments_033_034_checkpoint.json"
RESULTS_PATH = ROOT / "reports" / ".experiments_033_034_results.json"
MAX_MLP_EPOCHS = 100
MLP_LEARNING_RATE = 0.0001
MLP_DROPOUT = 0.1
XGB_025 = {
    "objective": "multi:softprob",
    "num_class": NUM_CLASSES,
    "tree_method": "hist",
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.1,
    "random_state": SEED,
    "n_jobs": 1,
    "verbosity": 0,
}
BLEND_WEIGHTS = ((1.0, 0.0), (0.9, 0.1), (0.75, 0.25), (0.5, 0.5), (0.0, 1.0))
torch.set_num_threads(1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_versions() -> dict[str, str]:
    return {
        package: importlib.metadata.version(package)
        for package in ("numpy", "pandas", "scikit-learn", "torch", "xgboost", "catboost")
    }


def _context(experiment: str, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "config": config,
        "train_sha256": _sha256(ROOT / "notebooks" / "dataset_train.csv"),
        "test_sha256": _sha256(ROOT / "notebooks" / "dataset_test_X.csv"),
        "preparation_sha256": _sha256(ROOT / "src" / "preparar_dados.py"),
        "runner_sha256": _sha256(Path(__file__)),
        "reference_runner_sha256": _sha256(ROOT / "src" / "experiments_030_032.py"),
        "versions": _package_versions(),
    }


def _load_checkpoint(context: dict[str, Any]) -> dict[str, Any]:
    if not CHECKPOINT_PATH.is_file():
        return {"contexts": {context["experiment"]: context}, "experiments": {}}
    checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    experiment = context["experiment"]
    stored = checkpoint.setdefault("contexts", {}).get(experiment)
    if stored is not None and stored != context:
        raise ValueError(
            "Checkpoint incompatível com dados, preparação, código, parâmetros ou "
            "versões atuais; não será reutilizado."
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
    stored = (
        json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        if RESULTS_PATH.is_file()
        else {}
    )
    stored[experiment] = result
    temporary = RESULTS_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(stored, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, RESULTS_PATH)


def _validate_features(X: pd.DataFrame) -> None:
    forbidden = {"AGE_RANGE", "PID", "HHID", "PROXY", "COUNTRY"}
    leaked = sorted(forbidden.intersection(X.columns))
    if leaked:
        raise ValueError(f"Colunas proibidas nas entradas: {leaked}")


def _metrics_with_predictions(y: Any, probabilities: np.ndarray) -> dict[str, Any]:
    predictions = np.asarray(probabilities).argmax(axis=1)
    return _classification_metrics(np.asarray(y, dtype=np.int64), predictions)


def _weights_for_alpha(y: Any, alpha: float) -> list[float] | None:
    if alpha == 0:
        return None
    labels = np.asarray(y, dtype=np.int64)
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    if np.any(counts == 0):
        raise ValueError(
            f"Não é possível ponderar classes ausentes: {np.flatnonzero(counts == 0).tolist()}."
        )
    weights = counts.astype(np.float64) ** (-alpha)
    weights /= weights.mean()
    return weights.tolist()


def _catboost_candidate(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_monitor: pd.DataFrame,
    y_monitor: pd.Series,
    alpha: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    weights = _weights_for_alpha(y_fit, alpha)
    model, detail = _catboost_fit(
        X_fit,
        y_fit,
        X_monitor,
        y_monitor,
        iterations=CATBOOST_MAX_ITERATIONS,
        class_weights_list=weights,
        early_stopping=True,
    )
    probabilities = _catboost_probabilities(model, X_monitor)
    score = _score(y_monitor, probabilities)
    native_scores = detail["best_score"].get("validation", {})
    native_macro = next(
        (
            value
            for name, value in native_scores.items()
            if "TotalF1" in name and "Macro" in name
        ),
        None,
    )
    if native_macro is None or not np.isclose(native_macro, score, atol=1e-12):
        raise ValueError(
            f"Macro F1 CatBoost ({native_macro}) não coincide com sklearn ({score})."
        )
    result = {
        "alpha": alpha,
        "class_weights_from_fit_labels": weights,
        "selected_iterations": max(int(model.get_best_iteration()) + 1, 1),
        "sklearn_macro_f1": score,
        "catboost_macro_f1": float(native_macro),
        "internal_metrics": _metrics_with_predictions(y_monitor, probabilities),
        "fit_seconds": detail["seconds"],
        "evals_result": detail["evals_result"],
    }
    del model
    gc.collect()
    return result, detail


def _plot_033_curves(histories: dict[str, Any]) -> str:
    path = ROOT / "reports" / "experiencia_033_curvas.png"
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for alpha, history in histories.items():
        label = f"alpha={alpha}"
        for split, style in (("learn", "-"), ("validation", "--")):
            metrics = history.get(split, {})
            key = next(
                (name for name in metrics if "TotalF1" in name and "Macro" in name),
                None,
            )
            if key:
                axes[0].plot(
                    np.arange(1, len(metrics[key]) + 1),
                    metrics[key],
                    linestyle=style,
                    label=f"{label} {split}",
                )
        loss = history.get("learn", {}).get("MultiClass", [])
        if loss:
            axes[1].plot(np.arange(1, len(loss) + 1), loss, label=label)
    axes[0].set(title="Macro F1 interna não ponderada", xlabel="Iteração", ylabel="Macro F1")
    axes[1].set(title="Perda CatBoost no treino", xlabel="Iteração", ylabel="MultiClass")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return str(path.relative_to(ROOT))


def _plot_034_curves(
    selection_history: dict[str, Any], candidates: list[dict[str, Any]]
) -> str:
    path = ROOT / "reports" / "experiencia_034_curvas.png"
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = selection_history["epoch"]
    axes[0].plot(epochs, selection_history["train_macro_f1"], label="MLP treino")
    axes[0].plot(
        epochs, selection_history["validation_macro_f1"], label="MLP seleção"
    )
    axes[0].set(title="MLP 025 — escolha interna da época", xlabel="Época", ylabel="Macro F1")
    labels = [row["weight_label"] for row in candidates]
    scores = [row["internal_macro_f1"] for row in candidates]
    axes[1].plot(labels, scores, marker="o")
    axes[1].set(title="Macro F1 da mistura na partição própria", xlabel="Ensemble/CatBoost", ylabel="Macro F1")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return str(path.relative_to(ROOT))


def _plot_confusion(y: Any, probabilities: np.ndarray, experiment: str) -> str:
    path = ROOT / "reports" / f"experiencia_{experiment}_matriz_confusao.png"
    matrix = confusion_matrix(
        np.asarray(y, dtype=np.int64),
        np.asarray(probabilities).argmax(axis=1),
        labels=CLASS_IDS,
    )
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


def _compare_and_export(
    data: dict[str, Any], predictions: np.ndarray, filename: str
) -> dict[str, Any]:
    reference = _official_row_reference(data)
    labels = np.asarray(predictions, dtype=np.int64)
    if len(labels) != 3655 or not np.isin(labels, CLASS_IDS).all():
        raise ValueError("As previsões para Kaggle têm dimensão/classes inválidas.")
    differences: dict[str, int] = {}
    duplicates: list[str] = []
    for path in sorted((ROOT / "submissions").glob("submission_*.csv")):
        if path.name == filename:
            continue
        previous = pd.read_csv(path)
        if previous.columns.tolist() != ["RowId", "AGE_RANGE"]:
            raise ValueError(f"Esquema inválido na submissão anterior {path.name}.")
        if previous.RowId.isna().any() or not previous.RowId.is_unique:
            raise ValueError(f"RowIds inválidos na submissão anterior {path.name}.")
        if (
            not pd.api.types.is_integer_dtype(previous.AGE_RANGE)
            or not previous.AGE_RANGE.between(0, 15).all()
        ):
            raise ValueError(f"Classes inválidas na submissão anterior {path.name}.")
        aligned = previous.set_index("RowId").reindex(reference.RowId)
        if aligned.AGE_RANGE.isna().any():
            raise ValueError(f"RowIds incompatíveis na submissão anterior {path.name}.")
        count = int((labels != aligned.AGE_RANGE.to_numpy(dtype=np.int64)).sum())
        differences[path.name] = count
        if count == 0:
            duplicates.append(path.name)
    if duplicates:
        return {
            "path": None,
            "duplicate_of": duplicates,
            "differences": differences,
            "reason": "As previsões reproduzem um CSV anterior; nenhum duplicado foi criado.",
        }

    path = ROOT / "submissions" / filename
    if path.exists():
        existing = pd.read_csv(path)
        if (
            existing.columns.tolist() != ["RowId", "AGE_RANGE"]
            or not existing.RowId.equals(reference.RowId)
            or not np.array_equal(existing.AGE_RANGE.to_numpy(), labels)
        ):
            raise FileExistsError(f"Não será substituída a submissão diferente {path}.")
    else:
        _export_predictions(data, labels, filename)
    reread = pd.read_csv(path)
    if reread.shape != (3655, 2):
        raise ValueError(f"Dimensão inválida na releitura da submissão: {reread.shape}.")
    if reread.columns.tolist() != ["RowId", "AGE_RANGE"]:
        raise ValueError("O CSV relido não tem cabeçalho RowId,AGE_RANGE.")
    if not reread.RowId.equals(reference.RowId):
        raise ValueError("Os RowIds do CSV não correspondem à ordem oficial.")
    if (
        not pd.api.types.is_integer_dtype(reread.AGE_RANGE)
        or not reread.AGE_RANGE.between(0, 15).all()
    ):
        raise ValueError("O CSV relido contém classes fora do intervalo inteiro 0–15.")
    return {
        "path": str(path.relative_to(ROOT)),
        "duplicate_of": [],
        "differences": differences,
        "reason": None,
        "rows": int(len(reread)),
        "columns": reread.columns.tolist(),
        "index_written": False,
    }


def _fixed_split_ids(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "train_rows": int(len(data["X_train"])),
        "validation_rows": int(len(data["X_val"])),
        "train_index_sha256": hashlib.sha256(
            np.asarray(data["X_train"].index).tobytes()
        ).hexdigest(),
        "validation_index_sha256": hashlib.sha256(
            np.asarray(data["X_val"].index).tobytes()
        ).hexdigest(),
        "train_target_counts": {
            str(k): int(v)
            for k, v in data["y_train"].value_counts().sort_index().items()
        },
        "validation_target_counts": {
            str(k): int(v)
            for k, v in data["y_val"].value_counts().sort_index().items()
        },
    }


def run_033() -> dict[str, Any]:
    experiment = "033"
    config = {
        "control_031": {
            "depth": CATBOOST_SELECTED["depth"],
            "learning_rate": CATBOOST_SELECTED["learning_rate"],
            "selected_029_iterations": CATBOOST_SELECTED["iterations"],
            "max_iterations": CATBOOST_MAX_ITERATIONS,
            "early_stopping_rounds": CATBOOST_EARLY_STOPPING,
            "resource_parameters": CATBOOST_RESOURCE_PARAMETERS,
            "thread_count": 1,
            "control_031_weighted_selected_iterations": 376,
        },
        "alphas": [0.0, 0.25, 0.5],
        "weight_formula": "freq^-alpha normalized to arithmetic mean 1 using fit labels only",
        "selection_metric": "sklearn Macro F1, labels 0..15, unweighted",
        "seed": SEED,
        "outer_split": {"test_size": 0.2, "random_state": SEED, "stratified": True},
        "inner_split": {"test_size": 0.15, "random_state": SEED, "stratified": True},
    }
    context = _context(experiment, config)
    checkpoint = _load_checkpoint(context)
    stored = checkpoint["experiments"].setdefault(
        experiment, {"internal": {}, "result": None}
    )
    if stored.get("complete") and stored.get("result"):
        print("033: resultado reutilizado após validar dados, preparação, código, parâmetros e versões.")
        return stored["result"]

    run_started = time.perf_counter()
    data = load_experiment_data()
    _validate_features(data["X_train"])
    split = _fixed_split_ids(data)
    internal_rows: list[dict[str, Any]] = []
    histories: dict[str, Any] = {}
    for alpha in (0.0, 0.25, 0.5):
        key = f"alpha_{alpha:g}"
        if key in stored["internal"]:
            candidate = stored["internal"][key]
        else:
            print(f"033: selecionar alpha={alpha:g}")
            candidate, _ = _catboost_candidate(
                data["X_inner_train"],
                data["y_inner_train"],
                data["X_inner_val"],
                data["y_inner_val"],
                alpha,
            )
            stored["internal"][key] = candidate
            _save_checkpoint(checkpoint)
        internal_rows.append(
            {name: value for name, value in candidate.items() if name != "evals_result"}
        )
        histories[str(alpha)] = candidate["evals_result"]

    selected = max(internal_rows, key=lambda row: row["sklearn_macro_f1"])
    alpha = float(selected["alpha"])
    selected_weights = _weights_for_alpha(data["y_train"], alpha)
    fixed_started = time.perf_counter()
    fixed_model, fixed_fit = _catboost_fit(
        data["X_train"],
        data["y_train"],
        iterations=int(selected["selected_iterations"]),
        class_weights_list=selected_weights,
    )
    train_probabilities = _catboost_probabilities(fixed_model, data["X_train"])
    validation_probabilities = _catboost_probabilities(fixed_model, data["X_val"])
    fixed_metrics = {
        "train": _metrics(data["y_train"], train_probabilities),
        "validation": _metrics(data["y_val"], validation_probabilities),
        "selected_iterations": int(selected["selected_iterations"]),
        "class_weights_from_8769_training_labels": selected_weights,
        "fit_seconds": fixed_fit["seconds"],
        "elapsed_seconds": time.perf_counter() - fixed_started,
    }
    curves = _plot_033_curves(histories)
    confusion = _plot_confusion(data["y_val"], validation_probabilities, experiment)
    del fixed_model
    gc.collect()

    raw_train, raw_test = _raw_frames()
    full_prepared = preparar_dados(raw_train, raw_test)
    if len(full_prepared["X_train"]) != 10962 or len(full_prepared["X_test"]) != 3655:
        raise ValueError("As dimensões do treino completo/teste diferem de 10.962/3.655.")
    _validate_features(full_prepared["X_train"])
    weights_full = _weights_for_alpha(full_prepared["y_train"], alpha)
    export_model, export_detail = _catboost_fit(
        full_prepared["X_train"],
        full_prepared["y_train"],
        iterations=int(selected["selected_iterations"]),
        class_weights_list=weights_full,
    )
    test_probabilities = _catboost_probabilities(
        export_model, full_prepared["X_test"]
    )
    submission = _compare_and_export(
        data,
        test_probabilities.argmax(axis=1),
        "submission_033_catboost_ponderacao_intermedia.csv",
    )
    result = {
        "experiment": experiment,
        "configuration": config,
        "fixed_split": split,
        "inner_split_rows": {
            "fit": int(len(data["X_inner_train"])),
            "selection": int(len(data["X_inner_val"])),
        },
        "internal_candidates": internal_rows,
        "selected_internally": selected,
        "fixed_metrics": fixed_metrics,
        "fixed_validation_used_for_selection": False,
        "curves": curves,
        "confusion_matrix": confusion,
        "training_seconds": float(
            sum(row["fit_seconds"] for row in internal_rows)
            + fixed_fit["seconds"]
            + export_detail["seconds"]
        ),
        "wall_seconds": time.perf_counter() - run_started,
        "export": {
            "training_rows": int(len(full_prepared["X_train"])),
            "class_weights_from_full_training_labels": weights_full,
            "fit_seconds": export_detail["seconds"],
        },
        "submission": submission,
        "kaggle_score": None,
        "control_reconstruction_note": (
            "031: CatBoost depth=4, learning_rate=0.1, max 1000 iterations, "
            "early stopping 50, seed 42, thread_count=1 and all documented CTR/RAM "
            "parameters reproduced. 031's weighted candidate selected 376 iterations. "
            "The original 029 runner is absent; the stored config/output and the executable "
            "031 runner are the source for reconstruction."
        ),
        "resource_note": (
            "used_ram_limit=512mb is CatBoost's CTR memory setting, not a process-wide "
            "limit. All original 031 resource parameters were preserved without adjustment."
        ),
    }
    _record_result(experiment, result)
    stored["result"] = result
    stored["complete"] = True
    _save_checkpoint(checkpoint)
    return result


def _three_way_split(data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    X_fit, X_held, y_fit, y_held = train_test_split(
        data["X_train"],
        data["y_train"],
        test_size=0.30,
        random_state=SEED,
        stratify=data["y_train"],
    )
    X_select, X_blend, y_select, y_blend = train_test_split(
        X_held,
        y_held,
        test_size=0.50,
        random_state=SEED,
        stratify=y_held,
    )
    groups = {
        "fit": (X_fit, y_fit),
        "selection": (X_select, y_select),
        "blend": (X_blend, y_blend),
    }
    indexes = [pair[0].index for pair in groups.values()]
    if any(not left.intersection(right).empty for i, left in enumerate(indexes) for right in indexes[i + 1 :]):
        raise ValueError("As partições de fit, seleção e mistura não são disjuntas.")
    if set.intersection(*(set(y.unique()) for _, y in groups.values())) != set(CLASS_IDS):
        raise ValueError("Cada partição 034 deve conter as 16 classes.")
    hashes = {
        name: hashlib.sha256(np.asarray(X.index).tobytes()).hexdigest()
        for name, (X, _) in groups.items()
    }
    return groups, hashes


def _fit_mlp_internal(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_selection: pd.DataFrame,
    y_selection: pd.Series,
) -> tuple[int, dict[str, list[float]], float, Any]:
    preprocessor, fit_array, selection_array = fit_preprocessor(X_fit, X_selection)
    model, history, seconds = _fit_mlp(
        fit_array,
        y_fit.to_numpy(dtype=np.int64),
        MAX_MLP_EPOCHS,
        MLP_LEARNING_RATE,
        weighting="inverse_sqrt",
        dropout=MLP_DROPOUT,
        seed=SEED,
        X_monitor=selection_array,
        y_monitor=y_selection.to_numpy(dtype=np.int64),
    )
    epoch, _ = _best_epoch(history)
    del model
    gc.collect()
    return epoch, history, seconds, preprocessor


def _fit_mlp_for_epochs(
    X_fit: pd.DataFrame, y_fit: pd.Series, epochs: int
) -> tuple[Any, Any, np.ndarray, float]:
    preprocessor, fit_array, _ = fit_preprocessor(X_fit, X_fit)
    model, _, seconds = _fit_mlp(
        fit_array,
        y_fit.to_numpy(dtype=np.int64),
        epochs,
        MLP_LEARNING_RATE,
        weighting="inverse_sqrt",
        dropout=MLP_DROPOUT,
        seed=SEED,
    )
    return model, preprocessor, fit_array, seconds


def _xgb_probabilities(model: Any, X: pd.DataFrame) -> np.ndarray:
    return align_probabilities(
        model.predict_proba(X), np.asarray(model.named_steps["model"].classes_)
    )


def run_034() -> dict[str, Any]:
    experiment = "034"
    config = {
        "ensemble_025": {
            "xgboost": XGB_025,
            "mlp": {
                "architecture": [64, 32],
                "dropout": MLP_DROPOUT,
                "weighting": "inverse_sqrt",
                "learning_rate": MLP_LEARNING_RATE,
                "batch_size": 128,
                "max_epochs_for_internal_selection": MAX_MLP_EPOCHS,
                "original_selected_epochs": 85,
                "seed": SEED,
            },
            "internal_blend": [0.75, 0.25],
        },
        "catboost_031": {
            **CATBOOST_SELECTED,
            "class_weight_alpha": 0.5,
            "max_iterations": CATBOOST_MAX_ITERATIONS,
            "early_stopping_rounds": CATBOOST_EARLY_STOPPING,
            "resource_parameters": CATBOOST_RESOURCE_PARAMETERS,
        },
        "blend_weights_ensemble_catboost": [list(weight) for weight in BLEND_WEIGHTS],
        "three_way_split": {
            "fit": 0.70,
            "component_selection": 0.15,
            "blend_selection": 0.15,
            "random_state": SEED,
            "stratified": True,
        },
        "seed": SEED,
    }
    context = _context(experiment, config)
    checkpoint = _load_checkpoint(context)
    stored = checkpoint["experiments"].setdefault(
        experiment,
        {"component_selection": None, "blend_selection": None, "result": None},
    )
    if stored.get("complete") and stored.get("result"):
        print("034: resultado reutilizado após validar dados, preparação, código, parâmetros e versões.")
        return stored["result"]

    data = load_experiment_data()
    _validate_features(data["X_train"])
    groups, hashes = _three_way_split(data)
    X_fit, y_fit = groups["fit"]
    X_select, y_select = groups["selection"]
    X_blend, y_blend = groups["blend"]
    if stored["component_selection"] is None:
        print("034: selecionar época MLP e iterações CatBoost numa partição própria")
        mlp_epoch, mlp_history, mlp_seconds, _ = _fit_mlp_internal(
            X_fit, y_fit, X_select, y_select
        )
        cat_weights_fit = _weights_for_alpha(y_fit, 0.5)
        cat_model, cat_detail = _catboost_fit(
            X_fit,
            y_fit,
            X_select,
            y_select,
            iterations=CATBOOST_MAX_ITERATIONS,
            class_weights_list=cat_weights_fit,
            early_stopping=True,
        )
        cat_select_probabilities = _catboost_probabilities(cat_model, X_select)
        cat_selection_score = _score(y_select, cat_select_probabilities)
        cat_native = next(
            (
                float(value)
                for name, value in cat_detail["best_score"]["validation"].items()
                if "TotalF1" in name and "Macro" in name
            ),
            None,
        )
        if cat_native is None or not np.isclose(
            cat_native, cat_selection_score, atol=1e-12
        ):
            raise ValueError("A Macro F1 CatBoost de seleção não confere com sklearn.")
        stored["component_selection"] = {
            "mlp_epoch": mlp_epoch,
            "mlp_best_selection_macro_f1": float(
                max(mlp_history["validation_macro_f1"])
            ),
            "mlp_history": mlp_history,
            "mlp_selection_seconds": mlp_seconds,
            "catboost_iterations": max(int(cat_model.get_best_iteration()) + 1, 1),
            "catboost_selection_macro_f1": cat_selection_score,
            "catboost_native_macro_f1": cat_native,
            "catboost_class_weights_from_fit_labels": cat_weights_fit,
            "catboost_evals_result": cat_detail["evals_result"],
            "catboost_selection_seconds": cat_detail["seconds"],
            "fit_selection_blend_index_sha256": hashes,
        }
        del cat_model
        gc.collect()
        _save_checkpoint(checkpoint)
    selection = stored["component_selection"]

    if stored["blend_selection"] is None:
        print("034: ajustar componentes no fit e escolher mistura na partição blend")
        xgb_started = time.perf_counter()
        xgb = make_xgb().fit(X_fit, y_fit)
        xgb_blend_fit_seconds = time.perf_counter() - xgb_started
        xgb_blend = _xgb_probabilities(xgb, X_blend)
        mlp, mlp_preprocessor, _, mlp_blend_fit_seconds = _fit_mlp_for_epochs(
            X_fit, y_fit, int(selection["mlp_epoch"])
        )
        blend_array = to_dense_float32(mlp_preprocessor.transform(X_blend))
        mlp_blend = _predict_probabilities(mlp, blend_array)
        cat_model, cat_blend_timing = _catboost_fit(
            X_fit,
            y_fit,
            iterations=int(selection["catboost_iterations"]),
            class_weights_list=_weights_for_alpha(y_fit, 0.5),
        )
        cat_blend = _catboost_probabilities(cat_model, X_blend)
        if not np.allclose(xgb_blend.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError("Probabilidades XGBoost não somam 1.")
        ensemble_blend = 0.75 * xgb_blend + 0.25 * mlp_blend
        candidates: list[dict[str, Any]] = []
        probabilities_by_weight: dict[str, np.ndarray] = {}
        for ensemble_weight, catboost_weight in BLEND_WEIGHTS:
            label = f"{ensemble_weight:.0%}/{catboost_weight:.0%}"
            probabilities = (
                ensemble_weight * ensemble_blend + catboost_weight * cat_blend
            )
            if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
                raise ValueError(f"A mistura {label} não conserva probabilidades normalizadas.")
            metrics = _metrics_with_predictions(y_blend, probabilities)
            candidates.append(
                {
                    "weight_label": label,
                    "ensemble_weight": ensemble_weight,
                    "catboost_weight": catboost_weight,
                    "internal_macro_f1": metrics["macro_f1"],
                    "internal_metrics": metrics,
                }
            )
            probabilities_by_weight[label] = probabilities
        selected = max(candidates, key=lambda row: row["internal_macro_f1"])
        stored["blend_selection"] = {
            "candidates": candidates,
            "selected": selected,
            "xgb_fit_seconds": None,
            "mlp_fit_seconds": mlp_blend_fit_seconds,
            "catboost_fit_seconds": cat_blend_timing["seconds"],
            "blend_index_sha256": hashlib.sha256(
                np.asarray(X_blend.index).tobytes()
            ).hexdigest(),
        }
        stored["blend_selection"]["xgb_fit_seconds"] = xgb_blend_fit_seconds
        _save_checkpoint(checkpoint)
        del xgb, mlp, mlp_preprocessor, cat_model
        gc.collect()
    blend_selection = stored["blend_selection"]
    selected = blend_selection["selected"]

    print("034: reajustar componentes nas 8.769 linhas e avaliar validação fixa")
    fixed_started = time.perf_counter()
    xgb_fixed_started = time.perf_counter()
    xgb_fixed = make_xgb().fit(data["X_train"], data["y_train"])
    xgb_fixed_seconds = time.perf_counter() - xgb_fixed_started
    xgb_train = _xgb_probabilities(xgb_fixed, data["X_train"])
    xgb_validation = _xgb_probabilities(xgb_fixed, data["X_val"])
    mlp_fixed, mlp_preprocessor, mlp_train_array, mlp_fixed_seconds = _fit_mlp_for_epochs(
        data["X_train"], data["y_train"], int(selection["mlp_epoch"])
    )
    mlp_validation_array = to_dense_float32(
        mlp_preprocessor.transform(data["X_val"])
    )
    mlp_train = _predict_probabilities(mlp_fixed, mlp_train_array)
    mlp_validation = _predict_probabilities(mlp_fixed, mlp_validation_array)
    ensemble_train = 0.75 * xgb_train + 0.25 * mlp_train
    ensemble_validation = 0.75 * xgb_validation + 0.25 * mlp_validation
    cat_weights_train = _weights_for_alpha(data["y_train"], 0.5)
    cat_fixed, cat_fixed_timing = _catboost_fit(
        data["X_train"],
        data["y_train"],
        iterations=int(selection["catboost_iterations"]),
        class_weights_list=cat_weights_train,
    )
    cat_train = _catboost_probabilities(cat_fixed, data["X_train"])
    cat_validation = _catboost_probabilities(cat_fixed, data["X_val"])
    blend_train = (
        selected["ensemble_weight"] * ensemble_train
        + selected["catboost_weight"] * cat_train
    )
    blend_validation = (
        selected["ensemble_weight"] * ensemble_validation
        + selected["catboost_weight"] * cat_validation
    )
    fixed_metrics = {
        "train": _metrics(data["y_train"], blend_train),
        "validation": _metrics(data["y_val"], blend_validation),
        "selected_mlp_epochs": int(selection["mlp_epoch"]),
        "selected_catboost_iterations": int(selection["catboost_iterations"]),
        "catboost_class_weights_from_8769_training_labels": cat_weights_train,
        "elapsed_seconds": time.perf_counter() - fixed_started,
        "component_fit_seconds": {
            "xgboost": xgb_fixed_seconds,
            "mlp": mlp_fixed_seconds,
            "catboost": cat_fixed_timing["seconds"],
        },
    }
    curves = _plot_034_curves(
        selection["mlp_history"], blend_selection["candidates"]
    )
    confusion = _plot_confusion(data["y_val"], blend_validation, experiment)
    del xgb_fixed, mlp_fixed, mlp_preprocessor, cat_fixed
    gc.collect()

    print("034: reajustar componentes nas 10.962 linhas para previsões Kaggle")
    raw_train, raw_test = _raw_frames()
    full_prepared = preparar_dados(raw_train, raw_test)
    if len(full_prepared["X_train"]) != 10962 or len(full_prepared["X_test"]) != 3655:
        raise ValueError("As dimensões do treino completo/teste diferem de 10.962/3.655.")
    _validate_features(full_prepared["X_train"])
    xgb_export_started = time.perf_counter()
    full_xgb = make_xgb().fit(full_prepared["X_train"], full_prepared["y_train"])
    xgb_export_seconds = time.perf_counter() - xgb_export_started
    full_xgb_test = _xgb_probabilities(full_xgb, full_prepared["X_test"])
    mlp_full, mlp_full_preprocessor, _, mlp_export_seconds = _fit_mlp_for_epochs(
        full_prepared["X_train"],
        full_prepared["y_train"],
        int(selection["mlp_epoch"]),
    )
    mlp_test_array = to_dense_float32(
        mlp_full_preprocessor.transform(full_prepared["X_test"])
    )
    mlp_test = _predict_probabilities(mlp_full, mlp_test_array)
    ensemble_test = 0.75 * full_xgb_test + 0.25 * mlp_test
    full_cat_weights = _weights_for_alpha(full_prepared["y_train"], 0.5)
    cat_full, cat_export_timing = _catboost_fit(
        full_prepared["X_train"],
        full_prepared["y_train"],
        iterations=int(selection["catboost_iterations"]),
        class_weights_list=full_cat_weights,
    )
    cat_test = _catboost_probabilities(cat_full, full_prepared["X_test"])
    test_probabilities = (
        selected["ensemble_weight"] * ensemble_test
        + selected["catboost_weight"] * cat_test
    )
    submission = _compare_and_export(
        data,
        test_probabilities.argmax(axis=1),
        "submission_034_ensemble_catboost_ponderado.csv",
    )
    result = {
        "experiment": experiment,
        "configuration": config,
        "fixed_split": _fixed_split_ids(data),
        "three_way_split": {
            name: int(len(pair[0])) for name, pair in groups.items()
        },
        "three_way_split_index_sha256": hashes,
        "component_selection": {
            key: value
            for key, value in selection.items()
            if key not in {"mlp_history", "catboost_evals_result"}
        },
        "blend_candidates": blend_selection["candidates"],
        "selected_internally": selected,
        "fixed_metrics": fixed_metrics,
        "fixed_validation_used_for_selection": False,
        "component_fixed_validation_metrics": {
            "ensemble_025": {
                "train": _metrics(data["y_train"], ensemble_train),
                "validation": _metrics(data["y_val"], ensemble_validation),
            },
            "catboost_031": {
                "train": _metrics(data["y_train"], cat_train),
                "validation": _metrics(data["y_val"], cat_validation),
            },
        },
        "curves": curves,
        "confusion_matrix": confusion,
        "training_seconds": {
            "internal_mlp_selection": selection["mlp_selection_seconds"],
            "internal_catboost_selection": selection["catboost_selection_seconds"],
            "internal_xgboost_blend_fit": blend_selection["xgb_fit_seconds"],
            "internal_mlp_blend_fit": blend_selection["mlp_fit_seconds"],
            "internal_catboost_blend_fit": blend_selection["catboost_fit_seconds"],
            "fixed_total": fixed_metrics["elapsed_seconds"],
            "fixed_xgboost": xgb_fixed_seconds,
            "export_mlp": mlp_export_seconds,
            "export_xgboost": xgb_export_seconds,
            "export_catboost": cat_export_timing["seconds"],
        },
        "total_model_fit_seconds": float(
            sum(
                (
                    selection["mlp_selection_seconds"],
                    selection["catboost_selection_seconds"],
                    blend_selection["xgb_fit_seconds"],
                    blend_selection["mlp_fit_seconds"],
                    blend_selection["catboost_fit_seconds"],
                    xgb_fixed_seconds,
                    mlp_fixed_seconds,
                    cat_fixed_timing["seconds"],
                    mlp_export_seconds,
                    xgb_export_seconds,
                    cat_export_timing["seconds"],
                )
            )
        ),
        "export": {
            "training_rows": int(len(full_prepared["X_train"])),
            "class_weights_from_full_training_labels": full_cat_weights,
        },
        "submission": submission,
        "kaggle_score": None,
        "control_reconstruction_note": (
            "O ensemble 025 é XGBoost 007 (200 estimadores, depth 5, lr 0.1) e MLP "
            "64→32 (dropout 0.1, inverse_sqrt, Adam 0.0001, batch 128, seed 42), mistura "
            "interna 75/25. A época foi novamente selecionada apenas na partição componente "
            "dedicada, em vez de impor 85. O CatBoost é o 031 ponderado alpha=0.5, com "
            "parâmetros e restrições originais; as iterações são selecionadas na partição "
            "componente dedicada. A partição blend nunca é usada para early stopping."
        ),
        "resource_note": (
            "Todas as restrições de recursos do CatBoost 031 foram mantidas, com uma thread. "
            "used_ram_limit=512mb não é limite total do processo."
        ),
    }
    _record_result(experiment, result)
    stored["result"] = result
    stored["complete"] = True
    _save_checkpoint(checkpoint)
    del full_xgb, mlp_full, mlp_full_preprocessor, cat_full
    gc.collect()
    return result


def write_final_report() -> Path:
    stored = (
        json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        if RESULTS_PATH.is_file()
        else {}
    )
    lines = [
        "# Experiências DAA 033–034",
        "",
        "A validação fixa é desenvolvimento, não teste independente. Kaggle não foi usado "
        "para seleção e fica em branco. As execuções foram sequenciais e com uma thread.",
        "",
        "| Experiência | Macro F1 interna | Macro F1 fixa | Accuracy fixa | Seleção | CSV | Leitura |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    for experiment in ("033", "034"):
        result = stored.get(experiment)
        if not result:
            lines.append(f"| {experiment} | — | — | — | por executar | — | — |")
            continue
        if experiment == "033":
            internal = result["selected_internally"]["sklearn_macro_f1"]
            selection = f"alpha={result['selected_internally']['alpha']:g}"
        else:
            internal = result["selected_internally"]["internal_macro_f1"]
            selection = result["selected_internally"]["weight_label"]
        fixed = result["fixed_metrics"]["validation"]
        submission = result["submission"]
        csv = submission.get("path") or "—"
        controls = [0.4614702838167389, 0.4587, 0.448318]
        status = (
            "acima dos controlos fixos (referência descritiva)"
            if fixed["macro_f1"] > max(controls)
            else "exploratório: abaixo do melhor controlo 025/028"
        )
        if submission.get("duplicate_of"):
            status += "; previsões duplicadas"
        lines.append(
            f"| {experiment} | {internal:.4f} | {fixed['macro_f1']:.4f} | "
            f"{fixed['accuracy']:.4f} | {selection} | {csv} | {status} |"
        )
    lines.extend(
        [
            "",
            "## Interpretação",
            "",
            "Os candidatos foram selecionados só pelas partições internas. As pontuações "
            "fixas e as comparações de CSVs são diagnósticas, não serviram para escolher "
            "alpha, épocas, iterações ou mistura. Uma submissão só é criada para previsões "
            "distintas; duplicados de controlos são descartados.",
            "",
        ]
    )
    path = ROOT / "reports" / "experiments_033_034.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
