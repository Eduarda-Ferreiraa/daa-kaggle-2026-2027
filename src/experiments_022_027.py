"""Reproducible runners for DAA experiments 022–027."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader, TensorDataset

from src.experiments_016_018 import (
    AGE_BANDS,
    BATCH_SIZE,
    CLASS_IDS,
    NUM_CLASSES,
    AgeRangeMLP,
    _classification_metrics,
    _export_predictions,
    _official_row_reference,
    align_probabilities,
    class_weights,
    fit_preprocessor,
    load_experiment_data,
    make_xgb,
    probabilities_from_logits,
)
from src.preparar_dados import build_preprocessor


ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "reports" / ".experiments_022_027_results.json"
_DATA_CACHE: dict[str, Any] | None = None
torch.set_num_threads(1)


class BatchNormAgeRangeMLP(torch.nn.Module):
    def __init__(self, input_features: int, dropout: float):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_features, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(64, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(32, NUM_CLASSES),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


def _load_data() -> dict[str, Any]:
    global _DATA_CACHE
    if _DATA_CACHE is None:
        _DATA_CACHE = load_experiment_data()
    return _DATA_CACHE


def _fit_mlp(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    epochs: int,
    learning_rate: float,
    *,
    weighting: str = "inverse_sqrt",
    optimizer_name: str = "Adam",
    momentum: float = 0.9,
    dropout: float = 0.0,
    batch_norm: bool = False,
    seed: int = 42,
    X_monitor: np.ndarray | None = None,
    y_monitor: np.ndarray | None = None,
) -> tuple[torch.nn.Module, dict[str, list[float]], float]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model_type = BatchNormAgeRangeMLP if batch_norm else AgeRangeMLP
    model = model_type(X_fit.shape[1], dropout=dropout).to("cpu")
    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights(y_fit, weighting=weighting)
    )
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=0
        )
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=0,
        )
    else:
        raise ValueError(f"Otimizador desconhecido: {optimizer_name}")

    dataset = TensorDataset(torch.from_numpy(X_fit), torch.from_numpy(y_fit))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    history = {
        "epoch": [],
        "train_loss": [],
        "validation_loss": [],
        "train_macro_f1": [],
        "validation_macro_f1": [],
    }
    y_tensor = torch.from_numpy(y_fit)
    y_monitor_tensor = (
        torch.from_numpy(y_monitor) if y_monitor is not None else None
    )
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        for features, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), targets)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_logits = model(torch.from_numpy(X_fit))
            history["epoch"].append(epoch)
            history["train_loss"].append(float(criterion(train_logits, y_tensor).item()))
            history["train_macro_f1"].append(
                float(
                    f1_score(
                        y_fit,
                        train_logits.argmax(dim=1).numpy(),
                        labels=CLASS_IDS,
                        average="macro",
                        zero_division=0,
                    )
                )
            )
            if X_monitor is not None and y_monitor_tensor is not None:
                monitor_logits = model(torch.from_numpy(X_monitor))
                history["validation_loss"].append(
                    float(criterion(monitor_logits, y_monitor_tensor).item())
                )
                history["validation_macro_f1"].append(
                    float(
                        f1_score(
                            y_monitor,
                            monitor_logits.argmax(dim=1).numpy(),
                            labels=CLASS_IDS,
                            average="macro",
                            zero_division=0,
                        )
                    )
                )
    return model, history, time.perf_counter() - started


def _best_epoch(history: dict[str, list[float]]) -> tuple[int, float]:
    scores = history["validation_macro_f1"]
    if not scores:
        raise ValueError("Não há Macro F1 interna para selecionar a época.")
    index = int(np.argmax(scores))
    return int(history["epoch"][index]), float(scores[index])


def _predict_probabilities(
    model: torch.nn.Module, features: np.ndarray
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        probabilities = (
            torch.softmax(model(torch.from_numpy(features)), dim=1).cpu().numpy()
        )
    if probabilities.shape != (len(features), NUM_CLASSES):
        raise ValueError(
            f"Esperadas probabilidades para {NUM_CLASSES} classes; "
            f"recebido {probabilities.shape}."
        )
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("As probabilidades do MLP não somam 1.")
    return probabilities


def _fit_model_for_split(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_eval: pd.DataFrame,
    *,
    epochs: int,
    learning_rate: float,
    weighting: str = "inverse_sqrt",
    optimizer_name: str = "Adam",
    momentum: float = 0.9,
    dropout: float = 0.0,
    batch_norm: bool = False,
    seed: int = 42,
) -> tuple[torch.nn.Module, np.ndarray, float]:
    _, X_fit_arr, X_eval_arr = fit_preprocessor(X_fit, X_eval)
    model, _, seconds = _fit_mlp(
        X_fit_arr,
        y_fit.to_numpy(dtype=np.int64),
        epochs,
        learning_rate,
        weighting=weighting,
        optimizer_name=optimizer_name,
        momentum=momentum,
        dropout=dropout,
        batch_norm=batch_norm,
        seed=seed,
    )
    return model, _predict_probabilities(model, X_eval_arr), seconds


def _xgb_probabilities(
    X_fit: pd.DataFrame, y_fit: pd.Series, X_eval: pd.DataFrame
) -> tuple[Any, np.ndarray]:
    model = make_xgb().fit(X_fit, y_fit)
    probabilities = align_probabilities(
        model.predict_proba(X_eval), model.named_steps["model"].classes_
    )
    return model, probabilities


def _plot_histories(
    experiment: str, histories: dict[str, dict[str, list[float]]]
) -> Path:
    path = ROOT / "reports" / f"experiencia_{experiment}_curvas.png"
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for label, history in histories.items():
        epochs = history["epoch"]
        axes[0].plot(epochs, history["train_loss"], label=f"{label} treino")
        if history["validation_loss"]:
            axes[0].plot(
                epochs, history["validation_loss"], linestyle="--", label=f"{label} validação"
            )
        axes[1].plot(epochs, history["train_macro_f1"], label=f"{label} treino")
        if history["validation_macro_f1"]:
            axes[1].plot(
                epochs,
                history["validation_macro_f1"],
                linestyle="--",
                label=f"{label} validação",
            )
    axes[0].set(title="Perda", xlabel="Época", ylabel="Cross-entropy ponderada")
    axes[1].set(title="Macro F1", xlabel="Época", ylabel="Macro F1")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize="small")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def _record_result(experiment: str, result: dict[str, Any]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    stored = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else {}
    stored[experiment] = result
    RESULTS_PATH.write_text(json.dumps(stored, ensure_ascii=False, indent=2))


def _submission_differences(path: Path) -> dict[str, int]:
    current = pd.read_csv(path)
    differences: dict[str, int] = {}
    for previous_path in sorted((ROOT / "submissions").glob("submission_*.csv")):
        if previous_path.resolve() == path.resolve():
            continue
        previous = pd.read_csv(previous_path)
        aligned = previous.set_index("RowId").reindex(current["RowId"])
        if aligned["AGE_RANGE"].isna().any():
            raise ValueError(f"RowIds incompatíveis em {previous_path}.")
        differences[previous_path.name] = int(
            (current["AGE_RANGE"].to_numpy() != aligned["AGE_RANGE"].to_numpy()).sum()
        )
    return differences


def _export_if_useful(
    data: dict[str, Any],
    predictions: np.ndarray,
    experiment: str,
    filename: str,
    fixed_f1: float,
    best_control_f1: float,
) -> tuple[str | None, dict[str, int]]:
    if fixed_f1 <= best_control_f1:
        return None, {}
    path = _export_predictions(data, predictions, filename)
    return str(path.relative_to(ROOT)), _submission_differences(path)


def _select_and_evaluate_mlp(
    experiment: str,
    data: dict[str, Any],
    candidates: list[dict[str, Any]],
    histories: dict[str, dict[str, list[float]]],
    *,
    weighting: str,
    learning_rate_key: str = "learning_rate",
    optimizer_name: str = "Adam",
    momentum: float = 0.9,
    batch_norm: bool = False,
) -> tuple[dict[str, Any], np.ndarray]:
    selected = max(candidates, key=lambda row: row["internal_macro_f1"])
    _, X_train_arr, X_val_arr = fit_preprocessor(data["X_train"], data["X_val"])
    model, _, seconds = _fit_mlp(
        X_train_arr,
        data["y_train"].to_numpy(dtype=np.int64),
        selected["best_epoch"],
        float(selected.get(learning_rate_key, 0.0001)),
        weighting=weighting,
        optimizer_name=optimizer_name,
        momentum=momentum,
        dropout=float(selected.get("dropout", 0.0)),
        batch_norm=batch_norm,
    )
    predictions = _predict_probabilities(model, X_val_arr).argmax(axis=1)
    metrics = _classification_metrics(
        data["y_val"].to_numpy(dtype=np.int64), predictions
    )
    chart = _plot_histories(experiment, histories)
    return {
        "selected": selected,
        "fixed_metrics": metrics,
        "fixed_training_seconds": seconds,
        "curves": str(chart.relative_to(ROOT)),
    }, predictions


def _run_dropout_sweep(
    experiment: str,
    dropouts: tuple[float, ...],
    *,
    batch_norm: bool = False,
) -> dict[str, Any]:
    data = _load_data()
    _, X_fit, X_val = fit_preprocessor(data["X_inner_train"], data["X_inner_val"])
    y_fit = data["y_inner_train"].to_numpy(dtype=np.int64)
    y_val = data["y_inner_val"].to_numpy(dtype=np.int64)
    candidates: list[dict[str, Any]] = []
    histories: dict[str, dict[str, list[float]]] = {}
    started = time.perf_counter()
    for dropout in dropouts:
        model, history, seconds = _fit_mlp(
            X_fit,
            y_fit,
            100,
            0.0001,
            weighting="inverse_sqrt",
            dropout=dropout,
            batch_norm=batch_norm,
            X_monitor=X_val,
            y_monitor=y_val,
        )
        epoch, score = _best_epoch(history)
        label = f"dropout_{dropout:g}" + ("_batchnorm" if batch_norm else "")
        histories[label] = history
        candidates.append(
            {
                "dropout": dropout,
                "best_epoch": epoch,
                "internal_macro_f1": score,
                "training_seconds": seconds,
            }
        )

    control_scores = {"017": 0.4351, "020": 0.4453596932933473}
    best_control = max(control_scores.values())
    config = "MLP 64→32, Adam lr=0.0001, pesos 1/sqrt(freq), dropout variável"
    evaluation, predictions = _select_and_evaluate_mlp(
        experiment,
        data,
        candidates,
        histories,
        weighting="inverse_sqrt",
        batch_norm=batch_norm,
    )
    selected = evaluation["selected"]
    csv_path: str | None = None
    differences: dict[str, int] = {}
    export_seconds = 0.0
    if evaluation["fixed_metrics"]["macro_f1"] > best_control:
        _, X_all, X_test = fit_preprocessor(
            data["prepared"]["X_train"], data["prepared"]["X_test"]
        )
        model_all, _, export_seconds = _fit_mlp(
            X_all,
            data["prepared"]["y_train"].to_numpy(dtype=np.int64),
            selected["best_epoch"],
            0.0001,
            weighting="inverse_sqrt",
            dropout=selected["dropout"],
            batch_norm=batch_norm,
        )
        predictions_test = _predict_probabilities(model_all, X_test).argmax(axis=1)
        csv_path, differences = _export_if_useful(
            data,
            predictions_test,
            experiment,
            "submission_023_mlp_dropout_refinado.csv",
            evaluation["fixed_metrics"]["macro_f1"],
            best_control,
        )
    result = {
        "experiment": experiment,
        "model_seeds": [42],
        "configuration": (
            f"{config}; seleção interna dropout={selected['dropout']:g}, "
            f"época={selected['best_epoch']}"
        ),
        "control_configurations_reproduced": [
            "020: arquitetura 64→32, Adam 0.0001, inverse_sqrt, batch 128, seed 42"
        ],
        "internal_candidates": candidates,
        **evaluation,
        "control_fixed_macro_f1": control_scores,
        "improved_best_control": evaluation["fixed_metrics"]["macro_f1"] > best_control,
        "exploratory": evaluation["fixed_metrics"]["macro_f1"] <= best_control,
        "total_seconds": time.perf_counter() - started,
        "export_training_seconds": export_seconds,
        "submission": csv_path,
        "prediction_differences": differences,
        "fixed_prediction_count": int(len(predictions)),
        "kaggle_score": None,
    }
    _record_result(experiment, result)
    return {"result": result, "histories": histories}


def _run_batchnorm_comparison() -> dict[str, Any]:
    experiment = "026"
    data = _load_data()
    started = time.perf_counter()
    _, X_fit, X_val = fit_preprocessor(
        data["X_inner_train"], data["X_inner_val"]
    )
    y_fit = data["y_inner_train"].to_numpy(dtype=np.int64)
    y_val = data["y_inner_val"].to_numpy(dtype=np.int64)
    histories: dict[str, dict[str, list[float]]] = {}
    candidates: dict[bool, dict[str, Any]] = {}
    for batch_norm, label in ((False, "sem_batchnorm"), (True, "com_batchnorm")):
        _, history, seconds = _fit_mlp(
            X_fit,
            y_fit,
            100,
            0.0001,
            weighting="inverse_sqrt",
            dropout=0.1,
            batch_norm=batch_norm,
            X_monitor=X_val,
            y_monitor=y_val,
        )
        epoch, score = _best_epoch(history)
        histories[label] = history
        candidates[batch_norm] = {
            "batch_norm": batch_norm,
            "dropout": 0.1,
            "best_epoch": epoch,
            "internal_macro_f1": score,
            "training_seconds": seconds,
        }

    _, X_train, X_fixed_val = fit_preprocessor(data["X_train"], data["X_val"])
    fixed_by_condition: dict[bool, dict[str, Any]] = {}
    fixed_seconds: dict[str, float] = {}
    for batch_norm in (False, True):
        candidate = candidates[batch_norm]
        model, _, seconds = _fit_mlp(
            X_train,
            data["y_train"].to_numpy(dtype=np.int64),
            candidate["best_epoch"],
            0.0001,
            weighting="inverse_sqrt",
            dropout=0.1,
            batch_norm=batch_norm,
        )
        predictions = _predict_probabilities(model, X_fixed_val).argmax(axis=1)
        fixed_by_condition[batch_norm] = _classification_metrics(
            data["y_val"].to_numpy(dtype=np.int64), predictions
        )
        fixed_seconds["com_batchnorm" if batch_norm else "sem_batchnorm"] = seconds

    chart = _plot_histories(experiment, histories)
    no_batchnorm = fixed_by_condition[False]
    with_batchnorm = fixed_by_condition[True]
    best_control = no_batchnorm["macro_f1"]
    csv_path: str | None = None
    differences: dict[str, int] = {}
    export_seconds = 0.0
    if with_batchnorm["macro_f1"] > best_control:
        _, X_all, X_test = fit_preprocessor(
            data["prepared"]["X_train"], data["prepared"]["X_test"]
        )
        model_all, _, export_seconds = _fit_mlp(
            X_all,
            data["prepared"]["y_train"].to_numpy(dtype=np.int64),
            candidates[True]["best_epoch"],
            0.0001,
            weighting="inverse_sqrt",
            dropout=0.1,
            batch_norm=True,
        )
        test_predictions = _predict_probabilities(model_all, X_test).argmax(axis=1)
        csv_path, differences = _export_if_useful(
            data,
            test_predictions,
            experiment,
            "submission_026_mlp_batchnorm.csv",
            with_batchnorm["macro_f1"],
            best_control,
        )
    result = {
        "experiment": experiment,
        "model_seeds": [42],
        "configuration": (
            f"MLP 020 com BatchNorm1d, dropout=0.1, Adam lr=0.0001, "
            f"inverse_sqrt, época={candidates[True]['best_epoch']}"
        ),
        "control_configurations_reproduced": [
            "020 sem BatchNorm: MLP 64→32, Adam lr=0.0001, inverse_sqrt, "
            f"dropout=0.1, época interna={candidates[False]['best_epoch']}"
        ],
        "internal_candidates": [candidates[False], candidates[True]],
        "selected": candidates[True],
        "fixed_metrics": with_batchnorm,
        "fixed_control_metrics": no_batchnorm,
        "control_fixed_macro_f1": {"020 sem BatchNorm": best_control},
        "reproduces_020_epoch": candidates[False]["best_epoch"] == 85,
        "reproduces_020_fixed_macro_f1": bool(
            np.isclose(best_control, 0.4453596932933473, atol=1e-8)
        ),
        "improved_best_control": with_batchnorm["macro_f1"] > best_control,
        "exploratory": with_batchnorm["macro_f1"] <= best_control,
        "total_seconds": time.perf_counter() - started,
        "fixed_training_seconds": fixed_seconds,
        "export_training_seconds": export_seconds,
        "curves": str(chart.relative_to(ROOT)),
        "submission": csv_path,
        "prediction_differences": differences,
        "kaggle_score": None,
    }
    _record_result(experiment, result)
    return {"result": result, "histories": histories}


def run_022() -> dict[str, Any]:
    data = _load_data()
    started = time.perf_counter()
    xgb_model, xgb_inner = _xgb_probabilities(
        data["X_inner_train"], data["y_inner_train"], data["X_inner_val"]
    )
    _, X_fit, X_val = fit_preprocessor(
        data["X_inner_train"], data["X_inner_val"]
    )
    y_fit = data["y_inner_train"].to_numpy(dtype=np.int64)
    y_val = data["y_inner_val"].to_numpy(dtype=np.int64)
    probabilities_by_seed: dict[int, np.ndarray] = {}
    histories: dict[str, dict[str, list[float]]] = {}
    seed_seconds: dict[str, float] = {}
    for seed in (42, 123, 2026):
        model, history, seconds = _fit_mlp(
            X_fit,
            y_fit,
            93,
            0.0001,
            weighting="inverse",
            seed=seed,
            X_monitor=X_val,
            y_monitor=y_val,
        )
        probabilities_by_seed[seed] = _predict_probabilities(model, X_val)
        histories[f"seed_{seed}"] = history
        seed_seconds[str(seed)] = seconds
    average_mlp = np.mean(list(probabilities_by_seed.values()), axis=0)
    candidates: list[dict[str, Any]] = []
    probability_sets = {
        "seed_42": probabilities_by_seed[42],
        "mean_3_seeds": average_mlp,
    }
    for label, mlp_probabilities in probability_sets.items():
        for xgb_weight, mlp_weight in ((0.75, 0.25), (0.60, 0.40)):
            predictions = (
                xgb_weight * xgb_inner + mlp_weight * mlp_probabilities
            ).argmax(axis=1)
            candidates.append(
                {
                    "mlp": label,
                    "xgboost_weight": xgb_weight,
                    "mlp_weight": mlp_weight,
                    "internal_metrics": _classification_metrics(y_val, predictions),
                }
            )
    selected = max(candidates, key=lambda row: row["internal_metrics"]["macro_f1"])

    xgb_fixed, xgb_fixed_prob = _xgb_probabilities(
        data["X_train"], data["y_train"], data["X_val"]
    )
    _, X_train_fixed, X_val_fixed = fit_preprocessor(data["X_train"], data["X_val"])
    fixed_mlps: dict[int, np.ndarray] = {}
    fixed_seconds: dict[str, float] = {}
    for seed in (42, 123, 2026):
        model, _, seconds = _fit_mlp(
            X_train_fixed,
            data["y_train"].to_numpy(dtype=np.int64),
            93,
            0.0001,
            weighting="inverse",
            seed=seed,
        )
        fixed_mlps[seed] = _predict_probabilities(model, X_val_fixed)
        fixed_seconds[str(seed)] = seconds
    fixed_probability_sets = {
        "seed_42": fixed_mlps[42],
        "mean_3_seeds": np.mean(list(fixed_mlps.values()), axis=0),
    }
    fixed_comparisons = {}
    for label, mlp_probabilities in fixed_probability_sets.items():
        for xgb_weight, mlp_weight in ((0.75, 0.25), (0.60, 0.40)):
            key = f"{label}_{xgb_weight:.2f}_{mlp_weight:.2f}"
            predictions = (
                xgb_weight * xgb_fixed_prob + mlp_weight * mlp_probabilities
            ).argmax(axis=1)
            fixed_comparisons[key] = _classification_metrics(
                data["y_val"].to_numpy(dtype=np.int64), predictions
            )
    selected_fixed_key = (
        f"{selected['mlp']}_{selected['xgboost_weight']:.2f}_{selected['mlp_weight']:.2f}"
    )
    fixed_metrics = fixed_comparisons[selected_fixed_key]
    chart = _plot_histories("022", histories)

    full_x = data["prepared"]["X_train"]
    full_y = data["prepared"]["y_train"]
    xgb_all, xgb_test = _xgb_probabilities(
        full_x, full_y, data["prepared"]["X_test"]
    )
    _, X_all, X_test = fit_preprocessor(full_x, data["prepared"]["X_test"])
    full_mlp_probs = []
    export_seconds = {}
    seeds = (42,) if selected["mlp"] == "seed_42" else (42, 123, 2026)
    for seed in seeds:
        model, _, seconds = _fit_mlp(
            X_all,
            full_y.to_numpy(dtype=np.int64),
            93,
            0.0001,
            weighting="inverse",
            seed=seed,
        )
        full_mlp_probs.append(_predict_probabilities(model, X_test))
        export_seconds[str(seed)] = seconds
    mlp_test = np.mean(full_mlp_probs, axis=0)
    test_predictions = (
        selected["xgboost_weight"] * xgb_test
        + selected["mlp_weight"] * mlp_test
    ).argmax(axis=1)
    prior_best = 0.4613
    csv_path, differences = _export_if_useful(
        data,
        test_predictions,
        "022",
        "submission_022_ensemble_multiplas_sementes.csv",
        fixed_metrics["macro_f1"],
        prior_best,
    )
    result = {
        "experiment": "022",
        "model_seeds": [42, 123, 2026],
        "configuration": (
            f"XGBoost 007 + MLP 015; seleção {selected['xgboost_weight']:.0%}/"
            f"{selected['mlp_weight']:.0%}, MLP={selected['mlp']}, Adam lr=0.0001, "
            "93 épocas, sementes 42/123/2026"
        ),
        "control_configurations_reproduced": [
            "007: XGBoost 200 árvores, depth 5, lr 0.1, hist, seed 42, n_jobs=1",
            "015/016: MLP 64→32, Adam lr=0.0001, inverse frequency, batch 128, 93 épocas",
            "016: mistura 75/25",
            "019: mistura 60/40",
        ],
        "internal_candidates": candidates,
        "selected": selected,
        "fixed_comparisons": fixed_comparisons,
        "fixed_metrics": fixed_metrics,
        "fixed_control_scores": {"014": 0.4613, "016": 0.4580, "019": 0.4557004629806364},
        "improved_best_control": fixed_metrics["macro_f1"] > prior_best,
        "exploratory": fixed_metrics["macro_f1"] <= prior_best,
        "average_seed_effects": {
            "same_weights_75_25_delta": (
                fixed_comparisons["mean_3_seeds_0.75_0.25"]["macro_f1"]
                - fixed_comparisons["seed_42_0.75_0.25"]["macro_f1"]
            ),
            "same_weights_60_40_delta": (
                fixed_comparisons["mean_3_seeds_0.60_0.40"]["macro_f1"]
                - fixed_comparisons["seed_42_0.60_0.40"]["macro_f1"]
            ),
            "weight_change_single_seed_delta": (
                fixed_comparisons["seed_42_0.60_0.40"]["macro_f1"]
                - fixed_comparisons["seed_42_0.75_0.25"]["macro_f1"]
            ),
            "weight_change_mean_seed_delta": (
                fixed_comparisons["mean_3_seeds_0.60_0.40"]["macro_f1"]
                - fixed_comparisons["mean_3_seeds_0.75_0.25"]["macro_f1"]
            ),
        },
        "seed_training_seconds_inner": seed_seconds,
        "seed_training_seconds_fixed": fixed_seconds,
        "export_training_seconds": export_seconds,
        "total_seconds": time.perf_counter() - started,
        "curves": str(chart.relative_to(ROOT)),
        "submission": csv_path,
        "prediction_differences": differences,
        "kaggle_score": None,
    }
    _record_result("022", result)
    return {"result": result, "histories": histories}


def run_023() -> dict[str, Any]:
    return _run_dropout_sweep("023", (0.0, 0.025, 0.05, 0.1))


def run_024() -> dict[str, Any]:
    data = _load_data()
    started = time.perf_counter()
    _, X_fit, X_val = fit_preprocessor(
        data["X_inner_train"], data["X_inner_val"]
    )
    y_fit = data["y_inner_train"].to_numpy(dtype=np.int64)
    y_val = data["y_inner_val"].to_numpy(dtype=np.int64)
    candidates = []
    histories = {}
    for learning_rate in (0.0005, 0.001, 0.002):
        model, history, seconds = _fit_mlp(
            X_fit,
            y_fit,
            100,
            learning_rate,
            weighting="inverse",
            optimizer_name="SGD",
            momentum=0.9,
            X_monitor=X_val,
            y_monitor=y_val,
        )
        epoch, score = _best_epoch(history)
        label = f"lr_{learning_rate:g}"
        histories[label] = history
        candidates.append(
            {
                "learning_rate": learning_rate,
                "best_epoch": epoch,
                "internal_macro_f1": score,
                "training_seconds": seconds,
            }
        )
    selected = max(candidates, key=lambda row: row["internal_macro_f1"])
    _, X_train_fixed, X_val_fixed = fit_preprocessor(
        data["X_train"], data["X_val"]
    )
    selected_model, _, fixed_seconds = _fit_mlp(
        X_train_fixed,
        data["y_train"].to_numpy(dtype=np.int64),
        selected["best_epoch"],
        selected["learning_rate"],
        weighting="inverse",
        optimizer_name="SGD",
        momentum=0.9,
    )
    fixed_predictions = _predict_probabilities(
        selected_model, X_val_fixed
    ).argmax(axis=1)
    fixed_metrics = _classification_metrics(
        data["y_val"].to_numpy(dtype=np.int64), fixed_predictions
    )
    chart = _plot_histories("024", histories)
    _, X_all, X_test = fit_preprocessor(
        data["prepared"]["X_train"], data["prepared"]["X_test"]
    )
    model_all, _, export_seconds = _fit_mlp(
        X_all,
        data["prepared"]["y_train"].to_numpy(dtype=np.int64),
        selected["best_epoch"],
        selected["learning_rate"],
        weighting="inverse",
        optimizer_name="SGD",
        momentum=0.9,
    )
    test_predictions = _predict_probabilities(model_all, X_test).argmax(axis=1)
    best_control = max(0.4365, 0.4250)
    csv_path, differences = _export_if_useful(
        data,
        test_predictions,
        "024",
        "submission_024_mlp_sgd_lr_refinado.csv",
        fixed_metrics["macro_f1"],
        best_control,
    )
    result = {
        "experiment": "024",
        "model_seeds": [42],
        "configuration": (
            f"MLP 64→32, SGD lr={selected['learning_rate']:g}, momentum=0.9, "
            f"pesos 1/frequência, batch=128, época={selected['best_epoch']}"
        ),
        "control_configurations_reproduced": [
            "018: MLP 64→32, SGD lr=0.001, momentum=0.9, inverse frequency, batch 128",
            "021: MLP 64→32, SGD lr=0.001, momentum=0.9, inverse_sqrt, batch 128",
        ],
        "internal_candidates": candidates,
        "selected": selected,
        "fixed_metrics": fixed_metrics,
        "control_fixed_macro_f1": {"018": 0.4365, "021": 0.4250},
        "improved_best_control": fixed_metrics["macro_f1"] > best_control,
        "exploratory": fixed_metrics["macro_f1"] <= best_control,
        "total_seconds": time.perf_counter() - started,
        "fixed_training_seconds": fixed_seconds,
        "export_training_seconds": export_seconds,
        "curves": str(chart.relative_to(ROOT)),
        "submission": csv_path,
        "prediction_differences": differences,
        "kaggle_score": None,
    }
    _record_result("024", result)
    return {"result": result, "histories": histories}


def run_025() -> dict[str, Any]:
    data = _load_data()
    started = time.perf_counter()
    _, X_inner, X_inner_val = fit_preprocessor(
        data["X_inner_train"], data["X_inner_val"]
    )
    model_inner, history, inner_seconds = _fit_mlp(
        X_inner,
        data["y_inner_train"].to_numpy(dtype=np.int64),
        100,
        0.0001,
        weighting="inverse_sqrt",
        dropout=0.1,
        X_monitor=X_inner_val,
        y_monitor=data["y_inner_val"].to_numpy(dtype=np.int64),
    )
    _, inner_best = _best_epoch(history)
    _, xgb_fixed = _xgb_probabilities(data["X_train"], data["y_train"], data["X_val"])
    _, X_train, X_val = fit_preprocessor(data["X_train"], data["X_val"])
    mlp_fixed, _, fixed_seconds = _fit_mlp(
        X_train,
        data["y_train"].to_numpy(dtype=np.int64),
        85,
        0.0001,
        weighting="inverse_sqrt",
        dropout=0.1,
    )
    fixed_predictions = (0.75 * xgb_fixed + 0.25 * _predict_probabilities(mlp_fixed, X_val)).argmax(axis=1)
    fixed_metrics = _classification_metrics(data["y_val"].to_numpy(dtype=np.int64), fixed_predictions)
    chart = _plot_histories("025", {"dropout_0.1": history})

    xgb_all, xgb_test = _xgb_probabilities(
        data["prepared"]["X_train"],
        data["prepared"]["y_train"],
        data["prepared"]["X_test"],
    )
    _, X_all, X_test = fit_preprocessor(
        data["prepared"]["X_train"], data["prepared"]["X_test"]
    )
    mlp_all, _, export_seconds = _fit_mlp(
        X_all,
        data["prepared"]["y_train"].to_numpy(dtype=np.int64),
        85,
        0.0001,
        weighting="inverse_sqrt",
        dropout=0.1,
    )
    test_predictions = (
        0.75 * xgb_test + 0.25 * _predict_probabilities(mlp_all, X_test)
    ).argmax(axis=1)
    best_control = max(0.4613, 0.4580)
    csv_path, differences = _export_if_useful(
        data,
        test_predictions,
        "025",
        "submission_025_xgboost_mlp_dropout.csv",
        fixed_metrics["macro_f1"],
        best_control,
    )
    result = {
        "experiment": "025",
        "model_seeds": [42],
        "configuration": "XGBoost 007 / MLP 020 dropout=0.1, mistura fixa 75/25, MLP 85 épocas",
        "control_configurations_reproduced": [
            "007: XGBoost 200 árvores, depth 5, lr 0.1, hist, seed 42, n_jobs=1",
            "020: MLP 64→32, Adam lr=0.0001, inverse_sqrt, dropout=0.1, época=85",
            "014/016: mistura fixa 75/25",
        ],
        "inner_mlp_best_epoch_diagnostic": _best_epoch(history)[0],
        "inner_macro_f1_diagnostic": inner_best,
        "fixed_metrics": fixed_metrics,
        "control_fixed_macro_f1": {"014": 0.4613, "016": 0.4580},
        "improved_best_control": fixed_metrics["macro_f1"] > best_control,
        "exploratory": fixed_metrics["macro_f1"] <= best_control,
        "total_seconds": time.perf_counter() - started,
        "inner_training_seconds": inner_seconds,
        "fixed_training_seconds": fixed_seconds,
        "export_training_seconds": export_seconds,
        "curves": str(chart.relative_to(ROOT)),
        "submission": csv_path,
        "prediction_differences": differences,
        "kaggle_score": None,
    }
    _record_result("025", result)
    return {"result": result, "histories": {"dropout_0.1": history}}


def _cluster_features(
    X_fit: pd.DataFrame,
    X_transform: pd.DataFrame,
    k: int,
    *,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], np.ndarray]:
    cluster_fit = X_fit.copy()
    cluster_transform = X_transform.copy()
    nominal_numeric_columns = [
        column for column in ("REFYEAR", "REFMONTH") if column in cluster_fit
    ]
    for column in nominal_numeric_columns:
        cluster_fit[column] = cluster_fit[column].astype("string")
        cluster_transform[column] = cluster_transform[column].astype("string")
    prep = build_preprocessor(cluster_fit, normalize_numeric=True)
    fit_matrix = prep.transform(cluster_fit)
    transform_matrix = prep.transform(cluster_transform)
    fit_matrix = fit_matrix.toarray() if hasattr(fit_matrix, "toarray") else fit_matrix
    transform_matrix = (
        transform_matrix.toarray()
        if hasattr(transform_matrix, "toarray")
        else transform_matrix
    )
    fit_matrix = np.asarray(fit_matrix, dtype=np.float32)
    transform_matrix = np.asarray(transform_matrix, dtype=np.float32)
    model = KMeans(
        n_clusters=k,
        n_init=3,
        max_iter=100,
        random_state=seed,
        algorithm="lloyd",
    )
    with threadpool_limits(limits=1):
        labels = model.fit_predict(fit_matrix)
        fit_distances = model.transform(fit_matrix).astype(np.float32)
        transform_distances = model.transform(transform_matrix).astype(np.float32)
    names = [f"cluster_distance_{index}" for index in range(k)]
    fit_augmented = X_fit.copy()
    transform_augmented = X_transform.copy()
    for index, name in enumerate(names):
        fit_augmented[name] = fit_distances[:, index]
        transform_augmented[name] = transform_distances[:, index]
    sample_size = min(300, len(fit_matrix))
    with threadpool_limits(limits=1):
        sil = float(
            silhouette_score(
                fit_matrix,
                labels,
                sample_size=sample_size,
                random_state=seed,
            )
        )
    details = {
        "k": k,
        "sizes": np.bincount(labels, minlength=k).astype(int).tolist(),
        "silhouette_sample": sil,
        "silhouette_sample_size": sample_size,
        "kmeans_n_init": 3,
        "kmeans_max_iter": 100,
        "thread_limit": 1,
        "categorical_year_month": nominal_numeric_columns,
        "age_range_description_only": None,
    }
    return fit_augmented, transform_augmented, details, labels


def run_027() -> dict[str, Any]:
    data = _load_data()
    started = time.perf_counter()
    baseline_model, baseline_inner_probs = _xgb_probabilities(
        data["X_inner_train"], data["y_inner_train"], data["X_inner_val"]
    )
    baseline_inner_metrics = _classification_metrics(
        data["y_inner_val"].to_numpy(dtype=np.int64),
        baseline_inner_probs.argmax(axis=1),
    )
    candidates = [
        {
            "k": None,
            "internal_metrics": baseline_inner_metrics,
            "cluster_details": None,
        }
    ]
    cluster_fit: dict[int, pd.DataFrame] = {}
    cluster_val: dict[int, pd.DataFrame] = {}
    cluster_details: dict[int, dict[str, Any]] = {}
    for k in (4, 8, 12):
        X_fit_aug, X_val_aug, details, cluster_labels = _cluster_features(
            data["X_inner_train"], data["X_inner_val"], k
        )
        details["age_range_description_only"] = (
            pd.DataFrame(
                {
                    "cluster": cluster_labels,
                    "AGE_RANGE": data["y_inner_train"].to_numpy(dtype=int),
                }
            )
            .groupby("cluster")["AGE_RANGE"]
            .agg(["mean", "median"])
            .round(2)
            .reset_index()
            .to_dict("records")
        )
        cluster_fit[k] = X_fit_aug
        cluster_val[k] = X_val_aug
        cluster_details[k] = details
        _, probabilities = _xgb_probabilities(
            X_fit_aug, data["y_inner_train"], X_val_aug
        )
        candidates.append(
            {
                "k": k,
                "internal_metrics": _classification_metrics(
                    data["y_inner_val"].to_numpy(dtype=np.int64),
                    probabilities.argmax(axis=1),
                ),
                "cluster_details": details,
            }
        )
    selected = max(candidates, key=lambda row: row["internal_metrics"]["macro_f1"])
    selected_k = selected["k"]

    xgb_fixed, baseline_fixed_probs = _xgb_probabilities(
        data["X_train"], data["y_train"], data["X_val"]
    )
    if selected_k is None:
        fixed_probs = baseline_fixed_probs
        details_fixed = None
    else:
        X_train_aug, X_val_aug, details_fixed, fixed_cluster_labels = _cluster_features(
            data["X_train"], data["X_val"], int(selected_k)
        )
        details_fixed["age_range_description_only"] = (
            pd.DataFrame(
                {
                    "cluster": fixed_cluster_labels,
                    "AGE_RANGE": data["y_train"].to_numpy(dtype=int),
                }
            )
            .groupby("cluster")["AGE_RANGE"]
            .agg(["mean", "median"])
            .round(2)
            .reset_index()
            .to_dict("records")
        )
        _, fixed_probs = _xgb_probabilities(
            X_train_aug, data["y_train"], X_val_aug
        )
    fixed_metrics = _classification_metrics(
        data["y_val"].to_numpy(dtype=np.int64), fixed_probs.argmax(axis=1)
    )
    baseline_fixed_metrics = _classification_metrics(
        data["y_val"].to_numpy(dtype=np.int64),
        baseline_fixed_probs.argmax(axis=1),
    )
    csv_path, differences = None, {}
    details_all = None
    if (
        selected_k is not None
        and fixed_metrics["macro_f1"] > baseline_fixed_metrics["macro_f1"]
    ):
        X_all_aug, X_test_aug, details_all, all_cluster_labels = _cluster_features(
            data["prepared"]["X_train"], data["prepared"]["X_test"], int(selected_k)
        )
        details_all["age_range_description_only"] = (
            pd.DataFrame(
                {
                    "cluster": all_cluster_labels,
                    "AGE_RANGE": data["prepared"]["y_train"].to_numpy(dtype=int),
                }
            )
            .groupby("cluster")["AGE_RANGE"]
            .agg(["mean", "median"])
            .round(2)
            .reset_index()
            .to_dict("records")
        )
        xgb_all, test_probs = _xgb_probabilities(
            X_all_aug, data["prepared"]["y_train"], X_test_aug
        )
        test_predictions = test_probs.argmax(axis=1)
        csv_path, differences = _export_if_useful(
            data,
            test_predictions,
            "027",
            "submission_027_xgboost_distancias_clusters.csv",
            fixed_metrics["macro_f1"],
            baseline_fixed_metrics["macro_f1"],
        )
    result = {
        "experiment": "027",
        "model_seeds": [42],
        "configuration": "XGBoost 007 com distâncias euclidianas a centróides KMeans; k=4/8/12",
        "representation": (
            "Pipeline central: variáveis numéricas contínuas/contagens (BM1, BM2, BMI, "
            "HH_ACT e HH_INACT) imputadas e normalizadas no treino; categóricas imputadas "
            "e one-hot. REFYEAR/REFMONTH são convertidos em categóricos para não impor "
            "distâncias quantitativas a códigos de calendário. "
            "KMeans apenas nessa representação; distâncias ao centróide acrescentadas às features XGBoost."
        ),
        "kmeans_configuration": "n_init=3, max_iter=100, random_state=42, threadpool limitado a 1; silhouette em amostra até 300",
        "excluded_from_clustering": ["AGE_RANGE", "PID", "HHID"],
        "control_configurations_reproduced": [
            "007: XGBoost 200 árvores, depth 5, lr 0.1, hist, seed 42, n_jobs=1"
        ],
        "internal_candidates": candidates,
        "selected": selected,
        "fixed_baseline_metrics": baseline_fixed_metrics,
        "fixed_metrics": fixed_metrics,
        "fixed_cluster_details": details_fixed,
        "full_train_cluster_details": details_all,
        "improved_best_control": (
            selected_k is not None
            and fixed_metrics["macro_f1"] > baseline_fixed_metrics["macro_f1"]
        ),
        "exploratory": (
            selected_k is None
            or fixed_metrics["macro_f1"] <= baseline_fixed_metrics["macro_f1"]
        ),
        "total_seconds": time.perf_counter() - started,
        "submission": csv_path,
        "prediction_differences": differences,
        "kaggle_score": None,
        "clinical_anomaly_interpretation": False,
    }
    _record_result("027", result)
    return {"result": result}


def run_experiment(experiment: str) -> dict[str, Any]:
    runners = {
        "022": run_022,
        "023": run_023,
        "024": run_024,
        "025": run_025,
        "026": _run_batchnorm_comparison,
        "027": run_027,
    }
    if experiment not in runners:
        raise ValueError(f"Experiência desconhecida: {experiment}")
    return runners[experiment]()


def verify_reproduced_predictions() -> dict[str, Any]:
    """Verify test predictions when an internally selected model is an old control."""
    data = _load_data()
    stored = json.loads(RESULTS_PATH.read_text())
    checks: dict[str, Any] = {}
    xgb_model, xgb_test = _xgb_probabilities(
        data["prepared"]["X_train"],
        data["prepared"]["y_train"],
        data["prepared"]["X_test"],
    )

    selected_022 = stored["022"]["selected"]
    _, X_all, X_test = fit_preprocessor(
        data["prepared"]["X_train"], data["prepared"]["X_test"]
    )
    seeds = (42,) if selected_022["mlp"] == "seed_42" else (42, 123, 2026)
    mlp_test_probabilities = []
    for seed in seeds:
        model, _, _ = _fit_mlp(
            X_all,
            data["prepared"]["y_train"].to_numpy(dtype=np.int64),
            93,
            0.0001,
            weighting="inverse",
            seed=seed,
        )
        mlp_test_probabilities.append(_predict_probabilities(model, X_test))
    mlp_test = np.mean(mlp_test_probabilities, axis=0)
    predictions_022 = (
        selected_022["xgboost_weight"] * xgb_test
        + selected_022["mlp_weight"] * mlp_test
    ).argmax(axis=1)
    row_reference = _official_row_reference(data)
    exported_022 = row_reference.copy()
    exported_022["AGE_RANGE"] = predictions_022
    compared_022 = {}
    for name in (
        "submission_019_ensemble_pesos.csv",
        "submission_016_xgboost_mlp015.csv",
    ):
        previous = pd.read_csv(ROOT / "submissions" / name)
        aligned = previous.set_index("RowId").reindex(exported_022["RowId"])
        compared_022[name] = int(
            (
                exported_022["AGE_RANGE"].to_numpy()
                != aligned["AGE_RANGE"].to_numpy()
            ).sum()
        )
    checks["022"] = {
        "configuration": stored["022"]["configuration"],
        "test_prediction_differences": compared_022,
        "duplicate_of_019": compared_022["submission_019_ensemble_pesos.csv"] == 0,
        "csv_created": stored["022"].get("submission"),
    }

    if stored["027"]["selected"]["k"] is None:
        predictions_027 = xgb_test.argmax(axis=1)
        exported_027 = row_reference.copy()
        exported_027["AGE_RANGE"] = predictions_027
        previous_007 = pd.read_csv(ROOT / "submissions" / "submission_007_xgboost.csv")
        aligned_007 = previous_007.set_index("RowId").reindex(exported_027["RowId"])
        differences = int(
            (
                exported_027["AGE_RANGE"].to_numpy()
                != aligned_007["AGE_RANGE"].to_numpy()
            ).sum()
        )
        checks["027"] = {
            "selected_control": "XGBoost 007 sem clustering",
            "test_prediction_differences_vs_007": differences,
            "duplicate_of_007": differences == 0,
            "csv_created": stored["027"].get("submission"),
        }

    for experiment, check in checks.items():
        stored[experiment]["reproduction_check"] = check
    RESULTS_PATH.write_text(json.dumps(stored, ensure_ascii=False, indent=2))
    return checks


def write_final_report(root: Path | None = None) -> Path:
    root = root or ROOT
    results = json.loads(RESULTS_PATH.read_text())
    lines = [
        "# Experiências 022–027",
        "",
        "Protocolo: preparação central e divisões estratificadas 80/20 + validação interna de 15% (`random_state=42`), com transformações ajustadas só no treino de cada fase. As escolhas usam exclusivamente a validação interna. A validação fixa é uma comparação de desenvolvimento, não um teste final independente. Os controlos e métricas por classe são registados nos resultados detalhados de cada notebook.",
        "",
        "Não foram consultadas pontuações Kaggle para escolher configurações. Pontuações novas ficam por preencher até eventual submissão voluntária.",
        "",
        "| Experiência | Alteração/configuração escolhida | Macro F1 fixa | Accuracy | F1/recall classes 1, 6, 13 | Melhora controlo? | Kaggle | CSV |",
        "|---|---|---:|---:|---|---|---:|---|",
    ]
    for experiment in ("022", "023", "024", "025", "026", "027"):
        if experiment not in results:
            continue
        result = results[experiment]
        metrics = result.get("fixed_metrics")
        if metrics is None:
            continue
        key_metrics = ", ".join(
            f"{row['classe']}: {row['F1']:.3f}/{row['recall']:.3f}"
            for row in metrics["key_classes"]
        )
        lines.append(
            f"| {experiment} | {result['configuration']} "
            f"(seed(s) {', '.join(map(str, result.get('model_seeds', [])))}) | "
            f"{metrics['macro_f1']:.4f} | "
            f"{metrics['accuracy']:.4f} | {key_metrics} | "
            f"{'Sim' if result['improved_best_control'] else 'Não; exploratória'} | "
            f"{result.get('kaggle_score') or '—'} | {result.get('submission') or '—'} |"
        )
    lines.extend(["", "## Métricas detalhadas por classe", ""])
    for experiment in ("022", "023", "024", "025", "026", "027"):
        if experiment not in results:
            continue
        result = results[experiment]
        metrics = result.get("fixed_metrics")
        if metrics is None:
            continue
        lines.extend(
            [
                f"### {experiment}",
                "",
                f"Configuração: {result['configuration']}. "
                f"Tempo total: {result.get('total_seconds', 0):.1f} s.",
                "",
                "| Classe | Faixa etária | F1 | Recall | Suporte |",
                "|---:|---|---:|---:|---:|",
            ]
        )
        lines.extend(
            f"| {row['classe']} | {row['escalão']} | {row['F1']:.4f} | "
            f"{row['recall']:.4f} | {row['support']} |"
            for row in metrics["per_class"]
        )
        if result.get("curves"):
            lines.extend(["", f"Curvas de perda e Macro F1: `{result['curves']}`."])
        lines.append("")

    lines.extend(["## Comparações, controlos e conclusões", ""])
    if "022" in results:
        result = results["022"]
        metrics = result["fixed_comparisons"]
        lines.extend(
            [
                "### 022 — média de sementes versus pesos",
                "",
                f"A validação interna escolheu `{result['selected']['mlp']}` com "
                f"mistura {result['selected']['xgboost_weight']:.0%}/"
                f"{result['selected']['mlp_weight']:.0%}; no conjunto fixo obteve "
                f"Macro F1 {result['fixed_metrics']['macro_f1']:.4f}.",
                "",
                "| Probabilidades MLP | Pesos XGB/MLP | Macro F1 fixa | Accuracy fixa |",
                "|---|---:|---:|---:|",
            ]
        )
        for key, mlp_name, weights in (
            ("seed_42_0.75_0.25", "Semente 42", "75/25"),
            ("seed_42_0.60_0.40", "Semente 42", "60/40"),
            ("mean_3_seeds_0.75_0.25", "Média de 3 sementes", "75/25"),
            ("mean_3_seeds_0.60_0.40", "Média de 3 sementes", "60/40"),
        ):
            row = metrics[key]
            lines.append(
                f"| {mlp_name} | {weights} | {row['macro_f1']:.4f} | "
                f"{row['accuracy']:.4f} |"
            )
        lines.extend(
            [
                "",
                "Seleção na validação interna:",
                "",
                "| Probabilidades MLP | Pesos XGB/MLP | Macro F1 interna | Accuracy interna |",
                "|---|---:|---:|---:|",
            ]
        )
        for candidate in result["internal_candidates"]:
            internal_metrics = candidate["internal_metrics"]
            lines.append(
                f"| {candidate['mlp']} | {candidate['xgboost_weight']:.0%}/"
                f"{candidate['mlp_weight']:.0%} | "
                f"{internal_metrics['macro_f1']:.4f} | "
                f"{internal_metrics['accuracy']:.4f} |"
            )
        effects = result["average_seed_effects"]
        lines.extend(
            [
                "",
                f"Na comparação fixa de desenvolvimento, a média melhora a mistura 75/25 "
                f"em {effects['same_weights_75_25_delta']:+.4f} e a 60/40 em "
                f"{effects['same_weights_60_40_delta']:+.4f}; mudar 75/25 para 60/40 "
                f"altera a Macro F1 em {effects['weight_change_single_seed_delta']:+.4f} "
                f"com uma semente e {effects['weight_change_mean_seed_delta']:+.4f} "
                "com a média. O melhor número fixo é a média de sementes com 75/25, "
                "mas não foi escolhido por esse conjunto e não supera o controlo 014. "
                "A seleção interna resultou na configuração 019; não há melhoria nova. "
                f"Verificação no conjunto de teste: {result.get('reproduction_check', {}).get('test_prediction_differences', {}).get('submission_019_ensemble_pesos.csv', 'a executar')} "
                "previsões diferentes face a 019; não se cria um CSV duplicado.",
                "",
            ]
        )
    if "023" in results:
        result = results["023"]
        lines.extend(
            [
                "### 023 — dropout",
                "",
                "| Dropout | Melhor época | Macro F1 interna | Tempo (s) |",
                "|---:|---:|---:|---:|",
                *[
                    f"| {candidate['dropout']:g} | {candidate['best_epoch']} | "
                    f"{candidate['internal_macro_f1']:.4f} | "
                    f"{candidate['training_seconds']:.1f} |"
                    for candidate in result["internal_candidates"]
                ],
                "",
                f"Dropout={result['selected']['dropout']:g}, época "
                f"{result['selected']['best_epoch']}: Macro F1 fixa "
                f"{result['fixed_metrics']['macro_f1']:.4f}, acima do 017 "
                f"({result['control_fixed_macro_f1']['017']:.4f}), mas abaixo do "
                f"020 ({result['control_fixed_macro_f1']['020']:.4f}); não é melhoria "
                "sobre o melhor controlo nem produz CSV.",
                "",
            ]
        )
    if "024" in results:
        result = results["024"]
        lines.extend(
            [
                "### 024 — SGD",
                "",
                "| Learning rate | Melhor época | Macro F1 interna | Tempo (s) |",
                "|---:|---:|---:|---:|",
                *[
                    f"| {candidate['learning_rate']:g} | {candidate['best_epoch']} | "
                    f"{candidate['internal_macro_f1']:.4f} | "
                    f"{candidate['training_seconds']:.1f} |"
                    for candidate in result["internal_candidates"]
                ],
                "",
                f"A taxa selecionada internamente foi {result['selected']['learning_rate']:g} "
                f"e época {result['selected']['best_epoch']}; Macro F1 fixa "
                f"{result['fixed_metrics']['macro_f1']:.4f}, face a 018="
                f"{result['control_fixed_macro_f1']['018']:.4f} e 021="
                f"{result['control_fixed_macro_f1']['021']:.4f}. Sem melhoria.",
                "",
            ]
        )
    if "025" in results:
        result = results["025"]
        lines.extend(
            [
                "### 025 — ensemble com dropout",
                "",
                f"Com a configuração MLP 020 reproduzida e mistura fixa 75/25, Macro F1 "
                f"{result['fixed_metrics']['macro_f1']:.4f}: +"
                f"{result['fixed_metrics']['macro_f1'] - result['control_fixed_macro_f1']['014']:.4f} "
                f"face a 014 e +{result['fixed_metrics']['macro_f1'] - result['control_fixed_macro_f1']['016']:.4f} "
                "face a 016. A diferença relativamente a 014 é pequena; a validação fixa "
                "é apenas comparação de desenvolvimento. O CSV é distinto e foi validado.",
                "",
            ]
        )
    if "026" in results:
        result = results["026"]
        difference = (
            result["fixed_metrics"]["macro_f1"]
            - result["fixed_control_metrics"]["macro_f1"]
        )
        lines.extend(
            [
                "### 026 — BatchNorm",
                "",
                "| Variante | Melhor época | Macro F1 interna | Tempo (s) |",
                "|---|---:|---:|---:|",
                *[
                    f"| {'com' if candidate['batch_norm'] else 'sem'} BatchNorm | "
                    f"{candidate['best_epoch']} | {candidate['internal_macro_f1']:.4f} | "
                    f"{candidate['training_seconds']:.1f} |"
                    for candidate in result["internal_candidates"]
                ],
                "",
                f"O controlo sem BatchNorm reproduziu a época 85="
                f"{'sim' if result['reproduces_020_epoch'] else 'não'} e a Macro F1 "
                f"fixa de 020="
                f"{'sim' if result['reproduces_020_fixed_macro_f1'] else 'não'}. "
                "Com BatchNorm, "
                f"época {result['selected']['best_epoch']} e Macro F1 fixa "
                f"{result['fixed_metrics']['macro_f1']:.4f}, diferença "
                f"{difference:+.4f} relativamente à mesma rede sem BatchNorm "
                f"({result['fixed_control_metrics']['macro_f1']:.4f}); não melhorou.",
                "",
            ]
        )
    if "027" in results:
        result = results["027"]
        lines.extend(
            [
                "### 027 — clustering",
                "",
                "A AGE_RANGE não entrou no ajuste ou nas distâncias; foi usada apenas "
                "depois para resumir a idade média/mediana por cluster. Os clusters não "
                "são interpretados como anomalias clínicas.",
                "",
                "| Variante | Macro F1 interna | Tamanhos | Silhouette (amostra) | AGE_RANGE médio/mediano (descrição) |",
                "|---|---:|---|---:|---|",
            ]
        )
        for candidate in result["internal_candidates"]:
            details = candidate.get("cluster_details")
            if details is None:
                lines.append(
                    f"| Sem clustering (007) | {candidate['internal_metrics']['macro_f1']:.4f} "
                    "| — | — | — |"
                )
            else:
                sizes = ", ".join(str(size) for size in details["sizes"])
                age_summary = "; ".join(
                    f"C{row['cluster']}={row['mean']}/{row['median']}"
                    for row in details["age_range_description_only"]
                )
                lines.append(
                    f"| k={details['k']} | {candidate['internal_metrics']['macro_f1']:.4f} "
                    f"| {sizes} | {details['silhouette_sample']:.4f} "
                    f"(n={details['silhouette_sample_size']}) | {age_summary} |"
                )
        lines.extend(
            [
                "",
                f"A seleção interna manteve o controlo sem clusters (`k=None`); Macro F1 "
                f"fixa {result['fixed_metrics']['macro_f1']:.4f}, igual ao XGBoost 007. "
                f"Verificação de teste face a 007: "
                f"{result.get('reproduction_check', {}).get('test_prediction_differences_vs_007', 'a executar')} "
                "previsões diferentes. Não foi criada submissão.",
                "",
            ]
        )

    available = [
        (experiment, results[experiment])
        for experiment in ("022", "023", "024", "025", "026", "027")
        if experiment in results
        and results[experiment].get("submission")
        and results[experiment].get("improved_best_control")
    ]
    available.sort(
        key=lambda item: item[1]["fixed_metrics"]["macro_f1"], reverse=True
    )
    recommendations: list[tuple[str, dict[str, Any]]] = []
    for experiment, result in available:
        current = pd.read_csv(root / result["submission"]).set_index("RowId")
        distinct = True
        for _, other in recommendations:
            previous = pd.read_csv(root / other["submission"]).set_index("RowId")
            previous = previous.reindex(current.index)
            if current["AGE_RANGE"].equals(previous["AGE_RANGE"]):
                distinct = False
                break
        if distinct:
            recommendations.append((experiment, result))
        if len(recommendations) == 3:
            break
    lines.extend(["## Candidatos locais para submissão", ""])
    if recommendations:
        for rank, (experiment, result) in enumerate(recommendations, start=1):
            lines.append(
                f"{rank}. `{result['submission']}` (experiência {experiment}; "
                f"Macro F1 fixa {result['fixed_metrics']['macro_f1']:.4f}; "
                "previsões distintas dos candidatos já recomendados)."
            )
    else:
        lines.append(
            "Não há CSV novo que tenha superado o respetivo controlo local; "
            "não se recomenda submeter ficheiros repetidos ou piores."
        )
    if len(recommendations) < 3:
        lines.append(
            f"Só {len(recommendations)} candidato(s) útil(eis) e distintos; "
            "não há três CSVs para recomendar."
        )
    lines.append("")
    lines.extend(["## Reprodução dos controlos", ""])
    for experiment in ("022", "023", "024", "025", "026", "027"):
        if experiment not in results:
            continue
        controls = results[experiment].get("control_configurations_reproduced", [])
        if controls:
            lines.append(f"- **{experiment}:** " + "; ".join(controls) + ".")
    lines.append("")
    lines.extend(
        [
            "## Submissões anteriores e pontuações",
            "",
            "As submissões anteriores foram preservadas. Pontuações fornecidas: 016=0,48; 019=0,48 (empate na precisão apresentada); 020=0,45; 021=0,43. Nenhuma pontuação Kaggle foi atribuída às experiências 022–027.",
            "",
            "Os CSVs novos só são exportados quando a validação fixa de desenvolvimento supera o melhor controlo local comparável. Cada CSV é relido e verificado; o número de previsões diferentes face a cada CSV anterior está em `prediction_differences` no resultado JSON.",
            "",
        ]
    )
    path = root / "reports" / "experiments_022_027.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
