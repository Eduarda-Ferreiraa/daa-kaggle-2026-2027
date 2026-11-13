"""Shared execution helpers for notebooks 016–018."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

from src.preparar_dados import (
    build_preprocessor,
    make_model_pipeline,
    make_stratified_split,
    preparar_dados,
)


SEED = 42
NUM_CLASSES = 16
CLASS_IDS = np.arange(NUM_CLASSES)
BATCH_SIZE = 128
MAX_EPOCHS = 100
AGE_BANDS = [
    "15–17 anos",
    "18–19 anos",
    "20–24 anos",
    "25–29 anos",
    "30–34 anos",
    "35–39 anos",
    "40–44 anos",
    "45–49 anos",
    "50–54 anos",
    "55–59 anos",
    "60–64 anos",
    "65–69 anos",
    "70–74 anos",
    "75–79 anos",
    "80–84 anos",
    "85+ anos",
]
KEY_CLASSES = {1: AGE_BANDS[1], 6: AGE_BANDS[6], 13: AGE_BANDS[13]}
RESULTS_PATH = Path(__file__).resolve().parents[1] / "reports" / ".experiments_016_018_results.json"

torch.set_num_threads(1)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_experiment_data() -> dict[str, Any]:
    root = repository_root()
    train_df = pd.read_csv(root / "notebooks" / "dataset_train.csv")
    test_df = pd.read_csv(root / "notebooks" / "dataset_test_X.csv")
    prepared = preparar_dados(train_df, test_df)
    X_train, X_val, y_train, y_val = make_stratified_split(
        prepared["X_train"], prepared["y_train"], test_size=0.2, random_state=SEED
    )
    X_inner_train, X_inner_val, y_inner_train, y_inner_val = train_test_split(
        X_train,
        y_train,
        test_size=0.15,
        random_state=SEED,
        stratify=y_train,
    )
    if len(X_train) != 8769 or len(X_val) != 2193 or len(X_inner_train) != 7453:
        raise ValueError("As dimensões das partições diferem do protocolo do notebook 015.")
    if not X_train.index.equals(y_train.index) or not X_val.index.equals(y_val.index):
        raise ValueError("Os índices X/y da validação fixa não estão alinhados.")
    if not X_inner_train.index.equals(y_inner_train.index) or not X_inner_val.index.equals(y_inner_val.index):
        raise ValueError("Os índices X/y da validação interna não estão alinhados.")
    if not X_inner_train.index.intersection(X_inner_val.index).empty:
        raise ValueError("Treino e validação internos não são disjuntos.")
    if not X_train.index.intersection(X_val.index).empty:
        raise ValueError("Treino e validação fixa não são disjuntos.")
    return {
        "root": root,
        "prepared": prepared,
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "X_inner_train": X_inner_train,
        "X_inner_val": X_inner_val,
        "y_inner_train": y_inner_train,
        "y_inner_val": y_inner_val,
    }


def to_dense_float32(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def class_weights(targets: np.ndarray, weighting: str = "inverse") -> torch.Tensor:
    labels = np.asarray(targets, dtype=np.int64)
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    if np.any(counts == 0):
        raise ValueError(
            f"Classes ausentes no conjunto usado para calcular pesos: "
            f"{np.flatnonzero(counts == 0).tolist()}"
        )
    if weighting == "inverse":
        weights = len(labels) / (NUM_CLASSES * counts)
    elif weighting == "inverse_sqrt":
        weights = 1 / np.sqrt(counts.astype(np.float64))
        weights /= weights.mean()
    else:
        raise ValueError(f"Tipo de ponderação desconhecido: {weighting}")
    return torch.tensor(weights, dtype=torch.float32)


class AgeRangeMLP(torch.nn.Module):
    def __init__(self, input_features: int, dropout: float = 0.0):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_features, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(32, NUM_CLASSES),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


def make_mlp(input_features: int, dropout: float = 0.0) -> AgeRangeMLP:
    torch.manual_seed(SEED)
    return AgeRangeMLP(input_features, dropout=dropout).to("cpu")


def reset_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def fit_preprocessor(
    X_fit: pd.DataFrame, X_transform: pd.DataFrame
) -> tuple[Any, np.ndarray, np.ndarray]:
    preprocessor = build_preprocessor(X_fit, normalize_numeric=True)
    return (
        preprocessor,
        to_dense_float32(preprocessor.transform(X_fit)),
        to_dense_float32(preprocessor.transform(X_transform)),
    )


def probabilities_from_logits(model: AgeRangeMLP, features: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(torch.from_numpy(features)), dim=1).cpu().numpy()


def align_probabilities(probabilities: np.ndarray, model_classes: np.ndarray) -> np.ndarray:
    aligned = np.zeros((probabilities.shape[0], NUM_CLASSES), dtype=np.float64)
    for column, class_id in enumerate(model_classes):
        class_id = int(class_id)
        if class_id not in CLASS_IDS:
            raise ValueError(f"Classe inesperada no modelo: {class_id}")
        aligned[:, class_id] = probabilities[:, column]
    if not np.allclose(aligned.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("As probabilidades alinhadas não somam 1.")
    return aligned


def make_xgb() -> Any:
    return make_model_pipeline(
        XGBClassifier(
            objective="multi:softprob",
            num_class=NUM_CLASSES,
            tree_method="hist",
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=SEED,
            n_jobs=1,
            verbosity=0,
        ),
        normalize_numeric=False,
    )


def _macro_f1(targets: np.ndarray, predictions: np.ndarray) -> float:
    return float(
        f1_score(
            targets,
            predictions,
            labels=CLASS_IDS,
            average="macro",
            zero_division=0,
        )
    )


def _classification_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    report = classification_report(
        targets,
        predictions,
        labels=CLASS_IDS,
        target_names=[str(i) for i in CLASS_IDS],
        output_dict=True,
        zero_division=0,
    )
    per_class = [
        {
            "classe": int(class_id),
            "escalão": AGE_BANDS[int(class_id)],
            "F1": float(report[str(class_id)]["f1-score"]),
            "recall": float(report[str(class_id)]["recall"]),
            "support": int(report[str(class_id)]["support"]),
        }
        for class_id in CLASS_IDS
    ]
    return {
        "macro_f1": _macro_f1(targets, predictions),
        "accuracy": float(accuracy_score(targets, predictions)),
        "per_class": per_class,
        "key_classes": [row for row in per_class if row["classe"] in KEY_CLASSES],
    }


def _fit_epochs(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    epochs: int,
    learning_rate: float,
    weighting: str = "inverse",
    optimizer_name: str = "Adam",
    momentum: float = 0.9,
    X_monitor: np.ndarray | None = None,
    y_monitor: np.ndarray | None = None,
    record_history: bool = False,
    dropout: float = 0.0,
) -> tuple[AgeRangeMLP, dict[str, list[float]], float]:
    reset_seed()
    model = make_mlp(X_fit.shape[1], dropout=dropout)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights(y_fit, weighting))
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=0
        )
    else:
        raise ValueError(f"Otimizador desconhecido: {optimizer_name}")

    dataset = TensorDataset(torch.from_numpy(X_fit), torch.from_numpy(y_fit))
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    history: dict[str, list[float]] = {
        "epoch": [],
        "train_loss": [],
        "train_macro_f1": [],
        "validation_macro_f1": [],
    }
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        for features, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), targets)
            loss.backward()
            optimizer.step()

        if record_history:
            model.eval()
            with torch.no_grad():
                train_logits = model(torch.from_numpy(X_fit))
                train_predictions = train_logits.argmax(dim=1).numpy()
                train_loss = criterion(train_logits, torch.from_numpy(y_fit)).item()
                history["epoch"].append(epoch)
                history["train_loss"].append(float(train_loss))
                history["train_macro_f1"].append(_macro_f1(y_fit, train_predictions))
                if X_monitor is not None and y_monitor is not None:
                    monitor_logits = model(torch.from_numpy(X_monitor))
                    history["validation_macro_f1"].append(
                        _macro_f1(y_monitor, monitor_logits.argmax(dim=1).numpy())
                    )
    return model, history, time.perf_counter() - start


def _best_epoch(history: dict[str, list[float]]) -> tuple[int, float]:
    if not history["validation_macro_f1"]:
        raise ValueError("Não há Macro F1 de validação interna para selecionar a época.")
    best_index = int(np.argmax(history["validation_macro_f1"]))
    return history["epoch"][best_index], history["validation_macro_f1"][best_index]


def _fit_mlp_for_fixed_validation(
    data: dict[str, Any],
    epochs: int,
    learning_rate: float,
    weighting: str = "inverse",
    optimizer_name: str = "Adam",
    momentum: float = 0.9,
    dropout: float = 0.0,
) -> dict[str, Any]:
    y_train = data["y_train"].to_numpy(dtype=np.int64)
    y_val = data["y_val"].to_numpy(dtype=np.int64)
    _, X_train, X_val = fit_preprocessor(data["X_train"], data["X_val"])
    model, _, seconds = _fit_epochs(
        X_train,
        y_train,
        epochs,
        learning_rate,
        weighting,
        optimizer_name,
        momentum,
        dropout=dropout,
    )
    probabilities = probabilities_from_logits(model, X_val)
    predictions = probabilities.argmax(axis=1)
    return {
        "model": model,
        "probabilities": probabilities,
        "predictions": predictions,
        "metrics": _classification_metrics(y_val, predictions),
        "training_seconds": seconds,
    }


def _official_row_reference(data: dict[str, Any]) -> pd.DataFrame:
    """Load the official ordered RowId reference, with a validated-export fallback."""
    root = data["root"]
    example_candidates = [
        root / "example_submission_10.csv",
        root / "notebooks" / "example_submission_10.csv",
        Path.home() / ".Trash" / "example_submission_10 21.31.28.csv",
    ]
    example_path = next((path for path in example_candidates if path.is_file()), None)
    if example_path is not None:
        example = pd.read_csv(example_path)
        if not {"RowId", "AGE_RANGE"}.issubset(example.columns):
            raise ValueError(f"O exemplo oficial tem colunas inesperadas: {example.columns.tolist()}.")
        row_reference = example[["RowId"]].copy()
        if len(row_reference) != len(data["prepared"]["X_test"]):
            raise ValueError("O exemplo oficial não corresponde ao número de linhas do conjunto de teste.")
        if row_reference.RowId.isna().any() or not row_reference.RowId.is_unique:
            raise ValueError("Os RowId do exemplo oficial estão ausentes ou duplicados.")
        return row_reference

    reference_path = root / "submissions" / "submission_014_xgboost_mlp.csv"
    if not reference_path.is_file():
        raise FileNotFoundError(
            "Não encontrei o exemplo oficial nem o CSV 014 validado para recuperar "
            "a ordem original de RowId."
        )
    previous = pd.read_csv(reference_path)
    if previous.columns.tolist() != ["RowId", "AGE_RANGE"]:
        raise ValueError("O CSV 014 não tem o cabeçalho esperado.")
    row_reference = previous[["RowId"]].copy()
    if len(row_reference) != len(data["prepared"]["X_test"]):
        raise ValueError("O CSV 014 não corresponde ao número de linhas do conjunto de teste.")
    if row_reference.RowId.isna().any() or not row_reference.RowId.is_unique:
        raise ValueError("Os RowId do CSV de referência estão ausentes ou duplicados.")
    return row_reference


def _export_predictions(
    data: dict[str, Any],
    predictions: np.ndarray,
    filename: str,
) -> Path:
    root = data["root"]
    row_reference = _official_row_reference(data)
    if len(predictions) != len(row_reference):
        raise ValueError("O número de previsões não corresponde aos RowId oficiais.")
    path = root / "submissions" / filename
    if path.exists():
        raise FileExistsError(f"Não vou substituir uma submissão existente: {path}")
    submission = row_reference.copy()
    submission["AGE_RANGE"] = np.asarray(predictions, dtype=np.int64)
    if submission.columns.tolist() != ["RowId", "AGE_RANGE"]:
        raise ValueError("Cabeçalho de submissão inválido antes de guardar.")
    if not pd.api.types.is_integer_dtype(submission["AGE_RANGE"]):
        raise TypeError("AGE_RANGE tem de ser uma classe inteira.")
    if not submission["AGE_RANGE"].between(0, 15).all():
        raise ValueError("AGE_RANGE contém classes fora do intervalo [0, 15].")
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)
    check = pd.read_csv(path)
    if check.shape != (3655, 2):
        raise ValueError(f"Dimensão inválida no CSV reaberto: {check.shape}.")
    if check.columns.tolist() != ["RowId", "AGE_RANGE"]:
        raise ValueError("Cabeçalho inválido ou índice adicional no CSV guardado.")
    if not check.RowId.is_unique or not check.RowId.equals(row_reference.RowId):
        raise ValueError("RowId duplicados ou fora da ordem oficial no CSV guardado.")
    if not pd.api.types.is_integer_dtype(check["AGE_RANGE"]) or not check["AGE_RANGE"].between(0, 15).all():
        raise ValueError("As classes do CSV guardado não são inteiras em [0, 15].")
    return path


def _record_result(name: str, result: dict[str, Any]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    stored = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.is_file() else {}
    stored[name] = result
    RESULTS_PATH.write_text(json.dumps(stored, ensure_ascii=False, indent=2))


def run_016() -> dict[str, Any]:
    data = load_experiment_data()
    xgb_inner = make_xgb().fit(
        data["X_inner_train"], data["y_inner_train"]
    )
    xgb_inner_model = xgb_inner.named_steps["model"]
    xgb_inner_probabilities = align_probabilities(
        xgb_inner.predict_proba(data["X_inner_val"]), xgb_inner_model.classes_
    )
    _, X_inner_train, X_inner_val = fit_preprocessor(
        data["X_inner_train"], data["X_inner_val"]
    )
    y_inner_train = data["y_inner_train"].to_numpy(dtype=np.int64)
    mlp_inner, _, inner_seconds = _fit_epochs(
        X_inner_train, y_inner_train, 93, 0.0001, weighting="inverse"
    )
    mlp_inner_probabilities = probabilities_from_logits(mlp_inner, X_inner_val)
    inner_mix = 0.75 * xgb_inner_probabilities + 0.25 * mlp_inner_probabilities
    inner_metrics = _classification_metrics(
        data["y_inner_val"].to_numpy(dtype=np.int64), inner_mix.argmax(axis=1)
    )

    xgb_val = make_xgb().fit(data["X_train"], data["y_train"])
    xgb_val_model = xgb_val.named_steps["model"]
    xgb_val_probabilities = align_probabilities(
        xgb_val.predict_proba(data["X_val"]), xgb_val_model.classes_
    )
    mlp_val = _fit_mlp_for_fixed_validation(data, 93, 0.0001)
    mixed_val_probabilities = 0.75 * xgb_val_probabilities + 0.25 * mlp_val["probabilities"]
    y_val = data["y_val"].to_numpy(dtype=np.int64)
    mixed_val_predictions = mixed_val_probabilities.argmax(axis=1)
    metrics = _classification_metrics(y_val, mixed_val_predictions)

    all_x = data["prepared"]["X_train"]
    all_y = data["prepared"]["y_train"]
    all_test = data["prepared"]["X_test"]
    xgb_all = make_xgb().fit(all_x, all_y)
    xgb_all_model = xgb_all.named_steps["model"]
    xgb_test_probabilities = align_probabilities(
        xgb_all.predict_proba(all_test), xgb_all_model.classes_
    )
    _, X_all, X_test = fit_preprocessor(all_x, all_test)
    mlp_all, _, export_seconds = _fit_epochs(
        X_all,
        all_y.to_numpy(dtype=np.int64),
        93,
        0.0001,
        weighting="inverse",
    )
    test_probabilities = 0.75 * xgb_test_probabilities + 0.25 * probabilities_from_logits(
        mlp_all, X_test
    )
    path = _export_predictions(
        data, test_probabilities.argmax(axis=1), "submission_016_xgboost_mlp015.csv"
    )
    result = {
        "configuration": "75% XGBoost 007 + 25% MLP 015; MLP Adam lr=0.0001, 93 épocas",
        "metrics": metrics,
        "inner_metrics": inner_metrics,
        "xgboost_metrics": _classification_metrics(
            y_val, xgb_val_probabilities.argmax(axis=1)
        ),
        "mlp015_metrics": mlp_val["metrics"],
        "reference_macro_f1": 0.4613,
        "reference_kaggle": 0.47,
        "exploratory": metrics["macro_f1"] < 0.4613,
        "submission": str(path.relative_to(data["root"])),
        "training_seconds": {
            "mlp_inner": inner_seconds,
            "mlp_80_percent": mlp_val["training_seconds"],
            "mlp_export": export_seconds,
        },
    }
    _record_result("016", result)
    return result


def run_017() -> dict[str, Any]:
    data = load_experiment_data()
    _, X_inner_train, X_inner_val = fit_preprocessor(
        data["X_inner_train"], data["X_inner_val"]
    )
    y_inner_train = data["y_inner_train"].to_numpy(dtype=np.int64)
    y_inner_val = data["y_inner_val"].to_numpy(dtype=np.int64)
    histories: dict[str, dict[str, list[float]]] = {}
    selections: dict[str, dict[str, Any]] = {}
    for weighting, label in [
        ("inverse", "inverso"),
        ("inverse_sqrt", "inverso_raiz"),
    ]:
        model, history, seconds = _fit_epochs(
            X_inner_train,
            y_inner_train,
            MAX_EPOCHS,
            0.0001,
            weighting=weighting,
            X_monitor=X_inner_val,
            y_monitor=y_inner_val,
            record_history=True,
        )
        best_epoch, best_f1 = _best_epoch(history)
        histories[label] = history
        selections[label] = {
            "best_epoch": best_epoch,
            "best_internal_macro_f1": best_f1,
            "training_seconds": seconds,
        }

    control = _fit_mlp_for_fixed_validation(data, selections["inverso"]["best_epoch"], 0.0001)
    soft = _fit_mlp_for_fixed_validation(
        data,
        selections["inverso_raiz"]["best_epoch"],
        0.0001,
        weighting="inverse_sqrt",
    )
    all_x = data["prepared"]["X_train"]
    all_y = data["prepared"]["y_train"]
    all_test = data["prepared"]["X_test"]
    _, X_all, X_test = fit_preprocessor(all_x, all_test)
    mlp_all, _, export_seconds = _fit_epochs(
        X_all,
        all_y.to_numpy(dtype=np.int64),
        selections["inverso_raiz"]["best_epoch"],
        0.0001,
        weighting="inverse_sqrt",
    )
    path = _export_predictions(
        data,
        probabilities_from_logits(mlp_all, X_test).argmax(axis=1),
        "submission_017_mlp_ponderacao_suave.csv",
    )
    result = {
        "configuration": (
            "MLP 64→32, ReLU, Adam lr=0.0001, batch_size=128; "
            "exportação com pesos normalizados 1/sqrt(frequência)"
        ),
        "selections": selections,
        "control_metrics": control["metrics"],
        "soft_metrics": soft["metrics"],
        "reference_macro_f1": 0.4224,
        "reference_kaggle": 0.44,
        "exploratory": soft["metrics"]["macro_f1"] < 0.4224,
        "submission": str(path.relative_to(data["root"])),
        "export_training_seconds": export_seconds,
    }
    _record_result("017", result)
    return {"result": result, "histories": histories}


def run_018() -> dict[str, Any]:
    data = load_experiment_data()
    _, X_inner_train, X_inner_val = fit_preprocessor(
        data["X_inner_train"], data["X_inner_val"]
    )
    y_inner_train = data["y_inner_train"].to_numpy(dtype=np.int64)
    y_inner_val = data["y_inner_val"].to_numpy(dtype=np.int64)
    configs = [
        ("Adam", 0.0001, 0.0),
        ("Adam", 0.001, 0.0),
        ("SGD", 0.001, 0.9),
        ("SGD", 0.01, 0.9),
        ("SGD", 0.05, 0.9),
    ]
    histories: dict[str, dict[str, list[float]]] = {}
    rows = []
    for optimizer_name, learning_rate, momentum in configs:
        label = f"{optimizer_name}_lr_{learning_rate:g}"
        _, history, seconds = _fit_epochs(
            X_inner_train,
            y_inner_train,
            MAX_EPOCHS,
            learning_rate,
            weighting="inverse",
            optimizer_name=optimizer_name,
            momentum=momentum,
            X_monitor=X_inner_val,
            y_monitor=y_inner_val,
            record_history=True,
        )
        best_epoch, best_f1 = _best_epoch(history)
        histories[label] = history
        rows.append(
            {
                "optimizer": optimizer_name,
                "learning_rate": learning_rate,
                "momentum": momentum if optimizer_name == "SGD" else None,
                "best_epoch": best_epoch,
                "internal_macro_f1": best_f1,
                "training_seconds": seconds,
                "history_key": label,
            }
        )
    search_table = pd.DataFrame(rows)
    selected_rows = {}
    fixed_results = {}
    for optimizer_name in ("Adam", "SGD"):
        candidates = search_table[search_table.optimizer == optimizer_name]
        selected = candidates.sort_values(
            ["internal_macro_f1", "learning_rate"],
            ascending=[False, True],
            kind="stable",
        ).iloc[0]
        selected_rows[optimizer_name] = selected.to_dict()
        fixed_results[optimizer_name] = _fit_mlp_for_fixed_validation(
            data,
            int(selected.best_epoch),
            float(selected.learning_rate),
            weighting="inverse",
            optimizer_name=optimizer_name,
            momentum=0.9,
        )

    sgd = selected_rows["SGD"]
    all_x = data["prepared"]["X_train"]
    all_y = data["prepared"]["y_train"]
    all_test = data["prepared"]["X_test"]
    _, X_all, X_test = fit_preprocessor(all_x, all_test)
    sgd_all, _, export_seconds = _fit_epochs(
        X_all,
        all_y.to_numpy(dtype=np.int64),
        int(sgd["best_epoch"]),
        float(sgd["learning_rate"]),
        weighting="inverse",
        optimizer_name="SGD",
        momentum=0.9,
    )
    path = _export_predictions(
        data,
        probabilities_from_logits(sgd_all, X_test).argmax(axis=1),
        "submission_018_mlp_sgd.csv",
    )
    result = {
        "configuration": (
            f"MLP 64→32, ReLU, SGD lr={float(sgd['learning_rate']):g}, "
            f"momentum=0.9, batch_size=128, época={int(sgd['best_epoch'])}"
        ),
        "search_rows": rows,
        "selected": selected_rows,
        "adam_metrics": fixed_results["Adam"]["metrics"],
        "sgd_metrics": fixed_results["SGD"]["metrics"],
        "reference_macro_f1": 0.4224,
        "reference_kaggle": 0.44,
        "exploratory": fixed_results["SGD"]["metrics"]["macro_f1"] < 0.4224,
        "submission": str(path.relative_to(data["root"])),
        "export_training_seconds": export_seconds,
    }
    _record_result("018", result)
    _write_final_report(data["root"])
    return {"result": result, "histories": histories, "search_table": search_table}


def compare_submissions(root: Path | None = None) -> pd.DataFrame:
    root = root or repository_root()
    paths = sorted((root / "submissions").glob("*.csv"))
    frames: dict[str, pd.DataFrame] = {}
    for path in paths:
        frame = pd.read_csv(path)
        if frame.columns.tolist() != ["RowId", "AGE_RANGE"]:
            raise ValueError(f"Cabeçalho inválido em {path}.")
        if frame.RowId.isna().any() or not frame.RowId.is_unique:
            raise ValueError(f"RowId ausente ou duplicado em {path}.")
        if not pd.api.types.is_integer_dtype(frame.AGE_RANGE) or not frame.AGE_RANGE.between(0, 15).all():
            raise ValueError(f"Classes inválidas em {path}.")
        frames[path.stem] = frame.set_index("RowId").sort_index()

    new_names = [
        "submission_016_xgboost_mlp015",
        "submission_017_mlp_ponderacao_suave",
        "submission_018_mlp_sgd",
    ]
    rows = []
    old_names = [name for name in frames if name not in new_names]
    for index, left_name in enumerate(new_names):
        if left_name not in frames:
            raise FileNotFoundError(f"Falta a submissão nova: {left_name}.csv")
        right_names = old_names + new_names[index + 1 :]
        for right_name in right_names:
            left = frames[left_name]
            right = frames[right_name]
            if not left.index.equals(right.index):
                raise ValueError(f"RowId incompatível entre {left_name} e {right_name}.")
            differences = int((left.AGE_RANGE.to_numpy() != right.AGE_RANGE.to_numpy()).sum())
            rows.append(
                {
                    "csv_a": f"{left_name}.csv",
                    "csv_b": f"{right_name}.csv",
                    "previsoes_diferentes": differences,
                    "duplicado": differences == 0,
                }
            )
    return pd.DataFrame(rows)


def _write_final_report(root: Path) -> None:
    if not RESULTS_PATH.is_file():
        return
    results = json.loads(RESULTS_PATH.read_text())
    if not all(name in results for name in ("016", "017", "018")):
        return
    comparisons = compare_submissions(root)
    lines = [
        "# Experiências 016–018",
        "",
        "## Protocolo comum",
        "",
        "Divisão estratificada fixa 80/20 (`random_state=42`) e validação interna estratificada de 15% do treino (`random_state=42`), exatamente como no 015. Imputação/normalização e pesos de classe foram ajustados apenas nas linhas de treino de cada fase. A escolha de épocas/taxas usou exclusivamente Macro F1 interna; a validação fixa foi usada apenas para avaliação final.",
        "",
        "A cópia do exemplo oficial não se encontrava no projeto. Para preservar a ordem oficial, os três CSVs reutilizam os `RowId` únicos e ordenados do CSV 014 existente, comparados com os demais CSVs disponíveis.",
        "",
        "## Resultados na validação fixa",
        "",
        "| Experiência | Configuração exportada | Macro F1 | Accuracy | Referência fixa | Kaggle referência | Estado | CSV |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    metric_map = {
        "016": results["016"]["metrics"],
        "017": results["017"]["soft_metrics"],
        "018": results["018"]["sgd_metrics"],
    }
    reference_map = {"016": "0.4613", "017": "0.4224", "018": "0.4224"}
    kaggle_map = {"016": "0.48", "017": "0.46", "018": "0.44"}
    for number in ("016", "017", "018"):
        item = results[number]
        metrics = metric_map[number]
        status = "Exploratória" if item["exploratory"] else "Acima/igual à referência"
        lines.append(
            f"| {number} | {item['configuration']} | {metrics['macro_f1']:.4f} | "
            f"{metrics['accuracy']:.4f} | {reference_map[number]} | {kaggle_map[number]} | "
            f"{status} | `{item['submission']}` |"
        )
    lines.extend(["", "## Métricas por escalão", ""])
    for number in ("016", "017", "018"):
        lines.extend([f"### {number}", "", "| Classe | Escalão | F1 | Recall | Suporte |", "|---:|---|---:|---:|---:|"])
        for row in metric_map[number]["per_class"]:
            lines.append(
                f"| {row['classe']} | {row['escalão']} | {row['F1']:.4f} | "
                f"{row['recall']:.4f} | {row['support']} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Comparação de previsões por RowId",
            "",
            "| CSV A | CSV B | Previsões diferentes | Duplicado |",
            "|---|---|---:|---|",
        ]
    )
    for row in comparisons.to_dict("records"):
        lines.append(
            f"| `{row['csv_a']}` | `{row['csv_b']}` | {row['previsoes_diferentes']} | "
            f"{'Sim' if row['duplicado'] else 'Não'} |"
        )
    lines.extend(
        [
            "",
            "## Conclusão",
            "",
            "- 016: compara diretamente a substituição do componente MLP 012 pelo MLP 015; a conclusão considera a diferença face à referência fixa do 014 (0.4613; Kaggle 0.47).",
            "- 017: compara 1/frequência com 1/√frequência e reporta o efeito nos escalões 18–19, 40–44 e 75–79; a referência é o 015 (0.4224; Kaggle 0.44).",
            "- 018: escolhe Adam e SGD pela Macro F1 interna, compara ambos na validação fixa e exporta sempre o melhor SGD interno, mesmo que Adam vença a validação fixa.",
            "",
            "Pontuação Kaggle 016: ____",
            "",
            "Pontuação Kaggle 017: ____",
            "",
            "Pontuação Kaggle 018: ____",
            "",
        ]
    )
    (root / "reports" / "experiments_016_018.md").write_text("\n".join(lines))
    RESULTS_PATH.unlink()
