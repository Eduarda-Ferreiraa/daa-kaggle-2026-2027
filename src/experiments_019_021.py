"""Shared execution helpers for notebooks 019–021."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.experiments_016_018 import (
    MAX_EPOCHS,
    _best_epoch,
    _classification_metrics,
    _export_predictions,
    _fit_epochs,
    _fit_mlp_for_fixed_validation,
    _official_row_reference,
    _record_result,
    _write_final_report,
    align_probabilities,
    fit_preprocessor,
    load_experiment_data,
    make_xgb,
    probabilities_from_logits,
)


def _xgb_inner_probabilities(data: dict[str, Any]) -> np.ndarray:
    model = make_xgb().fit(data["X_inner_train"], data["y_inner_train"])
    return align_probabilities(
        model.predict_proba(data["X_inner_val"]),
        model.named_steps["model"].classes_,
    )


def _submission_comparison(
    root: Path, new_path: Path, other_paths: list[Path] | None = None
) -> pd.DataFrame:
    """Compare one new submission against all available submissions by RowId."""
    row_reference = _official_row_reference({"root": root, "prepared": {"X_test": pd.DataFrame(index=range(3655))}})
    new_frame = pd.read_csv(new_path)
    if new_frame.columns.tolist() != ["RowId", "AGE_RANGE"]:
        raise ValueError(f"Cabeçalho inválido em {new_path}.")
    if not new_frame.RowId.equals(row_reference.RowId):
        raise ValueError(f"RowId divergentes da referência oficial em {new_path}.")

    comparison_paths = other_paths or sorted((root / "submissions").glob("*.csv"))
    rows = []
    for path in comparison_paths:
        if path.resolve() == new_path.resolve():
            continue
        frame = pd.read_csv(path)
        if frame.columns.tolist() != ["RowId", "AGE_RANGE"]:
            raise ValueError(f"Cabeçalho inválido em {path}.")
        if frame.RowId.isna().any() or not frame.RowId.is_unique:
            raise ValueError(f"RowId ausente ou duplicado em {path}.")
        if not pd.api.types.is_integer_dtype(frame.AGE_RANGE) or not frame.AGE_RANGE.between(0, 15).all():
            raise ValueError(f"Classes inválidas em {path}.")
        aligned = frame.set_index("RowId").reindex(new_frame.RowId)
        if aligned.AGE_RANGE.isna().any():
            raise ValueError(f"RowId incompatíveis entre {new_path} e {path}.")
        differences = int(
            (new_frame.AGE_RANGE.to_numpy() != aligned.AGE_RANGE.to_numpy()).sum()
        )
        rows.append(
            {
                "csv_a": new_path.name,
                "csv_b": path.name,
                "previsoes_diferentes": differences,
                "duplicado": differences == 0,
            }
        )
    return pd.DataFrame(rows)


def compare_new_submissions(root: Path | None = None) -> pd.DataFrame:
    """Compare submissions 019–021 with each other and every earlier CSV."""
    root = root or Path(__file__).resolve().parents[1]
    new_names = {
        "submission_019_ensemble_pesos.csv",
        "submission_020_mlp_dropout.csv",
        "submission_021_mlp_sgd_ponderacao_suave.csv",
    }
    new_paths = [root / "submissions" / name for name in sorted(new_names)]
    missing = [path for path in new_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Faltam submissões para comparar: {missing}.")

    all_paths = sorted((root / "submissions").glob("*.csv"))
    new_order = {name: index for index, name in enumerate(sorted(new_names))}
    rows = []
    for new_path in new_paths:
        eligible_paths = [
            path
            for path in all_paths
            if path.name not in new_names
            or new_order[path.name] <= new_order[new_path.name]
        ]
        rows.extend(
            _submission_comparison(root, new_path, eligible_paths).to_dict("records")
        )
    return pd.DataFrame(rows)


def run_019() -> dict[str, Any]:
    data = load_experiment_data()
    xgb_probabilities = _xgb_inner_probabilities(data)
    _, X_inner_train, X_inner_val = fit_preprocessor(
        data["X_inner_train"], data["X_inner_val"]
    )
    mlp, _, inner_training_seconds = _fit_epochs(
        X_inner_train,
        data["y_inner_train"].to_numpy(dtype=np.int64),
        93,
        0.0001,
        weighting="inverse",
    )
    mlp_probabilities = probabilities_from_logits(mlp, X_inner_val)
    y_inner_val = data["y_inner_val"].to_numpy(dtype=np.int64)
    weights = [(0.60, 0.40), (0.70, 0.30), (0.75, 0.25), (0.80, 0.20), (0.85, 0.15)]
    candidates = []
    for xgb_weight, mlp_weight in weights:
        predictions = (xgb_weight * xgb_probabilities + mlp_weight * mlp_probabilities).argmax(axis=1)
        metrics = _classification_metrics(y_inner_val, predictions)
        candidates.append(
            {
                "xgboost_weight": xgb_weight,
                "mlp_weight": mlp_weight,
                "internal_metrics": metrics,
            }
        )
    selected_index = int(
        np.argmax([row["internal_metrics"]["macro_f1"] for row in candidates])
    )
    selected = candidates[selected_index]
    selected_xgb = selected["xgboost_weight"]
    selected_mlp = selected["mlp_weight"]

    xgb_validation = make_xgb().fit(data["X_train"], data["y_train"])
    xgb_val_probabilities = align_probabilities(
        xgb_validation.predict_proba(data["X_val"]),
        xgb_validation.named_steps["model"].classes_,
    )
    mlp_validation = _fit_mlp_for_fixed_validation(data, 93, 0.0001, weighting="inverse")
    fixed_predictions = (
        selected_xgb * xgb_val_probabilities
        + selected_mlp * mlp_validation["probabilities"]
    ).argmax(axis=1)
    fixed_metrics = _classification_metrics(data["y_val"].to_numpy(dtype=np.int64), fixed_predictions)

    all_x, all_y, all_test = (
        data["prepared"]["X_train"],
        data["prepared"]["y_train"],
        data["prepared"]["X_test"],
    )
    xgb_all = make_xgb().fit(all_x, all_y)
    xgb_test_probabilities = align_probabilities(
        xgb_all.predict_proba(all_test), xgb_all.named_steps["model"].classes_
    )
    _, X_all, X_test = fit_preprocessor(all_x, all_test)
    mlp_all, _, export_seconds = _fit_epochs(
        X_all,
        all_y.to_numpy(dtype=np.int64),
        93,
        0.0001,
        weighting="inverse",
    )
    test_probabilities = (
        selected_xgb * xgb_test_probabilities
        + selected_mlp * probabilities_from_logits(mlp_all, X_test)
    )
    path = _export_predictions(
        data, test_probabilities.argmax(axis=1), "submission_019_ensemble_pesos.csv"
    )
    previous_path = data["root"] / "submissions" / "submission_016_xgboost_mlp015.csv"
    previous = pd.read_csv(previous_path)
    exported = pd.read_csv(path)
    different_from_016 = int((exported.AGE_RANGE != previous.AGE_RANGE).sum())
    repeats_016_config = (selected_xgb, selected_mlp) == (0.75, 0.25)
    result = {
        "configuration": (
            f"{selected_xgb:.0%} XGBoost 007 + {selected_mlp:.0%} MLP 015; "
            "MLP Adam lr=0.0001, 93 épocas"
        ),
        "candidates": candidates,
        "selected": selected,
        "fixed_metrics": fixed_metrics,
        "reference_macro_f1": 0.4580,
        "reference_kaggle": 0.48,
        "difference_from_016_fixed_macro_f1": fixed_metrics["macro_f1"] - 0.4580,
        "repeats_016_configuration": repeats_016_config,
        "different_predictions_from_016": different_from_016,
        "duplicate_016_predictions": different_from_016 == 0,
        "submission": str(path.relative_to(data["root"])),
        "inner_mlp_training_seconds": inner_training_seconds,
        "export_mlp_training_seconds": export_seconds,
    }
    _record_result("019", result)
    return result


def run_020() -> dict[str, Any]:
    data = load_experiment_data()
    _, X_inner_train, X_inner_val = fit_preprocessor(
        data["X_inner_train"], data["X_inner_val"]
    )
    y_inner_train = data["y_inner_train"].to_numpy(dtype=np.int64)
    y_inner_val = data["y_inner_val"].to_numpy(dtype=np.int64)
    histories: dict[str, dict[str, list[float]]] = {}
    candidates = []
    for dropout in (0.0, 0.1, 0.2):
        model, history, seconds = _fit_epochs(
            X_inner_train,
            y_inner_train,
            MAX_EPOCHS,
            0.0001,
            weighting="inverse_sqrt",
            X_monitor=X_inner_val,
            y_monitor=y_inner_val,
            record_history=True,
            dropout=dropout,
        )
        best_epoch, best_f1 = _best_epoch(history)
        label = f"dropout_{dropout:g}"
        histories[label] = history
        candidates.append(
            {
                "dropout": dropout,
                "best_epoch": best_epoch,
                "internal_macro_f1": best_f1,
                "training_seconds": seconds,
                "history_key": label,
            }
        )
    selected_index = int(np.argmax([row["internal_macro_f1"] for row in candidates]))
    selected = candidates[selected_index]

    fixed = _fit_mlp_for_fixed_validation(
        data,
        selected["best_epoch"],
        0.0001,
        weighting="inverse_sqrt",
        dropout=selected["dropout"],
    )
    all_x, all_y, all_test = (
        data["prepared"]["X_train"],
        data["prepared"]["y_train"],
        data["prepared"]["X_test"],
    )
    _, X_all, X_test = fit_preprocessor(all_x, all_test)
    model_all, _, export_seconds = _fit_epochs(
        X_all,
        all_y.to_numpy(dtype=np.int64),
        selected["best_epoch"],
        0.0001,
        weighting="inverse_sqrt",
        dropout=selected["dropout"],
    )
    path = _export_predictions(
        data,
        probabilities_from_logits(model_all, X_test).argmax(axis=1),
        "submission_020_mlp_dropout.csv",
    )
    previous_path = data["root"] / "submissions" / "submission_017_mlp_ponderacao_suave.csv"
    previous = pd.read_csv(previous_path)
    exported = pd.read_csv(path)
    different_from_017 = int((exported.AGE_RANGE != previous.AGE_RANGE).sum())
    result = {
        "configuration": (
            f"MLP 64→32, ReLU + Dropout({selected['dropout']:g}), Adam lr=0.0001, "
            f"pesos 1/sqrt(frequência), época={selected['best_epoch']}"
        ),
        "candidates": candidates,
        "selected": selected,
        "fixed_metrics": fixed["metrics"],
        "reference_macro_f1": 0.4351,
        "reference_kaggle": 0.45,
        "difference_from_017_fixed_macro_f1": fixed["metrics"]["macro_f1"] - 0.4351,
        "repeats_017_configuration": selected["dropout"] == 0.0,
        "different_predictions_from_017": different_from_017,
        "duplicate_017_predictions": different_from_017 == 0,
        "submission": str(path.relative_to(data["root"])),
        "export_training_seconds": export_seconds,
    }
    _record_result("020", result)
    return {"result": result, "histories": histories}


def run_021() -> dict[str, Any]:
    data = load_experiment_data()
    _, X_inner_train, X_inner_val = fit_preprocessor(
        data["X_inner_train"], data["X_inner_val"]
    )
    y_inner_train = data["y_inner_train"].to_numpy(dtype=np.int64)
    y_inner_val = data["y_inner_val"].to_numpy(dtype=np.int64)
    variants = [("inverse", "1/frequência"), ("inverse_sqrt", "1/sqrt(frequência)")]
    histories: dict[str, dict[str, list[float]]] = {}
    candidates = []
    for weighting, label in variants:
        _, history, seconds = _fit_epochs(
            X_inner_train,
            y_inner_train,
            MAX_EPOCHS,
            0.001,
            weighting=weighting,
            optimizer_name="SGD",
            momentum=0.9,
            X_monitor=X_inner_val,
            y_monitor=y_inner_val,
            record_history=True,
        )
        best_epoch, best_f1 = _best_epoch(history)
        histories[weighting] = history
        candidates.append(
            {
                "weighting": weighting,
                "label": label,
                "best_epoch": best_epoch,
                "internal_macro_f1": best_f1,
                "training_seconds": seconds,
            }
        )
    control_candidate, soft_candidate = candidates
    control = _fit_mlp_for_fixed_validation(
        data,
        control_candidate["best_epoch"],
        0.001,
        weighting="inverse",
        optimizer_name="SGD",
        momentum=0.9,
    )
    soft = _fit_mlp_for_fixed_validation(
        data,
        soft_candidate["best_epoch"],
        0.001,
        weighting="inverse_sqrt",
        optimizer_name="SGD",
        momentum=0.9,
    )

    all_x, all_y, all_test = (
        data["prepared"]["X_train"],
        data["prepared"]["y_train"],
        data["prepared"]["X_test"],
    )
    _, X_all, X_test = fit_preprocessor(all_x, all_test)
    model_all, _, export_seconds = _fit_epochs(
        X_all,
        all_y.to_numpy(dtype=np.int64),
        soft_candidate["best_epoch"],
        0.001,
        weighting="inverse_sqrt",
        optimizer_name="SGD",
        momentum=0.9,
    )
    path = _export_predictions(
        data,
        probabilities_from_logits(model_all, X_test).argmax(axis=1),
        "submission_021_mlp_sgd_ponderacao_suave.csv",
    )
    is_exploratory = soft["metrics"]["macro_f1"] < control["metrics"]["macro_f1"]
    result = {
        "configuration": (
            f"MLP 64→32, ReLU, SGD lr=0.001, momentum=0.9, "
            f"pesos 1/sqrt(frequência), época={soft_candidate['best_epoch']}"
        ),
        "candidates": candidates,
        "control_metrics": control["metrics"],
        "soft_metrics": soft["metrics"],
        "reference_macro_f1": 0.4365,
        "reference_kaggle": 0.43,
        "difference_from_018_fixed_macro_f1": soft["metrics"]["macro_f1"] - 0.4365,
        "difference_from_control_fixed_macro_f1": (
            soft["metrics"]["macro_f1"] - control["metrics"]["macro_f1"]
        ),
        "exploratory": is_exploratory,
        "submission": str(path.relative_to(data["root"])),
        "export_training_seconds": export_seconds,
    }
    _record_result("021", result)
    _write_final_report(data["root"])
    return {"result": result, "histories": histories}


def write_new_experiments_report(root: Path | None = None) -> Path:
    root = root or Path(__file__).resolve().parents[1]
    results_path = Path(__file__).resolve().parents[1] / "reports" / ".experiments_016_018_results.json"
    results = json.loads(results_path.read_text())
    metrics_by_id = {
        "019": results["019"]["fixed_metrics"],
        "020": results["020"]["fixed_metrics"],
        "021": results["021"]["soft_metrics"],
    }
    lines = [
        "# Experiências 019–021",
        "",
        "## Protocolo comum",
        "",
        "Divisão estratificada fixa 80/20 (`random_state=42`) e validação interna estratificada de 15% do treino (`random_state=42`), como nas experiências anteriores. A preparação e os pesos de classe foram ajustados apenas com as linhas de treino de cada fase. As escolhas de configuração e época usaram Macro F1 interna; a validação fixa, já consultada em várias experiências, é comparação de desenvolvimento e não avaliação final independente.",
        "",
        "Cada CSV foi gerado com preparação novamente ajustada e modelo(s) novo(s) treinado(s) nas 10.962 linhas rotuladas. Os pesos de classe e épocas foram recalculados/selecionados internamente. `RowId` foi copiado do exemplo oficial na ordem original; os CSVs validam cabeçalho, linhas, unicidade dos identificadores, classes inteiras 0–15 e `index=False`.",
        "",
        "Pontuações Kaggle registadas: 019=0,48; 020=0,45; 021=0,43. A experiência 016 também obteve 0,48; 016 e 019 estão empatadas na pontuação apresentada, sem vencedor entre as duas.",
        "",
        "| Experiência | Configuração | Macro F1 fixa | Accuracy | Comparação local | Kaggle prévio | Estado | CSV |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for number in ("019", "020", "021"):
        item = results[number]
        metrics = metrics_by_id[number]
        reference = f"{item['reference_macro_f1']:.4f}"
        delta_key = {
            "019": "difference_from_016_fixed_macro_f1",
            "020": "difference_from_017_fixed_macro_f1",
            "021": "difference_from_018_fixed_macro_f1",
        }[number]
        delta = item[delta_key]
        previous_id = {"019": "016", "020": "017", "021": "018"}[number]
        comparison = f"{delta:+.4f} vs. {previous_id} ({reference})"
        kaggle = f"{item['reference_kaggle']:.2f}"
        status = "Exploratória" if item.get("exploratory", False) else "Comparação de desenvolvimento"
        lines.append(
            f"| {number} | {item['configuration']} | {metrics['macro_f1']:.4f} | "
            f"{metrics['accuracy']:.4f} | {comparison} | {kaggle} (referência) | "
            f"{status} | `{item['submission']}` |"
        )
    for number in ("019", "020", "021"):
        metrics = metrics_by_id[number]
        lines.extend(
            [
                "",
                f"## {number} — métricas por escalão",
                "",
                "| Classe | Escalão | F1 | Recall | Suporte |",
                "|---:|---|---:|---:|---:|",
            ]
        )
        for row in metrics["per_class"]:
            emphasis = " **" if row["classe"] in (1, 6, 13) else ""
            close = "**" if emphasis else ""
            lines.append(
                f"| {emphasis}{row['classe']}{close} | {emphasis}{row['escalão']}{close} | "
                f"{row['F1']:.4f} | {row['recall']:.4f} | {row['support']} |"
            )
    comparisons = compare_new_submissions(root)
    lines.extend(
        [
            "",
            "## Comparação de previsões por RowId",
            "",
            "| CSV A | CSV B | Previsões diferentes | Duplicado |",
            "|---|---|---:|---|",
        ]
    )
    for row in comparisons.itertuples(index=False):
        lines.append(
            f"| `{row.csv_a}` | `{row.csv_b}` | {row.previsoes_diferentes} | "
            f"{'Sim' if row.duplicado else 'Não'} |"
        )
    lines.extend(
        [
            "",
            "## Configuração/duplicados",
            "",
            f"- 019 repete os pesos 75/25 do 016: {results['019']['repeats_016_configuration']}; "
            f"previsões iguais: {results['019']['duplicate_016_predictions']} "
            f"({results['019']['different_predictions_from_016']} diferentes).",
            f"- 020 repete a configuração sem dropout do 017: {results['020']['repeats_017_configuration']}; "
            f"previsões iguais: {results['020']['duplicate_017_predictions']} "
            f"({results['020']['different_predictions_from_017']} diferentes).",
            f"- 021 exploratória face ao controlo 1/frequência: {results['021']['exploratory']}.",
            "",
        ]
    )
    report_path = root / "reports" / "experiments_019_021.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
