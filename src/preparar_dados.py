from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SPECIAL_TOKENS = (
    "sem resposta",
    "nao aplicavel",
    "nao sabe",
    "ausente",
    "indeterminado",
    "nao identificado",
)


@dataclass
class DataPreparationReport:
    excluded_columns: List[str]
    numeric_columns: List[str]
    categorical_columns: List[str]
    in16_rule_summary: Dict[str, Dict[str, int]]
    bmi_summary: Dict[str, Any]


def _normalise_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.lower()


def _normalise_am1_value(value: Any) -> str:
    text = _normalise_text(value)
    text = text.replace(" nao ", " não ")
    text = text.replace("ha ", "há ")
    text = text.replace("mais de 6 meses", "mais de 6 meses")
    return text.strip()


def _is_maybe_numeric(value: Any) -> bool:
    if pd.isna(value):
        return False
    text = _normalise_text(value)
    if text == "":
        return False
    if any(token in text for token in SPECIAL_TOKENS):
        return False
    try:
        float(text)
        return True
    except ValueError:
        return False


def _classify_special_status(value: Any) -> str:
    if pd.isna(value):
        return "ausente"
    text = _normalise_text(value)
    if text == "":
        return "ausente"
    if "sem resposta" in text:
        return "sem_resposta"
    if "nao aplicavel" in text or "não aplicavel" in text or "nao aplicavel" in text:
        return "nao_aplicavel"
    if "nao sabe" in text or "não sabe" in text:
        return "nao_sabe"
    if "nao identificado" in text or "não identificado" in text:
        return "nao_identificado"
    if text in {"unknown", "unk", "undefined", "na", "n/a", "nao disponivel", "não disponivel"}:
        return "valor_nao_reconhecido"
    try:
        numeric = float(str(value).replace(",", ".").strip())
        if np.isfinite(numeric):
            return "valor_numeric"
    except (TypeError, ValueError):
        pass
    return "valor_nao_reconhecido"


def _coerce_numeric(series: pd.Series) -> pd.Series:
    cleaned = series.map(lambda value: value.strip() if isinstance(value, str) else value)
    cleaned = cleaned.map(lambda value: np.nan if _classify_special_status(value) != "valor_numeric" else value)
    return pd.to_numeric(cleaned, errors="coerce")


def inspect_schema(df: pd.DataFrame, target_col: str = "AGE_RANGE") -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": [],
        "categorical_columns": [],
        "missing": {},
        "constant_columns": [],
        "candidate_identifiers": [],
        "special_missing_candidates": {},
    }

    for column in df.columns:
        s = df[column]
        if column == target_col:
            continue
        if s.nunique(dropna=True) <= 1:
            report["constant_columns"].append(column)
        if pd.api.types.is_numeric_dtype(s):
            report["numeric_columns"].append(column)
        else:
            report["categorical_columns"].append(column)
        report["missing"][column] = int(s.isna().sum())
        report["special_missing_candidates"][column] = int(
            s.astype(str).str.contains(
                "Sem resposta|Nao aplicavel|Nao sabe|N/A|Unknown|NA|undefined|Sem resposta",
                case=False,
                na=False,
            ).sum()
        )

    for column in df.columns:
        if column in {target_col, "PID", "HHID", "PROXY", "COUNTRY"}:
            continue
        s = df[column]
        if s.nunique(dropna=True) == len(df):
            report["candidate_identifiers"].append(column)
    return report


def summarise_age_related_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    suggestions = sorted({column for column in df.columns if any(key in column.lower() for key in ("age", "birth", "born", "year", "month"))})
    return {
        "potential_age_or_birth_fields": suggestions,
        "require_documentation_before_use": suggestions,
    }


def summarise_variable_types(df: pd.DataFrame, target_col: str = "AGE_RANGE") -> Dict[str, Any]:
    """Review variable types and special values in a format suitable for a data-preparation report."""
    summary: Dict[str, Any] = {
        "numeric": [],
        "categorical": [],
        "object_or_string": [],
        "special_values": {},
    }
    for column in df.columns:
        if column == target_col:
            continue
        s = df[column]
        if pd.api.types.is_numeric_dtype(s):
            summary["numeric"].append(column)
        else:
            summary["categorical"].append(column)
            if s.dtype == object or pd.api.types.is_string_dtype(s):
                summary["object_or_string"].append(column)
        special_counts = {
            "sem_resposta": int(s.astype(str).str.contains("sem resposta", case=False, na=False).sum()),
            "nao_aplicavel": int(s.astype(str).str.contains("nao aplicavel|nao aplicável|não aplicavel|não aplicável", case=False, na=False).sum()),
            "nao_sabe": int(s.astype(str).str.contains("nao sabe|nao sabe|não sabe", case=False, na=False).sum()),
            "ausente": int(s.isna().sum()),
        }
        summary["special_values"][column] = special_counts
    return summary


def review_questionnaire_rules(df: pd.DataFrame) -> Dict[str, Any]:
    """Audit observed questionnaire logic related to skip rules and conditional fields."""
    result: Dict[str, Any] = {}
    if {"AM1", "IN16"}.issubset(df.columns):
        am1_norm = df["AM1"].map(_normalise_am1_value)
        result["am1_in16_cross_tab"] = pd.crosstab(am1_norm, df["IN16"].isna()).to_dict()
        result["in16_filled_when_am1_not_applicable"] = int(
            df.loc[am1_norm.isin({"há 12 meses ou mais", "nunca"}), "IN16"].notna().sum()
        )
        result["in16_missing_when_am1_recent"] = int(
            df.loc[am1_norm.isin({"há menos de 6 meses", "6 a menos de 12 meses"}), "IN16"].isna().sum()
        )
    return result


def validate_in16_am1_rules(train_df: pd.DataFrame, test_df: Optional[pd.DataFrame] = None) -> Dict[str, Dict[str, int]]:
    """Audit the AM1/IN16 observed relationship without treating it as a documented rule."""
    frames = {"train": train_df}
    if test_df is not None:
        frames["test"] = test_df

    summary: Dict[str, Dict[str, int]] = {}
    for name, frame in frames.items():
        if {"AM1", "IN16"}.difference(frame.columns):
            summary[name] = {"missing": 0, "invalid": 0}
            continue

        am1_norm = frame["AM1"].map(_normalise_am1_value)
        secondary = am1_norm.isin({"há 12 meses ou mais", "nunca"})
        rule_a = frame["IN16"].isna() & secondary
        rule_b = frame["IN16"].isna() & am1_norm.eq("sem resposta")
        rule_c = frame["IN16"].isna() & ~am1_norm.isin({"há 12 meses ou mais", "nunca", "sem resposta"})
        unexpected = frame["IN16"].notna() & secondary
        summary[name] = {
            "am1_12m_or_never_missing": int(rule_a.sum()),
            "am1_unknown_missing": int(rule_b.sum()),
            "other_missing": int(rule_c.sum()),
            "total_missing_in16": int(frame["IN16"].isna().sum()),
            "in16_filled_when_am1_suggests_not_applicable": int(unexpected.sum()),
        }
    return summary


def apply_in16_rule(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the observed AM1/IN16 hypothesis while preserving explicit IN16 responses.

    This is not treated as a formal questionnaire rule without documentation; it reflects the
    observed cross-tab between the two fields and is used to preserve the meaning of missing tooth-visit
    reasons in a transparent way.
    """
    out = df.copy()
    if "IN16" not in out.columns or "AM1" not in out.columns:
        return out

    am1_norm = out["AM1"].map(_normalise_am1_value)
    missing_mask = out["IN16"].isna()
    out.loc[missing_mask & am1_norm.isin({"há 12 meses ou mais", "nunca"}), "IN16"] = "Não aplicável - sem consulta nos últimos 12 meses"
    out.loc[missing_mask & am1_norm.eq("sem resposta"), "IN16"] = "Indeterminado - AM1 sem resposta"
    out.loc[missing_mask & ~am1_norm.isin({"há 12 meses ou mais", "nunca", "sem resposta"}), "IN16"] = "Ausente - motivo desconhecido"
    return out


def identify_excluded_columns(df: pd.DataFrame, target_col: str = "AGE_RANGE") -> List[str]:
    """Return only the fixed exclusions that are always excluded from feature inputs.

    Dynamic constant-column removal is not allowed before the train/validation split because it can
    leak information from the full dataset into the preprocessing pipeline. Any train-specific filtering
    must happen inside fit() on the training partition only.
    """
    excluded = {target_col, "PID", "HHID", "PROXY", "COUNTRY"}
    return sorted(excluded & set(df.columns))


def _prepare_fixed_frame(frame: pd.DataFrame, target_col: str = "AGE_RANGE", excluded_columns: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    work = frame.copy()
    if target_col in work.columns:
        y = work[target_col].copy()
        work = work.drop(columns=[target_col])
    else:
        y = None

    exclusions = set(excluded_columns or [])
    exclusions.update({target_col, "PID", "HHID", "PROXY", "COUNTRY"})
    work = work.drop(columns=[column for column in exclusions if column in work.columns], errors="ignore")

    if "AM1" in work.columns and "IN16" in work.columns:
        work = apply_in16_rule(work)

    for column in ("BM1", "BM2"):
        if column in work.columns:
            status = work[column].map(_classify_special_status).rename(f"{column}_status")
            work[f"{column}_status"] = status
            work[column] = _coerce_numeric(work[column])

    if {"BM1", "BM2"}.issubset(work.columns):
        height_m = work["BM1"]
        weight_kg = work["BM2"]
        valid_mask = (
            height_m.notna()
            & weight_kg.notna()
            & np.isfinite(height_m.to_numpy(dtype=float, na_value=np.nan))
            & np.isfinite(weight_kg.to_numpy(dtype=float, na_value=np.nan))
            & (height_m > 0)
            & (weight_kg > 0)
        )
        work["BMI"] = np.nan
        height_m_valid = height_m[valid_mask] / 100.0
        weight_valid = weight_kg[valid_mask]
        work.loc[valid_mask, "BMI"] = weight_valid / (height_m_valid ** 2)
        work["BMI_valid"] = valid_mask

    if "PID" in frame.columns:
        ids = frame[["PID", "HHID"]].copy() if "HHID" in frame.columns else frame[["PID"]].copy()
    else:
        ids = pd.DataFrame(index=frame.index)

    result = {
        "X": work,
        "y": y,
        "identifiers": ids,
        "original_index": frame.index.copy(),
        "excluded_columns": sorted(exclusions & set(frame.columns)),
    }
    return result


class DataPreprocessor(BaseEstimator, TransformerMixin):
    """Statistical preprocessing that learns from the training set only.

    Features are kept aligned to a stable set of columns learned at fit time. Numeric columns are
    imputed with the training median and optionally scaled. Categorical columns are imputed with a
    constant placeholder and one-hot encoded with handle_unknown='ignore'. Columns that are entirely
    missing in the training fold are preserved with an explicit zero-fill policy to avoid dropping them
    silently during validation/test transforms.
    """

    def __init__(self, target_col: str = "AGE_RANGE", numeric_strategy: str = "median", normalize_numeric: bool = False):
        self.target_col = target_col
        self.numeric_strategy = numeric_strategy
        self.normalize_numeric = normalize_numeric
        self.feature_columns_: List[str] = []
        self.numeric_columns_: List[str] = []
        self.categorical_columns_: List[str] = []
        self.output_columns_: List[str] = []
        self.preprocessor_: Optional[ColumnTransformer] = None
        self.all_missing_numeric_columns_: List[str] = []

    def _validate_input(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("DataPreprocessor expects a pandas DataFrame.")
        if self.target_col in X.columns:
            raise ValueError(f"{self.target_col} must be removed before fit/transform; it is target data, not a feature.")
        return X.copy()

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DataPreprocessor":
        frame = self._validate_input(X)
        self.feature_columns_ = list(frame.columns)
        self.numeric_columns_ = [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
        self.categorical_columns_ = [column for column in frame.columns if column not in self.numeric_columns_]
        self.all_missing_numeric_columns_ = [column for column in self.numeric_columns_ if frame[column].isna().all()]

        if self.all_missing_numeric_columns_:
            for column in self.all_missing_numeric_columns_:
                frame[column] = frame[column].fillna(0.0)

        if not self.feature_columns_:
            self.preprocessor_ = ColumnTransformer(transformers=[], remainder="drop")
            self.output_columns_ = []
            return self

        numeric_steps: List[tuple] = [("imputer", SimpleImputer(strategy=self.numeric_strategy))]
        if self.normalize_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        numeric_transformer = Pipeline(steps=numeric_steps)

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
            ]
        )

        transformers: List[tuple] = []
        if self.numeric_columns_:
            transformers.append(("num", numeric_transformer, self.numeric_columns_))
        if self.categorical_columns_:
            transformers.append(("cat", categorical_transformer, self.categorical_columns_))

        self.preprocessor_ = ColumnTransformer(transformers=transformers, remainder="drop")
        self.preprocessor_.fit(frame)
        self.output_columns_ = list(self.preprocessor_.get_feature_names_out())
        return self

    def transform(self, X: pd.DataFrame) -> Any:
        if self.preprocessor_ is None:
            raise ValueError("DataPreprocessor.fit() must be called before transform().")
        frame = self._validate_input(X)
        missing = [column for column in self.feature_columns_ if column not in frame.columns]
        if missing:
            raise ValueError(f"Required feature columns are missing before transform: {missing}")
        for column in self.all_missing_numeric_columns_:
            if column in frame.columns:
                frame[column] = frame[column].fillna(0.0)
        frame = frame[self.feature_columns_]
        return self.preprocessor_.transform(frame)

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Any:
        return self.fit(X, y).transform(X)


def build_preprocessor(X_train: pd.DataFrame, target_col: str = "AGE_RANGE", normalize_numeric: bool = False) -> DataPreprocessor:
    """Build a training-only preprocessor.

    This helper is safe only when X_train is the training split and not the full dataset or a mixed
    train+validation view. The caller must split before fitting; fit() learns imputations, categories and
    other learned statistics only from the training subset.
    """
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")
    if target_col in X_train.columns:
        raise ValueError(f"{target_col} must be removed before building a train-only preprocessor.")
    preprocessor = DataPreprocessor(target_col=target_col, normalize_numeric=normalize_numeric)
    preprocessor.fit(X_train)
    return preprocessor


def build_preparation_report(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str = "AGE_RANGE") -> Dict[str, Any]:
    train = train_df.copy()
    test = test_df.copy()
    train_schema = inspect_schema(train, target_col=target_col)
    test_schema = inspect_schema(test, target_col=target_col)

    train_fixed = _prepare_fixed_frame(train, target_col=target_col)
    test_fixed = _prepare_fixed_frame(test, target_col=target_col)

    rules = validate_in16_am1_rules(train, test)
    variable_review = {
        "train_types": summarise_variable_types(train, target_col=target_col),
        "test_types": summarise_variable_types(test, target_col=target_col),
        "questionnaire_rules": {
            "train": review_questionnaire_rules(train),
            "test": review_questionnaire_rules(test),
        },
    }
    report = {
        "train_schema": train_schema,
        "test_schema": test_schema,
        "target_distribution": train[target_col].value_counts().sort_index().to_dict(),
        "potential_age_related_columns": summarise_age_related_columns(train),
        "variable_review": variable_review,
        "in16_am1_rule_checks": rules,
        "bm1_bm2_policy": {
            "assumption": "BM1 e BM2 interpretados, respetivamente, como altura em cm e peso em kg; esta unidade ainda precisa de confirmação documental.",
            "special_statuses_preserved": ["sem_resposta", "nao_aplicavel", "nao_sabe", "ausente", "valor_nao_reconhecido"],
            "numeric_conversion": "A conversao apenas acontece para valores numericos reconhecidos; status especiais ficam em BM1_status e BM2_status.",
            "bmi_formula": "BMI = BM2 / (BM1 / 100)^2 quando BM1 > 0 e BM1/BM2 validos.",
        },
        "bmi_summary": {
            "train_valid_bmi": int(train_fixed["X"]["BMI"].notna().sum()) if "BMI" in train_fixed["X"].columns else 0,
            "test_valid_bmi": int(test_fixed["X"]["BMI"].notna().sum()) if "BMI" in test_fixed["X"].columns else 0,
        },
        "recommended_exclusions": identify_excluded_columns(train, target_col=target_col),
        "documented_assumptions": {
            "pid_hhid_are_identifiers": True,
            "proxy_country_are_constant": True,
            "bm1_assumed_height_cm": True,
            "bm2_assumed_weight_kg": True,
            "in16_rule_is_hypothesis_from_observed_am1_cross_tab": True,
            "do_not_use_age_related_fields_without_documentation": True,
        },
        "data_leakage_policy": {
            "description": "A aprendizagem de imputação, codificação e normalização ocorre somente na parcela de treino; as estatísticas aprendidas são aplicadas depois a validação/teste sem refit.",
            "train_fit_only": "As imputações e categorias são aprendidas em fit() do treino; transform() não recalcula estatísticas.",
            "target_handling": "AGE_RANGE é removida antes do fit do pré-processador, sendo tratada como variável alvo e não como feature.",
            "no_global_fit": "Não se concatena treino e teste para aprender categorias ou estatísticas.",
        },
    }
    return report


def preparar_dados_fixos(train_df: pd.DataFrame, target_col: str = "AGE_RANGE", excluded_columns: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    train = train_df.copy()
    if target_col not in train.columns:
        raise ValueError(f"The target column '{target_col}' is required in the training set.")

    prepared = _prepare_fixed_frame(train, target_col=target_col, excluded_columns=excluded_columns)
    X = prepared["X"].copy()
    y = prepared["y"].copy()
    return {
        "X": X,
        "y": y,
        "identifiers": prepared["identifiers"],
        "original_index": prepared["original_index"],
        "excluded_columns": prepared["excluded_columns"],
        "report": build_preparation_report(train_df, train_df, target_col=target_col),
    }


def preparar_dados(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "AGE_RANGE",
    excluded_columns: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Return the fixed preparation for train/test without fitting learned parameters.

    The caller can then create a new DataPreprocessor, fit it on the training split, and apply it on
    validation or Kaggle test data without leakage.
    """
    train = train_df.copy()
    test = test_df.copy()

    if target_col not in train.columns:
        raise ValueError(f"The target column '{target_col}' is required in the training set.")

    train_fixed = _prepare_fixed_frame(train, target_col=target_col, excluded_columns=excluded_columns)
    test_fixed = _prepare_fixed_frame(test, target_col=target_col, excluded_columns=excluded_columns)

    X_train = train_fixed["X"]
    X_test = test_fixed["X"]
    y_train = train_fixed["y"]

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "train_identifiers": train_fixed["identifiers"],
        "test_identifiers": test_fixed["identifiers"],
        "train_original_index": train_fixed["original_index"],
        "test_original_index": test_fixed["original_index"],
        "excluded_columns": sorted(set(train_fixed["excluded_columns"]) | set(test_fixed["excluded_columns"])),
        "preparation_summary": build_preparation_report(train_df, test_df, target_col=target_col),
        "preprocessor_factory": build_preprocessor,
    }


def prepare_data(*args, **kwargs):
    return preparar_dados(*args, **kwargs)


def make_stratified_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return a reproducible stratified train/validation split with the original indices preserved."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")
    if not isinstance(y, pd.Series):
        raise TypeError("y must be a pandas Series.")
    if len(X) != len(y):
        raise ValueError("X and y must have the same length.")
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_val, y_train, y_val


def make_model_pipeline(model: Any, target_col: str = "AGE_RANGE", normalize_numeric: bool = False) -> Pipeline:
    """Create a scikit-learn pipeline that applies the train-only preprocessor and then a model.

    The pipeline enforces the correct order: fit on the training split, then transform validation/test.
    Each experiment can create a new pipeline instance and reuse the same stratified split for fair model
    comparisons. If a cross-validation loop is used, the preparation is refit inside each fold.
    """
    if model is None:
        raise ValueError("A model instance is required.")
    preprocessor = DataPreprocessor(target_col=target_col, normalize_numeric=normalize_numeric)
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def _resolve_project_data_path(*parts: str) -> Path:
    root = Path(__file__).resolve().parents[1]
    return root.joinpath(*parts)


if __name__ == "__main__":
    train_path = _resolve_project_data_path("notebooks", "dataset_train.csv")
    test_path = _resolve_project_data_path("notebooks", "dataset_test_X.csv")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    result = preparar_dados(train, test)
    print(f"Prepared train rows: {len(result['X_train'])}")
    print(f"Prepared test rows: {len(result['X_test'])}")
    print(f"Train target distribution: {result['y_train'].value_counts().sort_index().to_dict()}")
    print(f"Excluded columns: {result['excluded_columns']}")
