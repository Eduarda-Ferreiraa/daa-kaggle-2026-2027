"""Reusable data-preparation utilities for the Kaggle age-range task."""

from .preparar_dados import (
    DataPreprocessor,
    apply_in16_rule,
    build_preparation_report,
    build_preprocessor,
    inspect_schema,
    preparar_dados,
    preparar_dados_fixos,
    prepare_data,
    summarise_age_related_columns,
    validate_in16_am1_rules,
)
