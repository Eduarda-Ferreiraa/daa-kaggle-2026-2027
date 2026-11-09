from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preparar_dados import build_preparation_report


TRAIN_PATH = ROOT / "notebooks" / "dataset_train.csv"
TEST_PATH = ROOT / "notebooks" / "dataset_test_X.csv"
REPORT_PATH = ROOT / "reports" / "data_preparation_report.json"


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    report = build_preparation_report(train, test, target_col="AGE_RANGE")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved report to: {REPORT_PATH}")
    print(f"Train rows: {len(train)} | Test rows: {len(test)}")
    print(f"Target distribution: {report['target_distribution']}")


if __name__ == "__main__":
    main()
