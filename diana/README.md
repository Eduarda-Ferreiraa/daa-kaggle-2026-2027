# Modelos da Diana: HGB, Random Forest, regressor ordinal e ensemble

Código das submissões `hgb_submission.csv` e `ensemble_submission.csv`.

## Estrutura

```
diana/
├── dados/          # não vai para o GitHub: CSV da competição + parquets gerados
├── resultados/     # não vai para o GitHub: previsões out-of-fold e do teste
├── submissions/    # não vai para o GitHub
├── notebooks/
│   ├── tratamento.ipynb                        # gera dados/train_tratado.parquet e test_tratado.parquet
│   ├── hist_gradient_boosting_classifier.ipynb # HGB: procuras de hiperparâmetros, sobreajuste, secção 9 (OOF)
│   ├── hgb_regressor_ordinal.ipynb             # regressor + cortes otimizados, secção 9 (OOF)
│   ├── 02.1_random_forest.py                   # Random Forest (procura de max_features, OOF, submissão)
│   └── ensemble.ipynb                          # média ponderada vs stacking, com os 3 modelos
└── scripts/
    ├── submissao_hgb.py                        # gera submissions/hgb_submission.csv
    └── submissao_ensemble.py                   # gera submissions/ensemble_submission.csv
```

## Como reproduzir

1. Copiar `dataset_train.csv`, `dataset_test_X.csv` e `example_submission_10.csv` (versão corrigida, sem `WGT`)
   para `diana/dados/`.
2. Correr `notebooks/tratamento.ipynb`.
3. Submissões rápidas, a partir da pasta `diana/`:
   - `python scripts/submissao_hgb.py` (cerca de 1 minuto)
   - `python scripts/submissao_ensemble.py` (cerca de 7 minutos)

## Tratamento (`tratamento.ipynb`)

1. Remover colunas constantes (`PROXY`, `COUNTRY`) e identificadores (`PID`, `HHID`).
2. Unificar "NR" com "Sem resposta".
3. Converter 30 colunas numéricas guardadas como texto (limites como ">=7"; `PE7`/`PE9` de HHMM para minutos);
   "Não aplicável" passa a `NaN` mais uma coluna indicadora. `PA1` passa a `PA1_meses`.
4. Mapear 77 escalas ordinais para números, com indicadoras para respostas especiais.
5. Restante texto passa a `category`, com as categorias do teste alinhadas com as do treino.

## Resultados (Macro F1, 5 folds estratificados, `random_state=42`, dados sem `WGT`)

| Modelo | Macro F1 |
|---|---|
| HistGradientBoosting | 0,470 ± 0,007 |
| Random Forest (`max_features=0.85`, `min_samples_leaf=3`) | 0,463 |
| Stacking HGB + RF + regressor ordinal | **0,479** ± 0,008 (melhor que o HGB em 4/5 folds) |

Os outputs gravados no `tratamento`, no `hist_gradient_boosting_classifier` e no `hgb_regressor_ordinal` já são
dos dados sem `WGT`. Os do `ensemble.ipynb` ainda são da versão antiga (com `WGT`, F1 ≈ 0,59); os valores
atuais do ensemble são os da tabela, obtidos com `scripts/submissao_ensemble.py`.
