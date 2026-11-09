# Preparação de dados para a competição Kaggle ML@DAA 2026/2027

Este repositório centraliza a preparação dos dados para a tarefa de previsão de `AGE_RANGE`.

## Decisões principais

- `PID` e `HHID` são tratados como identificadores e não entram no modelo.
- `PROXY` e `COUNTRY` são eliminadas por serem constantes no treino.
- `IN16` é normalizado com regras condicionais com base em `AM1` para preservar informação de consultas dentárias e de respostas vazias com significado diferente.
- `BM1` e `BM2` são interpretados como altura (cm) e peso (kg), respeitando a hipótese documental da unidade.
- A criação de `BMI` usa a fórmula `peso / (altura_m)^2` apenas quando as duas medidas são válidas e a altura é positiva.
- Variáveis numéricas em texto são convertidas somente quando o conteúdo é claramente numérico; valores especiais como `Sem resposta` são preservados em flags separadas.
- A lógica de preparação está num módulo reutilizável, separado do treino e da submissão, de modo a evitar duplicação em notebooks.

## Módulo central

O pipeline reutilizável vive em `src/preparar_dados.py` e expõe:

- `inspect_schema(df, target_col="AGE_RANGE")`
- `validate_in16_am1_rules(train_df, test_df=None)`
- `apply_in16_rule(df)`
- `DataPreprocessor`
- `preparar_dados(train_df, test_df, target_col="AGE_RANGE")`

A função `preparar_dados` é a entrada principal para a preparação e respeita a interface `fit/transform` do scikit-learn.
