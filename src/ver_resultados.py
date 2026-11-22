"""Consultar resultados existentes sem iniciar novo treino."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def carregar_resultado(numero):
    numero = str(numero).zfill(3)
    if int(numero) <= 41:
        path = ROOT / 'reports/.experiments_035_055_retoma_results.json'
        results = json.loads(path.read_text()) if path.exists() else {}
        if numero not in results:
            raise FileNotFoundError(f'{numero}: execução ainda não concluída; consultar o relatório de estado.')
        cp = json.loads((ROOT / 'reports/.experiments_035_055_retoma_checkpoint.json').read_text())
        context = cp['contexts'][numero]
        expected = {
            'train_sha256': ROOT / 'notebooks/dataset_train.csv',
            'test_sha256': ROOT / 'notebooks/dataset_test_X.csv',
            'preparation_sha256': ROOT / 'src/preparar_dados.py',
        }
        for name, file in expected.items():
            if _sha(file) != context[name]:
                raise ValueError(f'{numero}: {file.name} mudou desde a execução.')
        candidates = [ROOT / 'src/experiments_035_055.py',
                      ROOT / 'referencias/experiments_035_055_codigo_executado_037_039.py']
        if not any(p.exists() and _sha(p) == context['runner_sha256'] for p in candidates):
            raise ValueError(f'{numero}: fonte exata da execução não foi encontrada.')
        return results[numero]
    from src.experiments_042_055 import RESULT, context
    results = json.loads(RESULT.read_text()) if RESULT.exists() else {}
    if numero not in results:
        raise FileNotFoundError(f'{numero}: execução ainda não concluída; não há métricas para apresentar.')
    if results[numero]['context'] != context(numero):
        raise ValueError(f'{numero}: dados, código ou versões mudaram desde a execução.')
    return results[numero]
