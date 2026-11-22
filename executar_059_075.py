"""Executar 059–075 em sequência. Usar --grupo para dividir o trabalho."""
import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[name] = "1"
GROUPS = {"svm": [f"{i:03}" for i in range(59, 67)], "svr": ["067", "068", "069"],
          "ensembles": ["070", "071", "072"], "analise": ["073", "074", "075"],
          "todos": [f"{i:03}" for i in range(59, 76)]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ids", nargs="*")
    parser.add_argument("--grupo", choices=GROUPS, default="todos")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--one", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    ids = [str(int(k)).zfill(3) for k in args.ids] if args.ids else GROUPS[args.grupo]
    if any(k not in GROUPS["todos"] for k in ids):
        raise SystemExit("Usar apenas 059–075.")
    required = ["numpy", "pandas", "scipy", "sklearn", "joblib", "threadpoolctl", "matplotlib", "nbformat", "IPython"]
    if any(k in ids for k in ["071", "072", "073", "074"]):
        required += ["xgboost", "torch"]
    if "072" in ids:
        required.append("lightgbm")
    absent = [p for p in required if importlib.util.find_spec(p) is None]
    if absent:
        raise SystemExit("Dependências ausentes: " + ", ".join(absent) + ". Não foi iniciado nenhum treino.")
    from src.experiments_059_075 import ROOT as MODULE_ROOT, NAMES, write_report
    if MODULE_ROOT != ROOT:
        raise SystemExit("Foi importado src de outro projeto; executar na raiz deste repositório.")
    if args.check:
        from src.experiments_056_058 import _data, context
        from src.experiments_059_075 import _legacy_recipe
        data = _data()
        print("Dados fixos:", data["X"].shape, data["V"].shape, data["full"]["X_test"].shape)
        print("Versões:", context()["versions"])
        if any(k in ids for k in ["071", "072", "073", "074"]):
            _legacy_recipe("058")
        if "072" in ids:
            _legacy_recipe("056")
        print("Verificação inicial concluída; nenhum modelo treinado. Grupo:", ids)
        return
    from notebooks_059_075 import create_notebooks, update_notebook, execute_view
    if args.one:
        from src.experiments_059_075 import run_experiment
        k = ids[0]
        result = run_experiment(k)
        path = update_notebook(k, result)
        execute_view(path)
        metric = result.get("fixed", {}).get("selecionado", {}).get("validation", {}).get("macro_f1")
        print("CONCLUIDO", k, metric, result.get("submission"), flush=True)
        return
    import fcntl
    (ROOT/"reports").mkdir(exist_ok=True)
    lock = (ROOT/"reports/.execucao_sequencial.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Outra sequência de experiências está ativa; aguardar.")
    create_notebooks()
    write_report()
    for k in ids:
        print("INICIAR", k, NAMES[k], flush=True)
        done = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--one", k], cwd=ROOT)
        if done.returncode:
            raise SystemExit(f"INTERROMPIDO {k}, exit {done.returncode}. Não avançar sobre erros; os resultados concluídos estão preservados.")


if __name__ == "__main__":
    main()
