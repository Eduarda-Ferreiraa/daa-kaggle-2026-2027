"""Notebooks 059–075: criação sem sobrescrita e atualização só de células geradas."""
import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _cell(kind, text, tags=None):
    cell = dict(cell_type=kind, id=uuid.uuid4().hex[:8], metadata={"tags": tags or []},
                source=text.splitlines(keepends=True))
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    return cell


def _write(path, notebook):
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+"\n", encoding="utf-8")


def _code(k):
    return f'''from pathlib import Path
import sys
import pandas as pd
from IPython.display import display, Image
root = next(p for p in [Path.cwd().resolve(), *Path.cwd().resolve().parents]
            if (p/'src/experiments_059_075.py').is_file())
sys.path.insert(0, str(root))
from src.experiments_059_075 import run_experiment
result = run_experiment('{k}')
print('Estado: procedimento concluído; resultados guardados com contexto verificado.')
print('Versões:', result['context']['base']['versions'])
print('Duração original (s):', result['seconds'])
print('Notas:', result.get('note', ''))
if result.get('candidates'):
    display(pd.DataFrame([{{'configuração':r['label'],'Macro F1 CV':r['mean'],'DP':r['std'],'folds':r.get('folds',r.get('scores'))}} for r in result['candidates']]))
if result.get('selected'):
    print('Selecionado:', result['selected']['label'])
    print('Configuração:', result['selected']['spec'])
if result.get('paired_cv'):
    print('Diferenças emparelhadas:', result['paired_cv'])
if result.get('fixed'):
    display(pd.DataFrame([{{'modelo':name,'parcela':part,'Macro F1':groups[part]['macro_f1'],'accuracy':groups[part]['accuracy']}} for name,groups in result['fixed'].items() for part in ['train','validation']]))
    for name, groups in result['fixed'].items():
        print('Escalões:', name)
        display(pd.DataFrame(groups['validation']['per_class']))
        print('Diagnósticos de ajuste:', groups.get('diagnostics'))
        print('MAE em códigos de escalão (não anos):', groups.get('mae_in_class_codes'))
        if 'log_loss' in groups: print('Log loss:',groups['log_loss'])
if result.get('metrics_oof'):
    print('Métricas OOF de desenvolvimento:', result['metrics_oof']['macro_f1'], result['metrics_oof']['accuracy'])
    display(pd.DataFrame(result['metrics_oof']['per_class']))
if 'selected_cases' in result:
    print('Perfis sinalizados:', result['flagged'], '/', result['total'])
    display(pd.DataFrame(result['selected_cases']))
    display(pd.DataFrame(result['by_class']))
if 'categorical_profiles' in result:
    table = pd.DataFrame(result['categorical_profiles'])
    display(table[table.variable.isin(['HATLEVEL','MAINSTAT','HHINCOME','HS1','HS2'])])
    display(pd.DataFrame(result['numeric_profiles']))
    display(pd.DataFrame(result['chronic_diseases']))
for file in sorted((root/'reports').glob('experiencia_{k}_*.png')):
    display(Image(filename=str(file)))
print('CSV Kaggle:', result.get('submission'))
print('Pontuação Kaggle:', result['kaggle_score'])
'''


def create_notebooks():
    from src.experiments_059_075 import NAMES, DESCRIPTIONS
    folder = ROOT/"notebooks"
    folder.mkdir(exist_ok=True)
    for k, name in NAMES.items():
        path = folder/f"{k}_{name}.ipynb"
        collisions = [p for p in folder.glob(f"{k}_*.ipynb") if p != path]
        if collisions:
            raise FileExistsError(f"Número {k} já ocupado por {collisions}; não criar experiências com IDs duplicados.")
        if path.exists():
            continue
        header = (f"# {k} — {name.replace('_',' ')}\n\n{DESCRIPTIONS[k]}\n\n"
                  "**Estado: por executar.**\n\nSeleção dentro do treino de desenvolvimento (8.769 linhas), com os mesmos três folds. Validação fixa: 2.193 linhas, já observada anteriormente. Pré-processamento ajustado dentro de cada treino. Sem submissão automática.")
        notebook = dict(nbformat=4, nbformat_minor=5,
                        metadata={"kernelspec": {"display_name": "Python (daa)", "language": "python", "name": "python3"}},
                        cells=[_cell("markdown", header, ["generated_header"]), _cell("code", _code(k)),
                               _cell("markdown", "## Conclusões locais\n\nPor preencher após execução real.", ["generated_conclusion"]),
                               _cell("markdown", "## Registo manual\n\n- Data:\n- CSV submetido (se aplicável):\n- Nota Kaggle:\n- Conclusão:\n- Melhorias a experimentar:\n\nEsta célula e quaisquer notas pessoais não são substituídas pelo script.")])
        _write(path, notebook)


def update_notebook(k, result):
    from src.experiments_059_075 import NAMES
    create_notebooks()
    path = ROOT/"notebooks"/f"{k}_{NAMES[k]}.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    text = "## Conclusões locais\n\n"
    if result.get("fixed"):
        selected = result["fixed"]["selecionado"]["validation"]
        control = result["fixed"]["controlo"]["validation"]
        text += (f"- Selecionado internamente: `{result['selected']['label']}`.\n"
                 f"- Macro F1 desenvolvimento: **{selected['macro_f1']:.4f}**; accuracy: **{selected['accuracy']:.4f}**.\n"
                 f"- Diferença face ao controlo contemporâneo: **{selected['macro_f1']-control['macro_f1']:+.4f}**.\n"
                 f"- Diferença média nos mesmos folds: **{result['paired_cv']['mean']:+.4f}**, DP **{result['paired_cv']['std']:.4f}**.\n")
    else:
        text += "- Análise concluída; consultar tabelas e gráficos. Não gera submissão Kaggle.\n"
    submission = result.get("submission")
    if submission:
        text += f"- CSV: `{submission.get('path')}`. Motivo se não criado: {submission.get('reason', '—')}.\n"
    text += "\n"+result.get("note", "")+"\n\n**Decisão:** preencher depois da revisão; não inferir melhoria pública de uma pequena diferença local."
    for cell in notebook["cells"]:
        tags = cell["metadata"].get("tags", [])
        if "generated_header" in tags:
            source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            cell["source"] = source.replace("**Estado: por executar.**", "**Estado: executado pelo runner; outputs guardados.**").splitlines(keepends=True)
        if "generated_conclusion" in tags:
            cell["source"] = text.splitlines(keepends=True)
    notebook["metadata"]["runner_context"] = result["context"]
    _write(path, notebook)
    return path


def execute_view(path):
    import nbformat
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    InteractiveShell.instance()
    notebook = nbformat.read(path, as_version=4)
    namespace = {"__name__": "__main__"}
    count = 0
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        count += 1
        with capture_output(display=True) as captured:
            exec(compile(cell.source, str(path), "exec"), namespace)
        cell.execution_count, cell.outputs = count, []
        if captured.stdout:
            cell.outputs.append(nbformat.v4.new_output("stream", name="stdout", text=captured.stdout))
        if captured.stderr:
            cell.outputs.append(nbformat.v4.new_output("stream", name="stderr", text=captured.stderr))
        for output in captured.outputs:
            cell.outputs.append(nbformat.v4.new_output("display_data", data=output.data, metadata=output.metadata))
    notebook.metadata["execution_method"] = "Procedimento no runner; células executadas em Python/IPython consultando resultados completos com contexto verificado."
    nbformat.validate(notebook)
    nbformat.write(notebook, path)


if __name__ == "__main__":
    create_notebooks()
