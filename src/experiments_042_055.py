"""Continuação 042–055: preparação original, seleção interna, outputs reais.
Nenhuma execução é feita ao importar este módulo.
"""
from __future__ import annotations
import copy,gc,hashlib,json,time,importlib.metadata,random
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.preprocessing import FunctionTransformer,StandardScaler,OneHotEncoder
from sklearn.decomposition import TruncatedSVD,PCA
from sklearn.cluster import KMeans,DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import SelectPercentile,f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import StackingClassifier,VotingClassifier,RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance,PartialDependenceDisplay
from sklearn.metrics import silhouette_score
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.preparar_dados import DataPreprocessor,build_preprocessor,make_model_pipeline
from src.experiments_016_018 import load_experiment_data,make_xgb,CLASS_IDS,_classification_metrics,to_dense_float32
from src.experiments_022_027 import _fit_mlp,_predict_probabilities
from src.experiments_035_055 import _compare_and_export,_plot_confusion
ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports'
RESULT=REPORT/'.experiments_042_055_results.json'

def score(y,p):return _classification_metrics(np.asarray(y),np.asarray(p,dtype=int))
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def context(k):
 import inspect
 groups={'042':['factory','supervised'],'043':['factory','supervised'],'044':['factory','supervised','dense_checked'],'045':['factory','supervised','dense_checked'],
 '046':['ae_fit','autoencoder_latent','augmented_classifier'],'047':['ae_fit','crossfit_errors','autoencoder_error','augmented_classifier'],
 '048':['factory','supervised','ClusterID'],'049':['factory','supervised','MedoidDistances'],'050':['descriptive_dbscan'],
 '051':['mlp_comparisons','mlp_stop'],'052':['mlp_comparisons'],'053':['permutation'],'054':['shap_native'],'055':['pdp','recompute_bmi']}
 functions=groups[k]+['prep','csr','score','xgb_estimator','representation_frame']
 code='\n'.join(inspect.getsource(globals()[name]) for name in functions)
 return dict(train=digest(ROOT/'notebooks/dataset_train.csv'),test=digest(ROOT/'notebooks/dataset_test_X.csv'),preparation=digest(ROOT/'src/preparar_dados.py'),
  relevant_code_sha256=hashlib.sha256(code.encode()).hexdigest(),reference_helpers=digest(ROOT/'src/experiments_016_018.py'),mlp_helpers=digest(ROOT/'src/experiments_022_027.py'),
  versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','torch','xgboost']})

def write_result(k,r):
 results=json.loads(RESULT.read_text()) if RESULT.exists() else {}
 results[k]=r;tmp=RESULT.with_suffix('.tmp');tmp.write_text(json.dumps(results,ensure_ascii=False,indent=2));tmp.replace(RESULT)
def csr(A):return sparse.csr_matrix(A,dtype=np.float32)
def dense_checked(A):
 needed=int(A.shape[0])*int(A.shape[1])*4
 if needed*5>1024**3:raise MemoryError(f'PCA não densificada: estimativa conservadora {needed*5/2**20:.0f} MiB > 1024 MiB.')
 return to_dense_float32(A)
def prep(normalize=False):return DataPreprocessor(normalize_numeric=normalize)
def xgb_estimator():return make_xgb().named_steps['model']

def representation_frame(X):
 """Datas tratadas como categorias na representação não supervisionada."""
 out=X.copy()
 for c in ['REFYEAR','REFMONTH']:
  if c in out:out[c]=out[c].astype('string').astype(object).where(out[c].notna(),np.nan)
 return out


class ClusterID(BaseEstimator,TransformerMixin):
 def __init__(self,k=4):self.k=k
 def fit(self,X,y=None):
  self.representation_=Pipeline([('svd',TruncatedSVD(16,random_state=42)),('scale',StandardScaler())])
  Z=self.representation_.fit_transform(X)
  self.cluster_=KMeans(self.k,n_init=10,random_state=42).fit(Z)
  self.encoder_=OneHotEncoder(categories=[np.arange(self.k)],handle_unknown='ignore',sparse_output=True).fit(self.cluster_.labels_.reshape(-1,1))
  return self
 def transform(self,X):
  check_is_fitted(self,'cluster_')
  return self.encoder_.transform(self.cluster_.predict(self.representation_.transform(X)).reshape(-1,1))

class MedoidDistances(BaseEstimator,TransformerMixin):
 """K-medoids por atualizações alternadas; Gower sobre variáveis originais.
 Amostra aleatória sem alvo, sem matriz quadrática de todos os indivíduos.
 """
 def __init__(self,k=4,sample_size=800):self.k=k;self.sample_size=sample_size
 def _distance(self,A,B):
  out=np.zeros((len(A),len(B)),dtype=np.float32)
  for c in self.columns_:
   if c in self.numeric_:
    a=pd.to_numeric(A[c],errors='coerce').astype(float).fillna(self.medians_[c]).to_numpy(dtype=float)
    b=pd.to_numeric(B[c],errors='coerce').astype(float).fillna(self.medians_[c]).to_numpy(dtype=float)
    out+=np.minimum(np.abs(a[:,None]-b[None,:])/self.ranges_[c],1).astype(np.float32)
   else:
    a=A[c].astype('string').fillna('Ausente - motivo desconhecido').to_numpy();b=B[c].astype('string').fillna('Ausente - motivo desconhecido').to_numpy()
    out+=(a[:,None]!=b[None,:])
  return out/len(self.columns_)
 def fit(self,X,y=None):
  self.columns_=list(X);self.numeric_=[c for c in X if pd.api.types.is_numeric_dtype(X[c]) and c not in ['REFYEAR','REFMONTH']]
  self.medians_={c:float(X[c].median()) if X[c].notna().any() else 0. for c in self.numeric_}
  self.ranges_={c:max(float(X[c].max())-float(X[c].min()),1e-8) if X[c].notna().any() else 1. for c in self.numeric_}
  sample=X.sample(min(len(X),self.sample_size),random_state=42);D=self._distance(sample,sample)
  med=np.random.default_rng(42).choice(len(sample),self.k,replace=False)
  for _ in range(30):
   labels=D[:,med].argmin(1);updated=med.copy()
   for j in range(self.k):
    ids=np.flatnonzero(labels==j)
    if len(ids):updated[j]=ids[D[np.ix_(ids,ids)].sum(1).argmin()]
   if np.array_equal(med,updated):break
   med=updated
  self.medoids_=sample.iloc[med].copy();return self
 def transform(self,X):check_is_fitted(self,'medoids_');return self._distance(X,self.medoids_)

def factory(k,param):
 if k=='042':
  est=[('xgb',make_xgb()),('logistica',make_model_pipeline(LogisticRegression(C=.1,solver='lbfgs',max_iter=2000,random_state=42),normalize_numeric=True)),('rf',make_model_pipeline(RandomForestClassifier(n_estimators=200,min_samples_leaf=2,random_state=42,n_jobs=1)))]
  if param=='media':return VotingClassifier(est,voting='soft',n_jobs=1)
  return StackingClassifier(est,final_estimator=LogisticRegression(C=.1,max_iter=2000,random_state=42),cv=StratifiedKFold(3,shuffle=True,random_state=42),stack_method='predict_proba',n_jobs=1)
 if k=='043':return Pipeline([('prep',prep()),('select',SelectPercentile(f_classif,percentile=param)),('model',xgb_estimator())])
 if k in ['044','045']:
  projection=TruncatedSVD(param,random_state=42) if k=='044' else Pipeline([('dense',FunctionTransformer(dense_checked,accept_sparse=True)),('pca',PCA(param,svd_solver='randomized',random_state=42))])
  return Pipeline([('prep',prep(True)),('features',FeatureUnion([('originais',FunctionTransformer(accept_sparse=True)),('componentes',projection)])),('model',xgb_estimator())])
 if k=='048':return Pipeline([('features',FeatureUnion([('originais',prep(True)),('cluster',Pipeline([('date_categories',FunctionTransformer(representation_frame)),('prep',prep(True)),('cluster',ClusterID(param))]))])),('model',xgb_estimator())])
 if k=='049':return Pipeline([('features',FeatureUnion([('originais',prep()),('medoids',MedoidDistances(param))])),('model',xgb_estimator())])
 raise ValueError(k)

def supervised(k,data):
 params={'042':['media','stacking'],'043':[100,50,75],'044':[0,16,32],'045':[0,16,32],'048':[0,4,8,12],'049':[0,4,8,12]}[k]
 X,y=data['X_train'],data['y_train'];rows=[];folds=list(StratifiedKFold(3,shuffle=True,random_state=42).split(X,y))
 for param in params:
  path=REPORT/f'{k}_candidato_{param}.json';signature=context(k)
  if path.exists():
   r=json.loads(path.read_text())
   if r.get('context')==signature:rows.append(r['row']);continue
  scores=[];seconds=0.
  for fold,(a,b) in enumerate(folds):
   print(k,param,'fold',fold+1,flush=True);started=time.perf_counter()
   m=(Pipeline([('prep',prep(True)),('model',xgb_estimator())]) if k in ['044','045','048'] else make_xgb()) if param==0 else factory(k,param)
   m.fit(X.iloc[a],y.iloc[a]);scores.append(score(y.iloc[b],m.predict(X.iloc[b]))['macro_f1']);seconds+=time.perf_counter()-started
   del m;gc.collect()
  row=dict(param=param,folds=scores,mean=float(np.mean(scores)),std=float(np.std(scores,ddof=1)),seconds=seconds);rows.append(row)
  path.write_text(json.dumps(dict(context=signature,row=row)))
 best=max(rows,key=lambda r:r['mean']);delta=np.array(best['folds'])-np.array(rows[0]['folds'])
 paired=dict(control_param=rows[0]['param'],fold_differences=delta.tolist(),mean_difference=float(delta.mean()),std_difference=float(delta.std(ddof=1)))
 m=(Pipeline([('prep',prep(True)),('model',xgb_estimator())]) if k in ['044','045','048'] else make_xgb()) if best['param']==0 else factory(k,best['param']);m.fit(X,y)
 metrics=dict(train=score(y,m.predict(X)),validation=score(data['y_val'],m.predict(data['X_val'])))
 chart=_plot_confusion(k,data['y_val'],m.predict(data['X_val']))
 submission=None
 if best['param'] not in [0,100]:
  m=make_xgb() if best['param']==0 else factory(k,best['param']);full=data['prepared'];m.fit(full['X_train'],full['y_train']);submission=_compare_and_export(data,m.predict(full['X_test']),f'submission_{k}_novo_metodo.csv')
 return dict(candidates=rows,selected=best,paired_comparison=paired,metrics=metrics,confusion_matrix=chart,submission=submission,exploratory=metrics['validation']['macro_f1']<.4615,note=('042: bases XGBoost, regressão logística e Random Forest; RF escolhido em vez de CatBoost para maior diversidade estrutural e menor custo de memória, alteração explícita face à proposta inicial. Comparação com média dos mesmos modelos.' if k=='042' else 'Transformações ajustadas no treino de cada fold; controlo sem a nova representação incluído.'))


def descriptive_dbscan(data):
 X=representation_frame(data['X_train']);pp=build_preprocessor(X,normalize_numeric=True);Z=TruncatedSVD(16,random_state=42).fit_transform(pp.transform(X));Z=StandardScaler().fit_transform(Z)
 results=[]
 for minimum in [10,20]:
  nn=NearestNeighbors(n_neighbors=minimum,n_jobs=1).fit(Z);distances=nn.kneighbors(Z)[0][:,-1]
  for q in [.5,.75,.9]:
   eps=float(np.quantile(distances,q));sample=Z[np.random.default_rng(42).choice(len(Z),min(512,len(Z)),replace=False)]
   mean_neighbors=float(np.mean([len(x) for x in nn.radius_neighbors(sample,radius=eps,return_distance=False)]))
   if mean_neighbors*len(Z)>2_000_000:
    results.append(dict(min_samples=minimum,q=q,eps=eps,skipped='Pré-verificação: grafo estimado >2 milhões de arestas'));continue
   labels=DBSCAN(eps=eps,min_samples=minimum,n_jobs=1).fit_predict(Z);clusters=len(set(labels)-{-1});mask=labels>=0
   sil=None
   if clusters>=2 and mask.sum()>clusters:
    # Garantir que a amostra contém todos os clusters, sem consultar AGE_RANGE.
    rng=np.random.default_rng(42);members=np.flatnonzero(mask)
    mandatory=np.array([rng.choice(np.flatnonzero(labels==g)) for g in sorted(set(labels)-{-1})])
    pool=np.setdiff1d(members,mandatory);extra=rng.choice(pool,min(len(pool),1000-len(mandatory)),replace=False)
    sample_ids=np.concatenate([mandatory,extra])
    sil=float(silhouette_score(Z[sample_ids],labels[sample_ids]))
   results.append(dict(min_samples=minimum,q=q,eps=eps,clusters=clusters,noise=float(np.mean(labels==-1)),silhouette=sil,
      sizes={str(a):int(b) for a,b in pd.Series(labels).value_counts().items()},
      age_profile_posthoc=pd.crosstab(pd.Series(labels,index=X.index,name='cluster'),data['y_train'],normalize='index').to_dict()))
 eligible=[r for r in results if r.get('silhouette') is not None]
 selected=max(eligible,key=lambda r:r['silhouette']) if eligible else None
 profiles={};error_groups={}
 if selected:
  labels=DBSCAN(eps=selected['eps'],min_samples=selected['min_samples'],n_jobs=1).fit_predict(Z)
  for c in ['SEX','HATLEVEL','MAINSTAT','HHINCOME','HS1','HS2','REGION']:
   if c in data['X_train']:
    profiles[c]=pd.crosstab(pd.Series(labels,index=X.index,name='cluster'),data['X_train'][c].fillna('Ausente'),normalize='index').to_dict()
  # Erros do classificador obtidos fora da amostra de ajuste de cada fold.
  # Os clusters descrevem esta população, mas não entram no classificador.
  raw=data['X_train'];y=data['y_train'];oof=np.empty(len(raw),dtype=int)
  for fold,(a,b) in enumerate(StratifiedKFold(3,shuffle=True,random_state=42).split(raw,y)):
   print('050 erros OOF, fold',fold+1,flush=True)
   m=make_xgb().fit(raw.iloc[a],y.iloc[a]);oof[b]=m.predict(raw.iloc[b]);del m;gc.collect()
  gap=np.abs(oof-y.to_numpy())
  for label,mask in [('ruido',labels==-1),('agrupados',labels!=-1)]:
   if mask.any():error_groups[label]=dict(n=int(mask.sum()),accuracy=float(np.mean(oof[mask]==y.to_numpy()[mask])),mean_absolute_class_gap=float(gap[mask].mean()),class_gap_at_least_3=float(np.mean(gap[mask]>=3)))
 return dict(results=results,selected_unsupervised=selected,profiles=profiles,error_groups_oof=error_groups,submission=None,note='Seleção apenas por silhouette quando existem pelo menos dois clusters. Descritivo; sem predict nativo nem diagnóstico de anomalia. AGE_RANGE só usado a posteriori para descrever os grupos e os erros OOF de XGBoost. Os clusters não entram no classificador; diferenças entre códigos de escalão não equivalem a anos de idade.')


def permutation(data):
 from sklearn.model_selection import train_test_split
 m=make_xgb().fit(data['X_train'],data['y_train']);ids,_=train_test_split(np.arange(len(data['X_val'])),train_size=512,random_state=42,stratify=data['y_val'])
 X=data['X_val'].iloc[ids];y=data['y_val'].iloc[ids]
 r=permutation_importance(m,X,y,scoring='f1_macro',n_repeats=3,n_jobs=1,random_state=42)
 table=pd.DataFrame(dict(variable=X.columns,mean=r.importances_mean,std=r.importances_std)).sort_values('mean',ascending=False)
 table.to_csv(REPORT/'053_importancias.csv',index=False)
 fig,ax=plt.subplots(figsize=(9,7));top=table.head(20).iloc[::-1];ax.barh(top.variable,top['mean'],xerr=top['std']);ax.set(xlabel='Queda de Macro F1 (média ± DP)',title='Permutação de variáveis originais');fig.tight_layout();fig.savefig(REPORT/'053_permutacao.png',dpi=130);plt.close(fig)
 return dict(table=table.to_dict('records'),baseline=score(y,m.predict(X)),sample_size=len(X),repeats=3,submission=None,note='Desenvolvimento descritivo; correlação pode repartir a importância. Não usada para selecionar features nesta parcela.')


def shap_native(data):
    import xgboost as xgb

    m = make_xgb().fit(data["X_train"], data["y_train"])
    X = data["X_val"].sample(128, random_state=42)
    P = m.named_steps["preprocessor"]
    A = P.transform(X)

    booster = m.named_steps["model"].get_booster()
    d = xgb.DMatrix(A)
    contributions = booster.predict(d, pred_contribs=True, strict_shape=True)
    margin = booster.predict(d, output_margin=True, strict_shape=True)

    assert contributions.shape == (len(X), 16, A.shape[1] + 1)
    assert np.allclose(contributions.sum(axis=2), margin, atol=1e-4)

    names = P.preprocessor_.get_feature_names_out()
    original = sorted(X.columns, key=len, reverse=True)
    mapping = []

    for name in names:
        stripped = name.split("__", 1)[-1]
        column = next(
            c for c in original
            if stripped == c or stripped.startswith(c + "_")
        )
        mapping.append(column)

    mapping = np.asarray(mapping)
    feature_contributions = contributions[:, :, :-1]
    assert feature_contributions.shape[2] == len(mapping)

    grouped = np.zeros((len(X), 16, len(X.columns)), dtype=np.float32)
    for j, column in enumerate(X.columns):
        grouped[:, :, j] = feature_contributions[:, :, mapping == column].sum(axis=2)

    importance = np.abs(grouped).mean(axis=(0, 1))
    order = np.argsort(importance)[-20:]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(np.asarray(X.columns)[order], importance[order])
    ax.set(
        xlabel="Média |TreeSHAP| nas margens das 16 classes",
        title="Importância global por variável original",
    )
    fig.tight_layout()
    fig.savefig(REPORT / "054_shap_global.png", dpi=130)
    plt.close(fig)

    pred = m.predict(X)
    actual = data["y_val"].loc[X.index].to_numpy()
    examples = []

    for condition, label in [(pred == actual, "acerto"), (pred != actual, "erro")]:
        positions = np.flatnonzero(condition)
        if len(positions) == 0:
            continue

        i = int(positions[0])
        cls = int(pred[i])
        values = grouped[i, cls]
        top = np.argsort(np.abs(values))[-12:]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(
            np.asarray(X.columns)[top],
            values[top],
            color=np.where(values[top] >= 0, "#206a8b", "#d66b45"),
        )
        ax.set(
            xlabel=f"Contribuição para a margem da classe {cls}",
            title=f"Exemplo {label}: real {actual[i]}, previsto {cls}",
        )
        fig.tight_layout()
        fig.savefig(REPORT / f"054_shap_{label}.png", dpi=130)
        plt.close(fig)

        examples.append({
            "type": label,
            "real": int(actual[i]),
            "predicted": cls,
            "base": float(contributions[i, cls, -1]),
            "margin": float(margin[i, cls]),
        })

    return {
        "importance": dict(zip(X.columns, map(float, importance))),
        "examples": examples,
        "additivity_verified": True,
        "submission": None,
        "note": (
            "TreeSHAP nativo XGBoost, tree-path dependent. "
            "Contribuições nas margens, não nas probabilidades nem causas."
        ),
    }

def recompute_bmi(X):
 X=X.copy();name='BMI' if 'BMI' in X else 'IMC';X[name]=X.BM2/(X.BM1/100)**2;return X

def pdp(data):
 m=Pipeline([('derived',FunctionTransformer(recompute_bmi)),('classifier',make_xgb())]);m.fit(data['X_train'],data['y_train']);X=data['X_train'].loc[data['X_train'].BM1.notna() & data['X_train'].BM2.notna()].sample(256,random_state=42)
 for cls in [6,13]:
  fig,axes=plt.subplots(1,2,figsize=(12,4));PartialDependenceDisplay.from_estimator(m,X,['BM1','BM2'],target=cls,response_method='predict_proba',method='brute',kind='both',subsample=20,grid_resolution=15,percentiles=(.05,.95),random_state=42,ax=axes)
  fig.suptitle(f'PDP/ICE da pipeline original — classe {cls}');fig.tight_layout();fig.savefig(REPORT/f'055_pdp_ice_classe_{cls}.png',dpi=130);plt.close(fig)
 return dict(classes=[6,13],features=['BM1','BM2'],sample=256,submission=None,
  note='PDP/ICE da função das variáveis originais: o IMC derivado é recalculado após cada alteração de altura/peso. Não é um efeito de altura com IMC fixo, nem causal. Grelha limitada aos percentis 5–95 do treino; correlações continuam a limitar a interpretação.')


def ae_fit(X,epochs=100,latent=16,V=None,patience=10):
 """Sem alvo; MSE numérica mascarada + CE média por variável categórica."""
 import torch
 from torch import nn
 from torch.utils.data import DataLoader,TensorDataset
 torch.manual_seed(42);torch.set_num_threads(1);np.random.seed(42)
 X=representation_frame(X);pp=build_preprocessor(X,normalize_numeric=True);A=to_dense_float32(pp.transform(X));nnumeric=len(pp.numeric_columns_)
 sizes=[len(c) for c in pp.preprocessor_.named_transformers_['cat'].named_steps['onehot'].categories_]
 groups=torch.tensor(np.repeat(np.arange(len(sizes)),sizes),dtype=torch.long)
 class Net(nn.Module):
  def __init__(self):
   super().__init__();self.encoder=nn.Sequential(nn.Linear(A.shape[1],128),nn.ReLU(),nn.Linear(128,latent));self.decoder=nn.Sequential(nn.Linear(latent,128),nn.ReLU(),nn.Linear(128,A.shape[1]))
  def forward(self,a):return self.decoder(self.encoder(a))
 model=Net();opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
 def numeric_mask(frame):return torch.tensor(frame[pp.numeric_columns_].notna().to_numpy(),dtype=torch.float32)
 def loss_each(pred,truth,mask):
  number=((pred[:,:nnumeric]-truth[:,:nnumeric])**2*mask).sum(1)/mask.sum(1).clamp_min(1)
  logits=pred[:,nnumeric:];target=truth[:,nnumeric:];ix=groups[None,:].expand(len(pred),-1)
  maxima=torch.full((len(pred),len(sizes)),-float('inf'));maxima.scatter_reduce_(1,ix,logits.detach(),reduce='amax')
  sums=torch.zeros_like(maxima).scatter_add_(1,ix,torch.exp(logits-maxima[:,groups]))
  valid=torch.zeros_like(maxima).scatter_add_(1,ix,target)
  selected=torch.zeros_like(maxima).scatter_add_(1,ix,target*logits)
  category=(((torch.log(sums.clamp_min(1e-12))+maxima)*valid-selected).sum(1)/valid.sum(1).clamp_min(1))
  return .5*number+.5*category
 ta=torch.from_numpy(A);mask=numeric_mask(X);loader=DataLoader(TensorDataset(ta,mask),batch_size=128,shuffle=True)
 if V is not None:tv=torch.from_numpy(to_dense_float32(pp.transform(representation_frame(V))));mv=numeric_mask(V)
 history=[];best=float('inf');best_epoch=epochs;best_weights=None;stale=0
 for epoch in range(1,epochs+1):
  model.train()
  for a,m in loader:
   opt.zero_grad();loss=loss_each(model(a),a,m).mean();loss.backward();opt.step()
  model.eval()
  with torch.no_grad():
   train_loss=float(torch.cat([loss_each(model(ta[i:i+512]),ta[i:i+512],mask[i:i+512]) for i in range(0,len(ta),512)]).mean());val_loss=float(torch.cat([loss_each(model(tv[i:i+512]),tv[i:i+512],mv[i:i+512]) for i in range(0,len(tv),512)]).mean()) if V is not None else None
  history.append(dict(epoch=epoch,train_loss=train_loss,validation_loss=val_loss))
  if V is not None:
   if val_loss<best-1e-6:best=val_loss;best_epoch=epoch;best_weights=copy.deepcopy(model.state_dict());stale=0
   else:stale+=1
   if stale>=patience:break
  if epoch%20==0:print('AE',latent,'época',epoch,flush=True)
 if best_weights is not None:model.load_state_dict(best_weights)
 def embedding(frame):
  model.eval()
  with torch.no_grad():return torch.cat([model.encoder(torch.from_numpy(to_dense_float32(pp.transform(representation_frame(frame.iloc[i:i+512]))))) for i in range(0,len(frame),512)]).numpy()
 def errors(frame):
  model.eval();t=torch.from_numpy(to_dense_float32(pp.transform(representation_frame(frame))));m=numeric_mask(frame)
  with torch.no_grad():return torch.cat([loss_each(model(t[i:i+512]),t[i:i+512],m[i:i+512]) for i in range(0,len(t),512)]).numpy().astype(np.float32)
 return embedding,errors,dict(latent=latent,epoch=best_epoch,history=history,numeric_columns=list(pp.numeric_columns_),categorical_columns=list(pp.categorical_columns_),encoded_features=int(A.shape[1]),loss='0.5 MSE numérica observada + 0.5 CE média por variável categórica; categorias novas ignoradas no erro')


def augmented_classifier(X,y,Z,extra_fit,extra_eval):
 pp=build_preprocessor(X,normalize_numeric=False);A=sparse.hstack([csr(pp.transform(X)),csr(extra_fit)],format='csr');B=sparse.hstack([csr(pp.transform(Z)),csr(extra_eval)],format='csr')
 m=xgb_estimator();m.fit(A,y);return score(y,m.predict(A)),m.predict(B)


def autoencoder_latent(data):
 it,iv=data['X_inner_train'],data['X_inner_val'];yt,yv=data['y_inner_train'],data['y_inner_val'];rows=[]
 base=make_xgb().fit(it,yt);rows.append(dict(param=0,internal_macro_f1=score(yv,base.predict(iv))['macro_f1']))
 for dim in [16,32]:
  print('046 latente',dim,flush=True);embed,_,info=ae_fit(it,latent=dim,V=iv)
  _,pred=augmented_classifier(it,yt,iv,embed(it),embed(iv));rows.append(dict(param=dim,internal_macro_f1=score(yv,pred)['macro_f1'],ae=info));del embed;gc.collect()
 best=max(rows,key=lambda r:r['internal_macro_f1']);ae_reference=max(rows[1:],key=lambda r:r['internal_macro_f1'])
 X,y,Z=data['X_train'],data['y_train'],data['X_val'];submission=None
 if best['param']==0:
  m=make_xgb().fit(X,y);pred=m.predict(Z);metrics=dict(train=score(y,m.predict(X)),validation=score(data['y_val'],pred));fixed_confusion=_plot_confusion('046',data['y_val'],pred)
 else:
  embed,_,_=ae_fit(X,latent=best['param'],epochs=best['ae']['epoch']);tr,pred=augmented_classifier(X,y,Z,embed(X),embed(Z));metrics=dict(train=tr,validation=score(data['y_val'],pred));fixed_confusion=_plot_confusion('046',data['y_val'],pred)
  del embed;gc.collect();full=data['prepared'];embed,_,_=ae_fit(full['X_train'],latent=best['param'],epochs=best['ae']['epoch']);_,pred=augmented_classifier(full['X_train'],full['y_train'],full['X_test'],embed(full['X_train']),embed(full['X_test']));submission=_compare_and_export(data,pred,'submission_046_autoencoder_latente.csv')
 fig,ax=plt.subplots(figsize=(9,4))
 for row in rows[1:]:
  h=pd.DataFrame(row['ae']['history']);ax.plot(h.epoch,h.train_loss,'--',label=f"{row['param']} treino");ax.plot(h.epoch,h.validation_loss,label=f"{row['param']} validação")
 ax.set(xlabel='Época',ylabel='Perda equilibrada por blocos',title='046 — autoencoder');ax.legend();fig.tight_layout();fig.savefig(REPORT/'046_curvas.png',dpi=130);plt.close(fig)
 return dict(candidates=rows,selected=best,ae_reference=ae_reference,metrics=metrics,confusion_matrix=fixed_confusion,submission=submission,note='Autoencoder sem alvo; representação latente adicionada às features originais. Seleção interna, validação fixa de desenvolvimento.')


def crossfit_errors(X,others,reference):
 """Cada linha usa um AE que não a viu. Mesma regra em validação/teste.
 Tags são hashes do índice com sal fixo, independentes de AGE_RANGE.
 """
 tags=lambda F:np.array([hashlib.sha256(f'42:{v}'.encode()).digest()[0]%3 for v in F.index])
 tx=tags(X);out=np.zeros(len(X),np.float32);other=[np.zeros(len(F),np.float32) for F in others]
 for fold in range(3):
  print('047 AE cross-fit',fold,flush=True);fit=X.iloc[np.flatnonzero(tx!=fold)];_,error,_=ae_fit(fit,latent=reference['param'],epochs=reference['ae']['epoch'])
  hold=np.flatnonzero(tx==fold)
  if len(hold):out[hold]=error(X.iloc[hold])
  for i,F in enumerate(others):
   ids=np.flatnonzero(tags(F)==fold)
   if len(ids):other[i][ids]=error(F.iloc[ids])
  del error;gc.collect()
 return out[:,None],[v[:,None] for v in other]


def autoencoder_error(data):
 results=json.loads(RESULT.read_text())
 if results['046']['context']!=context('046'):raise ValueError('Configuração/dados do 046 mudaram; voltar a selecionar o autoencoder antes do 047.')
 reference=results['046']['ae_reference']
 it,iv=data['X_inner_train'],data['X_inner_val'];yt,yv=data['y_inner_train'],data['y_inner_val']
 train_error,(val_error,)=crossfit_errors(it,[iv],reference)
 base=make_xgb().fit(it,yt);control=score(yv,base.predict(iv))['macro_f1'];_,pred=augmented_classifier(it,yt,iv,train_error,val_error);aug=score(yv,pred)['macro_f1'];chosen=aug>control
 X,y,Z=data['X_train'],data['y_train'],data['X_val'];submission=None
 if chosen:
  err,(ev,)=crossfit_errors(X,[Z],reference);tr,pred=augmented_classifier(X,y,Z,err,ev);metrics=dict(train=tr,validation=score(data['y_val'],pred));fixed_confusion=_plot_confusion('047',data['y_val'],pred)
  full=data['prepared'];err,(et,)=crossfit_errors(full['X_train'],[full['X_test']],reference);_,pred=augmented_classifier(full['X_train'],full['y_train'],full['X_test'],err,et);submission=_compare_and_export(data,pred,'submission_047_erro_reconstrucao.csv')
 else:
  m=make_xgb().fit(X,y);pred=m.predict(Z);metrics=dict(train=score(y,m.predict(X)),validation=score(data['y_val'],pred));fixed_confusion=_plot_confusion('047',data['y_val'],pred)
 return dict(internal_control=control,internal_error_feature=aug,selected='erro' if chosen else 'controlo',metrics=metrics,confusion_matrix=fixed_confusion,submission=submission,
  note='Erro de reconstrução por cross-fitting: AE treinado sem a linha pontuada; uma tag hash independente do alvo escolhe um dos três AEs tanto no treino como em validação/teste. Não é um diagnóstico de anomalia.')


def mlp_stop(A,y,V,yv,patience=15,max_epochs=100):
 import torch
 from torch.utils.data import DataLoader,TensorDataset
 from src.experiments_016_018 import AgeRangeMLP,class_weights
 random.seed(42);np.random.seed(42);torch.manual_seed(42)
 model=AgeRangeMLP(A.shape[1],dropout=0.).to('cpu');criterion=torch.nn.CrossEntropyLoss(weight=class_weights(y,weighting='inverse_sqrt'))
 opt=torch.optim.Adam(model.parameters(),lr=.0001,weight_decay=0.)
 loader=DataLoader(TensorDataset(torch.from_numpy(A),torch.from_numpy(y)),batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(42),num_workers=0)
 hist={k:[] for k in ['epoch','train_loss','validation_loss','train_macro_f1','validation_macro_f1']};best=-1.;stale=0;epoch_best=1;weights=None
 for epoch in range(1,max_epochs+1):
  model.train()
  for x,t in loader:opt.zero_grad(set_to_none=True);loss=criterion(model(x),t);loss.backward();opt.step()
  model.eval()
  with torch.no_grad():
   a=model(torch.from_numpy(A));v=model(torch.from_numpy(V));hist['epoch'].append(epoch);hist['train_loss'].append(float(criterion(a,torch.from_numpy(y))));hist['validation_loss'].append(float(criterion(v,torch.from_numpy(yv))));hist['train_macro_f1'].append(score(y,a.argmax(1).numpy())['macro_f1']);value=score(yv,v.argmax(1).numpy())['macro_f1'];hist['validation_macro_f1'].append(value)
  if value>best:best=value;epoch_best=epoch;weights=copy.deepcopy(model.state_dict());stale=0
  else:stale+=1
  if stale>=patience:break
 model.load_state_dict(weights)
 assert np.isclose(score(yv,_predict_probabilities(model,V).argmax(1))['macro_f1'],best)
 return model,hist,epoch_best,epoch

def mlp_comparisons(k,data):
 """051: época fixa 75 vs paragem explícita; 052: ablação de normalização."""
 rows=[];it,iv=data['X_inner_train'],data['X_inner_val'];yt,yv=data['y_inner_train'].to_numpy(),data['y_inner_val'].to_numpy()
 for norm in ([True] if k=='051' else [True,False]):
  pp=build_preprocessor(it,normalize_numeric=norm);A=to_dense_float32(pp.transform(it));V=to_dense_float32(pp.transform(iv))
  if k=='051':
   model,h,epoch,stop=mlp_stop(A,yt,V,yv)
   choices=[('fixo_75',75),('early_stopping',epoch)]
  else:
   model,h,_=_fit_mlp(A,yt,100,.0001,weighting='inverse_sqrt',dropout=0.,X_monitor=V,y_monitor=yv)
   choices=[('normalizado' if norm else 'sem_normalizacao',int(np.argmax(h['validation_macro_f1'])+1))]
  for name,e in choices:
   model,current_history,_=_fit_mlp(A,yt,e,.0001,weighting='inverse_sqrt',dropout=0.,X_monitor=V,y_monitor=yv)
   if k=='051' and name=='early_stopping':current_history=h
   value=score(yv,_predict_probabilities(model,V).argmax(1))['macro_f1'];rows.append(dict(param=name,normalize=norm,epoch=e,internal_macro_f1=value,stop_epoch=stop if k=='051' else 100,history=current_history))
  del model,A,V;gc.collect()
 best=max(rows,key=lambda r:r['internal_macro_f1']);X,y,Z=data['X_train'],data['y_train'].to_numpy(),data['X_val']
 pp=build_preprocessor(X,normalize_numeric=best['normalize']);A=to_dense_float32(pp.transform(X));V=to_dense_float32(pp.transform(Z));m,_,_=_fit_mlp(A,y,best['epoch'],.0001,weighting='inverse_sqrt',dropout=0.)
 pred=_predict_probabilities(m,V).argmax(1);metrics=dict(train=score(y,_predict_probabilities(m,A).argmax(1)),validation=score(data['y_val'],pred))
 fixed_confusion=_plot_confusion(k,data['y_val'],pred)
 submission=None
 is_new=(k=='051' and best['param']=='early_stopping') or (k=='052' and not best['normalize'])
 if is_new:
  full=data['prepared'];pp=build_preprocessor(full['X_train'],normalize_numeric=best['normalize']);A=to_dense_float32(pp.transform(full['X_train']));V=to_dense_float32(pp.transform(full['X_test']));m,_,_=_fit_mlp(A,full['y_train'].to_numpy(),best['epoch'],.0001,weighting='inverse_sqrt',dropout=0.)
  submission=_compare_and_export(data,_predict_probabilities(m,V).argmax(1),f'submission_{k}_mlp.csv')
 fig,axes=plt.subplots(1,2,figsize=(12,4))
 for row in rows:
  h=row['history'];axes[0].plot(h['epoch'],h['train_loss'],'--',label=row['param']+' treino');axes[0].plot(h['epoch'],h['validation_loss'],label=row['param']+' validação');axes[1].plot(h['epoch'],h['validation_macro_f1'],label=row['param'])
 axes[0].set_title('Perda');axes[1].set_title('Macro F1 interna')
 for a in axes:a.set_xlabel('Época');a.legend(fontsize=7)
 fig.tight_layout();fig.savefig(REPORT/f'{k}_curvas.png',dpi=130);plt.close(fig)
 return dict(candidates=rows,selected=best,metrics=metrics,confusion_matrix=fixed_confusion,submission=submission,
  note='051 usa paragem real com patience=15 e restaura os melhores pesos; a métrica restaurada foi confirmada. Comparação com época fixa 75, política de um controlo anterior. 052 varia apenas normalização, selecionando época internamente.')

def run_experiment(k):
 REPORT.mkdir(exist_ok=True);signature=context(k);old=json.loads(RESULT.read_text()) if RESULT.exists() else {}
 if k in old and old[k].get('context')==signature:return old[k]
 data=load_experiment_data();started=time.perf_counter()
 forbidden={'AGE_RANGE','PID','HHID','PROXY','COUNTRY'}
 for name in ['X_train','X_val','X_inner_train','X_inner_val']:
  if forbidden.intersection(data[name].columns):raise ValueError(f'{name}: entradas proibidas presentes')
  target='y'+name[1:]
  if not data[name].index.equals(data[target].index):raise ValueError(f'{name}: desalinhamento do alvo')
 if set(data['X_train'].index)&set(data['X_val'].index):raise ValueError('Treino e validação sobrepostos')
 with threadpool_limits(limits=1):
  if k in ['042','043','044','045','048','049']:r=supervised(k,data)
  elif k=='046':r=autoencoder_latent(data)
  elif k=='047':r=autoencoder_error(data)
  elif k in ['051','052']:r=mlp_comparisons(k,data)
  elif k=='050':r=descriptive_dbscan(data)
  elif k=='053':r=permutation(data)
  elif k=='054':r=shap_native(data)
  elif k=='055':r=pdp(data)
  else:raise NotImplementedError(f'{k}: ainda não implementado, não apresentar como executado.')
 r.update(experiment=k,context=signature,seconds=time.perf_counter()-started,kaggle=None);write_result(k,r);return r
