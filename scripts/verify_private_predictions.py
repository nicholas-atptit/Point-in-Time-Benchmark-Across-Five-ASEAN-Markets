from pathlib import Path
import sys,json,hashlib,csv,importlib.util
import pandas as pd,numpy as np,pyarrow.parquet as pq

import argparse
parser=argparse.ArgumentParser(description='Read-only independent audit of lawfully held private PIT evidence')
parser.add_argument('--evidence-root',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
R=args.evidence_root.resolve(); ROOT=R
sys.path.insert(0,str(R))
P=R/'research_data/point_in_time'; OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,default=str,allow_nan=False),encoding='utf-8')
def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,list):return [clean(x) for x in v]
    if isinstance(v,(np.integer,)):return int(v)
    if isinstance(v,(np.floating,)):return float(v) if np.isfinite(v) else None
    return v

profiles=[];schema=[]
paths=sorted((P/'processed').glob('*v8.parquet'))+sorted((P/'reconstructed').glob('*v8.parquet'))
for path in paths:
    pf=pq.ParquetFile(path); cols=pf.schema_arrow.names
    nulls={c:0 for c in cols}; invalid_ohlc=0
    for batch in pf.iter_batches(batch_size=200000):
        for i,c in enumerate(cols):nulls[c]+=batch.column(i).null_count
        if {'open','high','low','close'}<=set(cols):
            d=batch.select(['open','high','low','close']).to_pandas()
            invalid_ohlc+=int((~np.isfinite(d).all(axis=1)|d.le(0).any(axis=1)|(d.high<d[['open','close','low']].max(axis=1))|(d.low>d[['open','close','high']].min(axis=1))).sum())
    key=next((k for k in [
        ['interval_id','universe_semantics','fold_id','security_id','prediction_date','target_date','model_id','seed'],
        ['interval_id','universe_semantics','fold_id','security_id','prediction_date','target_date','model_id'],
        ['index_id','universe_semantics','security_id','prediction_date'],
        ['index_id','universe_semantics','security_id','membership_date'],
        ['interval_id','security_id'],['market','security_id','date'],['fold_id']
    ] if set(k)<=set(cols)),None)
    smallcols=list(dict.fromkeys((key or [])+[c for c in ['market','index_id','security_id','date','prediction_date','membership_date','interval_start','target_date'] if c in cols]))
    d=pd.read_parquet(path,columns=smallcols)
    dates=[c for c in ['date','prediction_date','membership_date','interval_start'] if c in d]
    profile=dict(path=path.relative_to(R).as_posix(),sha256=sha(path),bytes=path.stat().st_size,rows=pf.metadata.num_rows,columns=len(cols),primary_key=key,duplicate_keys=int(d.duplicated(key).sum()) if key else None,markets=sorted(d.market.unique()) if 'market' in d else [],securities=int(d.security_id.nunique()) if 'security_id' in d else None,date_min=str(d[dates[0]].min()) if dates else None,date_max=str(d[dates[0]].max()) if dates else None,nulls=nulls,invalid_ohlc_rows=invalid_ohlc,redistribution='WITHHELD_RIGHTS_UNESTABLISHED')
    profiles.append(profile)
    schema += [dict(path=profile['path'],column=f.name,dtype=str(f.type),null_count=nulls[f.name]) for f in pf.schema_arrow]
    print(path.name,pf.metadata.num_rows,flush=True)
write('data_profiles.json',clean(profiles));pd.DataFrame(schema).to_csv(OUT/'data_schema.csv',index=False)

pred=pd.read_parquet(P/'processed/pit_multimarket_predictions_seed_collapsed_v8.parquet')
keys=['interval_id','market','index_id','universe_semantics','model_id','prediction_date']
pred['_brier']=(pred.prediction_probability-pred.actual_class)**2
prob=pred.prediction_probability.clip(1e-12,1-1e-12)
pred['_logloss']=-(pred.actual_class*np.log(prob)+(1-pred.actual_class)*np.log(1-prob))
date=pred.groupby(keys,sort=True).agg(brier_score=('_brier','mean'),log_loss=('_logloss','mean'),prediction_count=('actual_class','size')).reset_index()
members=pd.read_parquet(P/'processed/pit_membership_daily_all_arms_v8.parquet')
comp=members.groupby(['interval_id','membership_date','universe_semantics']).security_id.nunique().unstack().reset_index().rename(columns={'membership_date':'prediction_date','POINT_IN_TIME':'PIT_constituent_count','CURRENT_2026_APPLIED_BACKWARD':'current_constituent_count','PIT_CURRENT_INTERSECTION_CONTROL':'intersection_count'})
comp['prediction_date']=pd.to_datetime(comp.prediction_date).dt.date
date=date.merge(comp[['interval_id','prediction_date','PIT_constituent_count','current_constituent_count','intersection_count']],on=['interval_id','prediction_date'],validate='many_to_one')
denom=date.universe_semantics.map({'POINT_IN_TIME':'PIT_constituent_count','CURRENT_2026_APPLIED_BACKWARD':'current_constituent_count','PIT_CURRENT_INTERSECTION_CONTROL':'intersection_count'})
date['prediction_coverage']=[n/date.iloc[i][c] for i,(n,c) in enumerate(zip(date.prediction_count,denom))]
metrics=['brier_score','log_loss','prediction_coverage']
interval=date.groupby(keys[:-1])[metrics].mean().reset_index()
market=interval.groupby(['market','index_id','universe_semantics','model_id'])[metrics].mean().reset_index()
macro=market.groupby(['universe_semantics','model_id'])[metrics].mean().reset_index()
comparisons=[]
for level,frame,filename,on in [
 ('date',date,'pit_multimarket_performance_by_date_v8.csv',keys),
 ('interval',interval,'pit_multimarket_performance_by_interval_v8.csv',keys[:-1]),
 ('market',market,'pit_multimarket_performance_by_market_v8.csv',['market','index_id','universe_semantics','model_id']),
 ('macro',macro,'pit_multimarket_equal_market_macro_v8.csv',['universe_semantics','model_id'])]:
    saved=pd.read_csv(P/'reports'/filename)
    if 'prediction_date' in on:saved['prediction_date']=pd.to_datetime(saved.prediction_date).dt.date
    both=frame.merge(saved,on=on,suffixes=('_actual','_saved'),validate='one_to_one')
    for metric in metrics:
        err=float((both[metric+'_actual']-both[metric+'_saved']).abs().max())
        comparisons.append(dict(component=level,metric=metric,cells=len(both),expected_rows=len(saved),recomputed_rows=len(frame),max_abs_error=err,tolerance=1e-12,status='TOLERANCE_MATCH' if err<=1e-12 and len(both)==len(saved)==len(frame) else 'MISMATCH'))
pd.DataFrame(comparisons).to_csv(OUT/'independent_metric_verification.csv',index=False)
macro.to_csv(OUT/'recomputed_macro.csv',index=False)

temporal={}
for a,b in [('training_end','prediction_date'),('label_maturity_cutoff','prediction_date'),('prediction_date','target_date')]:temporal[a+'_before_'+b]=int((pd.to_datetime(pred[a])>=pd.to_datetime(pred[b])).sum())
temporal['nonfinite_probabilities']=int((~np.isfinite(pred.prediction_probability)).sum())
temporal['out_of_range_probabilities']=int((~pred.prediction_probability.between(0,1)).sum())
temporal['collapsed_seed_count_not_five']=int(pred.seed_count.ne(5).sum())
join_keys=['market','index_id','interval_id','fold_id','model_id','security_id','prediction_date','target_date','actual_class']
pit=pred[pred.universe_semantics.eq('POINT_IN_TIME')]
cur=pred[pred.universe_semantics.eq('CURRENT_2026_APPLIED_BACKWARD')]
common=pit[join_keys+['prediction_probability']].merge(cur[join_keys+['prediction_probability']],on=join_keys,validate='one_to_one')
temporal['exact_common_support_pairs']=len(common)
write('temporal_and_support_checks.json',temporal)

# Reuse the original bootstrap engine on independently recomputed date deltas.
from scripts.point_in_time import run_pit_v8_1_release as v81
from scripts.point_in_time import run_pit_v8_1_1_release as patch
dd=date[date.universe_semantics.eq('CURRENT_2026_APPLIED_BACKWARD')].merge(date[date.universe_semantics.eq('POINT_IN_TIME')],on=['interval_id','market','index_id','model_id','prediction_date'],suffixes=('_current','_pit'),validate='one_to_one')
dd['delta_brier_score_current_minus_pit']=dd.brier_score_current-dd.brier_score_pit
boot=v81._bootstrap_family(dd,support='FULL_SUPPORT',metrics=['brier_score'],samples=10000)
boot=patch.patch_bootstrap_schema(v81.apply_bh_families(boot))
if isinstance(boot,tuple):boot=boot[0]
saved=pd.read_csv(P/'reports/pit_multimarket_primary_bootstrap_10000_v8_1_1.csv')
on=['support_specification','model_id','block_length','macro_scope','index_id']
saved=saved[saved.support_specification.eq('FULL_SUPPORT')]
both=boot.merge(saved,on=on,suffixes=('_actual','_saved'),validate='one_to_one')
bs=[]
for c in ['point_estimate','ci_lower','ci_upper','raw_p_value','adjusted_p_value']:
    a=both[c+'_actual'];b=both[c+'_saved'];err=float((a-b).abs().max())
    bs.append(dict(metric=c,cells=len(both),max_abs_error=err,tolerance=1e-12,status='TOLERANCE_MATCH' if err<=1e-12 and (a.isna()==b.isna()).all() and len(boot)==len(saved)==len(both) else 'MISMATCH'))
pd.DataFrame(bs).to_csv(OUT/'bootstrap_verification.csv',index=False)
boot.to_csv(OUT/'recomputed_bootstrap_full_support.csv',index=False)
write('summary.json',clean(dict(datasets=len(profiles),metrics=comparisons,bootstrap=bs,temporal=temporal)))
print(json.dumps(clean(dict(metrics=comparisons,bootstrap=bs,temporal=temporal)),indent=2),flush=True)
