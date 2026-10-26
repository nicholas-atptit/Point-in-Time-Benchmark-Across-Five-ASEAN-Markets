from pathlib import Path
import sys,json,time,warnings
import pandas as pd,numpy as np
import argparse
parser=argparse.ArgumentParser(description='Refit first valid fold per market using all arms/models/seeds')
parser.add_argument('--evidence-root',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
R=args.evidence_root.resolve();sys.path.insert(0,str(R))
from scripts.point_in_time import pit_multimarket_forecasting_v8 as v8
from scripts.point_in_time import pit_multimarket_forecasting_v7 as v7
from scripts.point_in_time import pit_multimarket_forecasting_v6 as v6
P=R/'research_data/point_in_time';OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=True)
features=pd.read_parquet(P/'processed/pit_multimarket_feature_panel_v8.parquet')
folds=pd.read_parquet(P/'processed/pit_multimarket_fold_calendar_v8.parquet')
pred=pd.read_parquet(P/'processed/pit_multimarket_predictions_seed_level_v8.parquet',columns=['fold_id','index_id','universe_semantics','model_id','seed','security_id','prediction_date','target_date','prediction_probability','actual_class'])
results=[];start=time.time()
for index_id in ['IDX30','SET50','VN30','PSEI','FBM_KLCI']:
    fold=folds[folds.valid_fold.astype(bool)&folds.index_id.eq(index_id)].sort_values('test_start').iloc[0]
    for arm in v7.ARM_SEMANTICS.values():
        d=features[features.index_id.eq(index_id)&features.universe_semantics.eq(arm)&features.valid_for_model.astype(bool)]
        train=d[(d.prediction_date<fold.calibration_start)&(d.target_date<fold.calibration_start)]
        test=d[d.interval_id.eq(fold.interval_id)&(d.prediction_date>=fold.test_start)&(d.prediction_date<=fold.test_end)]
        for model in v7.MODEL_SPECS:
            for seed in v7.SEEDS:
                if model=='historical_class_probability':p=np.full(len(test),float(train.target_direction.mean()))
                else:
                    est=v6._build_model(model,seed)
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore');est.fit(train[v7.FEATURE_COLUMNS],train.target_direction.astype(int));p=v6._predict_probability(model,est,test[v7.FEATURE_COLUMNS])
                actual=test[['security_id','prediction_date','target_date']].assign(replayed=p)
                saved=pred[pred.fold_id.eq(fold.fold_id)&pred.index_id.eq(index_id)&pred.universe_semantics.eq(arm)&pred.model_id.eq(model)&pred.seed.eq(seed)]
                both=actual.merge(saved,on=['security_id','prediction_date','target_date'],validate='one_to_one')
                err=float((both.replayed-both.prediction_probability).abs().max())
                results.append(dict(index_id=index_id,fold_id=fold.fold_id,arm=arm,model_id=model,seed=seed,training_rows=len(train),predictions=len(both),max_abs_error=err,tolerance=1e-10,status='TOLERANCE_MATCH' if len(both)==len(saved)==len(test) and err<=1e-10 else 'MISMATCH'))
    print(index_id,len(results),time.time()-start,flush=True)
pd.DataFrame(results).to_csv(OUT/'sample_model_retraining.csv',index=False)
print(pd.DataFrame(results).groupby(['model_id','status']).agg(cells=('predictions','sum'),max_error=('max_abs_error','max')).to_string(),flush=True)
