"""Rebuild public figures/tables using original functions and aggregate inputs."""
from pathlib import Path
import importlib.util,json,hashlib
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def main():
    source=ROOT/'paper_outputs/five_market_pit_universe_benchmark_v8_1_1/reproducibility/build_data_assets.py'
    spec=importlib.util.spec_from_file_location('original_assets',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.REPORTS=ROOT/'results'
    module.OUTPUT=ROOT/'reproduced'
    module.TABLES=module.OUTPUT/'tables'
    module.FIGURES=module.OUTPUT/'figures'
    module.FIGURE_DATA=module.FIGURES/'figure_source_data'
    module.AUDITS=module.OUTPUT/'audit'
    module.OUT_MANIFESTS=module.OUTPUT/'manifests'
    module.ensure_dirs()
    module.build_main_tables();module.build_figures()
    checks=[]
    for folder in ['tables','figures/figure_source_data']:
        for saved in sorted((ROOT/folder).glob('*.csv')):
            rebuilt=module.OUTPUT/saved.relative_to(ROOT)
            a=pd.read_csv(saved);b=pd.read_csv(rebuilt)
            equal=a.equals(b)
            checks.append(dict(artifact=saved.relative_to(ROOT).as_posix(),rows=len(a),status='EXACT_MATCH' if equal else 'MISMATCH',original_sha256=hashlib.sha256(saved.read_bytes()).hexdigest(),regenerated_sha256=hashlib.sha256(rebuilt.read_bytes()).hexdigest()))
    result=dict(scope='PUBLIC_AGGREGATE_FIGURE_TABLE_REPRODUCTION',model_training=False,checks=checks,status='PASS' if len(checks)==14 and all(r['status']=='EXACT_MATCH' for r in checks) else 'FAIL')
    (module.OUTPUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
    return 0 if result['status']=='PASS' else 1

if __name__=='__main__':raise SystemExit(main())
