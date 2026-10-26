"""Public software and aggregate-data checks; no synthetic empirical claims."""
from pathlib import Path
import hashlib,json,importlib.util
import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]

def test_delivery_manifest():
    manifest=ROOT/'audit/PUBLIC_SHA256_MANIFEST.csv'
    assert manifest.exists(),'Delivery manifest must exist before release'
    rows=pd.read_csv(manifest)
    for r in rows.itertuples(index=False):
        p=ROOT/r.path
        assert p.is_file(),r.path
        assert p.stat().st_size==r.bytes,r.path
        assert hashlib.sha256(p.read_bytes()).hexdigest()==r.sha256,r.path

def test_confirmatory_family_and_figure_values():
    b=pd.read_csv(ROOT/'results/pit_multimarket_primary_bootstrap_10000_v8_1_1.csv')
    c=b[b.support_specification.eq('FULL_SUPPORT')&b.block_length.eq(20)&b.macro_scope.eq('FIVE_MARKET_MACRO')]
    assert len(c)==4
    assert c.contributing_market_count.eq(5).all()
    assert c.family_id.eq('CONFIRMATORY_MACRO_BRIER_BLOCK20').all()
    assert (c.ci_lower<=0).all() and (c.ci_upper>=0).all()
    f=pd.read_csv(ROOT/'figures/figure_source_data/figure_03_confirmatory_forest.csv')
    merged=c.merge(f,on='model_id',validate='one_to_one',suffixes=('_result','_figure'))
    for name in ['point_estimate','ci_lower','ci_upper','adjusted_p_value']:
        np.testing.assert_allclose(merged[name+'_result'],merged[name+'_figure'],rtol=0,atol=1e-14)

def test_block40_cannot_be_claimed_five_market():
    b=pd.read_csv(ROOT/'results/pit_multimarket_primary_bootstrap_10000_v8_1_1.csv')
    full=b[b.support_specification.eq('FULL_SUPPORT')&b.block_length.eq(40)]
    five=full[full.macro_scope.eq('FIVE_MARKET_MACRO')]
    assert len(five)==4 and not five.valid.any()
    four=full[full.macro_scope.eq('AVAILABLE_MARKET_MACRO')]
    assert len(four)==4 and four.contributing_market_count.eq(4).all()
    assert four.confirmatory_or_exploratory.eq('EXPLORATORY').all()

def test_bh_adjustment_from_actual_family():
    b=pd.read_csv(ROOT/'results/pit_multimarket_primary_bootstrap_10000_v8_1_1.csv')
    c=b[b.family_id.eq('CONFIRMATORY_MACRO_BRIER_BLOCK20')&b.valid]
    p=c.raw_p_value.to_numpy();order=np.argsort(p)
    adjusted=np.minimum(1,np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1])
    np.testing.assert_allclose(adjusted,c.adjusted_p_value.to_numpy()[order],atol=1e-14,rtol=0)

def test_feature_prefix_invariance_software_fixture():
    # Explicitly synthetic software fixture, not market evidence.
    from papers.paper_03_raw_ohlc_forecasting.src.features.raw_ohlc import build_lagged_raw_features
    n=30;close=100+np.arange(n)*0.1
    frame=pd.DataFrame(dict(market='fixture',symbol_canonical='fixture',date=pd.date_range('2000-01-01',periods=n),open_raw=close-0.01,high_raw=close+0.1,low_raw=close-0.1,close_raw=close))
    a,cols=build_lagged_raw_features(frame.iloc[:20]);b,_=build_lagged_raw_features(frame)
    pd.testing.assert_frame_equal(a[cols],b.iloc[:20][cols])

@pytest.mark.parametrize('module_name',[
 'papers.paper_03_raw_ohlc_forecasting.src.models.machine_learning.logistic_regression',
 'papers.paper_03_raw_ohlc_forecasting.src.models.machine_learning.lightgbm',
 'papers.paper_04_regime_transfer_reliability.src.models.machine_learning.random_forest',
])
def test_model_dependency_imports_and_registered_parameters(module_name):
    import importlib
    module=importlib.import_module(module_name)
    model=module.build_model(seed=11)
    params=model.get_params()
    for k,v in module.get_default_parameters().items():
        assert params.get(k,params.get('model__'+k))==v
