from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sri.raw_prepare import load_catalogue,build_state
from sri.config import SRIConfig
from sri.scorer import score_state
from sri.measured_kinematics import estimate_measured_entity
from sri.uncertainty import UncertaintyError
from sri.core import cross_loss_measurement,vcen_local

ROOT=Path(__file__).resolve().parents[1]
def inputs():
    n,g,_=load_catalogue(ROOT/'examples/raw_basic')
    return n,g[g.id_node.notna()].copy()

@pytest.mark.parametrize('field',['pmra_error','parallax_error','radial_velocity_error'])
def test_strict_errors_fail_entire_state(field):
    n,g=inputs();i=g.index[g.radial_velocity.notna()][0];g.loc[i,field]=np.nan
    with pytest.raises(UncertaintyError,match=field):build_state(n,g,use_uncertainty=True)

def test_no_rv_is_not_missing_required_rv_error():
    n,g=inputs();i=g.index[0];g.loc[i,['radial_velocity','radial_velocity_error']]=np.nan
    s=build_state(n,g,use_uncertainty=True)
    assert np.isfinite(score_state(s)[0]['SRI'])

def test_nominal_mode_ignores_missing_or_changed_errors():
    n,g=inputs();a=build_state(n,g,use_uncertainty=False)
    for c in g:
        if 'error' in c or c.endswith('_corr'):g[c]=np.nan
    b=build_state(n,g,use_uncertainty=False)
    cfg=SRIConfig(use_uncertainty=False)
    assert a.labels==b.labels
    np.testing.assert_allclose(a.uvw,b.uvw,equal_nan=True)
    ra,_=score_state(a,cfg);rb,_=score_state(b,cfg)
    for c in ['S_mem','V_mem','S_cen','V_cen','Cross_quality','SRI']:
        np.testing.assert_allclose(ra[c],rb[c],equal_nan=True)
    with pytest.raises(ValueError,match='uncertainty mode'):score_state(b)

def test_bottom_level_missing_covariance_never_becomes_zero():
    v=np.array([[0,0,0],[1,0,0],[2,1,0],[3,0,1]],float)
    c=np.tile(np.eye(3),(4,1,1));c[0]=np.nan
    with pytest.raises(ValueError,match='covariance'):vcen_local(v,c)
    with pytest.raises(ValueError,match='covariance'):cross_loss_measurement(v,v,c)

def test_batch_records_failure_and_continues(tmp_path):
    import run_sri
    _,g=inputs();a=g.copy();a['Snake']='bad';b=g.copy();b['Snake']='good'
    b['source_id']='good_'+b.source_id.astype(str)
    a.loc[a.index[0],'pmra_error']=np.nan
    source=tmp_path/'batch.csv';pd.concat([a,b]).to_csv(source,index=False)
    out=tmp_path/'out'
    rc=run_sri.main(['--input',str(source),'--output',str(out),'--use-uncertainty','--no-kpd'])
    assert rc==2
    scores=pd.read_csv(out/'SRI.csv');fail=pd.read_csv(out/'failures.csv')
    assert scores.system_id.tolist()==['good']
    assert fail.system_id.tolist()==['bad'] and 'pmra_error' in fail.error.iloc[0]
    assert 'source=' in fail.error.iloc[0]
    out2=tmp_path/'nominal'
    assert run_sri.main(['--input',str(source),'--output',str(out2),'--no-use-uncertainty','--no-kpd'])==0
    assert len(pd.read_csv(out2/'SRI.csv'))==2

def test_core_error_invalidates_full_output(tmp_path,monkeypatch):
    import run_sri
    _,g=inputs();g['Snake']='core_bad'
    src=tmp_path/'core.csv';g.to_csv(src,index=False)
    monkeypatch.setattr(run_sri,'review_state',lambda s,cfg,return_core_labels: ({'candidate':True,'KPD_mark':1,'partition':';'.join(s.labels),'core_selection_status':'selected'},s.labels[:2]))
    def fail(*args,**kwargs):raise UncertaintyError('entity=test: missing_core_covariance')
    monkeypatch.setattr(run_sri,'build_core_state',fail)
    out=tmp_path/'out'
    assert run_sri.main(['--input',str(src),'--output',str(out),'--use-uncertainty'])==2
    f=pd.read_csv(out/'failures.csv')
    assert 'missing_core_covariance' in f.error.iloc[0]
    assert len(pd.read_csv(out/'Derived_Entity_Kinematics.csv'))==0
    assert 'Gold' not in (out/'SRI.csv').read_text()

def test_component_nominal_needs_no_error_columns():
    _,g=inputs()
    a=estimate_measured_entity(g,method='component_median',use_uncertainty=False)
    for c in g:
        if 'error' in c:g[c]=np.nan
    b=estimate_measured_entity(g,method='component_median',use_uncertainty=False)
    assert a['valid'] and b['valid']
    np.testing.assert_allclose(a['uvw'],b['uvw'])
