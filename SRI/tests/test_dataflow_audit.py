from unittest.mock import patch
import numpy as np
import pandas as pd
from test_velocity_decoupling import fixture
from sri.raw_prepare import build_state
from sri.scorer import score_state
from sri_preprocess import background as bg,coordinates as co

def test_background_midrank_ties_not_double_counted():
    assert bg.midrank_percentile(1e-9,np.zeros(100))==.5
    for obs in [0.,1e-9,1.,2.]:
        assert 0<=bg.midrank_percentile(obs,np.array([0.,1e-9,1.,2.]))<=1

def test_background_velocities_do_not_enter_entity_or_nonspatial_scores():
    n,g=fixture();back=g.iloc[:8].copy();back['id_node']=pd.NA;back['source_id']='background_'+back.source_id.astype(str)
    a=pd.concat([g,back],ignore_index=True);b=a.copy();idx=b.id_node.isna()
    b.loc[idx,['Vra','Vdec','radial_velocity']]=[10000.,-10000.,20000.]
    b.loc[idx,'radial_velocity_error']=1.
    with patch('sri.raw_prepare.bg.N_BACKGROUND_ROTATIONS',3):
        x=build_state(n,a,compute_covariance=False);y=build_state(n,b,compute_covariance=False)
    assert x.components==y.components
    np.testing.assert_array_equal(x.uvw,y.uvw);np.testing.assert_array_equal(x.cov_analytic,y.cov_analytic)
    rx,_=score_state(x);ry,_=score_state(y)
    for k in ['S_mem','S_cen','V_mem','V_cen','Cross_quality','SRI']:np.testing.assert_allclose(rx[k],ry[k],equal_nan=True)
    assert sum(len(z) for z in x.prepared['entity_members'].values())==len(g)

def test_background_positions_only_change_spatial_support():
    n,g=fixture();back=g.iloc[:8].copy();back['id_node']=pd.NA;back['source_id']='background_'+back.source_id.astype(str)
    a=pd.concat([g,back],ignore_index=True);b=a.copy();b.loc[b.id_node.isna(),['X','Y','Z']]+=10000
    with patch('sri.raw_prepare.bg.N_BACKGROUND_ROTATIONS',3):
        x=build_state(n,a,compute_covariance=False);y=build_state(n,b,compute_covariance=False)
    assert x.components==y.components
    np.testing.assert_array_equal(x.xyz,y.xyz);np.testing.assert_array_equal(x.uvw,y.uvw)
    rx,_=score_state(x);ry,_=score_state(y)
    for k in ['S_cen','V_mem','V_cen','Cross_quality']:np.testing.assert_allclose(rx[k],ry[k],equal_nan=True)

def test_stale_model_columns_cannot_change_source_rebuild():
    n,g=fixture();mode='cartesian' if co.rv_complete_rows(n) else 'valid_rv_centroid'
    cached=co.add_member_score_columns(n,g,mode).copy()
    for col in cached:
        if col.endswith('_transport') and pd.api.types.is_numeric_dtype(cached[col]):cached[col]=0.
    x=build_state(n,g,compute_covariance=False);y=build_state(n,cached,compute_covariance=False)
    assert x.components==y.components
    np.testing.assert_array_equal(x.uvw,y.uvw)
