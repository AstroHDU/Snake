import numpy as np
from sri.core import cross_loss_measurement

def test_four_nonplanar_entities_compute_but_need_regularization():
    x=np.array([[1.,0,0],[0,1,0],[0,0,1],[2,2,2]])
    v=np.array([[0.,0,0],[1,1,0],[2,1,1],[3,2,1]])
    r=cross_loss_measurement(x,v,np.tile(np.eye(3),(4,1,1)),aggregation='mean_residual')
    assert r['status']=='computed' and np.isfinite(r['D_cross_kms'])
    assert not r['prediction_identifiable'].any()

def test_four_planar_targets_can_be_identifiable():
    x=np.array([[1.,1,0],[1,-1,0],[-1,1,0],[-1,-1,0]])
    r=cross_loss_measurement(x,x,np.tile(np.eye(3),(4,1,1)),aggregation='mean_residual')
    assert r['prediction_identifiable'].all()

def test_three_entities_still_inactive():
    r=cross_loss_measurement(np.eye(3),np.eye(3),np.tile(np.eye(3),(3,1,1)))
    assert r['status']=='insufficient_centers'
