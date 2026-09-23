from types import SimpleNamespace
import numpy as np
from sri.kpd import partition,choose_core,review_state
from sri import SRIConfig

def test_core_tie_uses_actual_stars():
    assert choose_core(((0,1),(2,3)),[10,20,40,5])==(2,3)
    assert choose_core(((0,1),(2,3)),[10,20,15,15]) is None
    assert choose_core(((0,1,2),(3,4)),[2,2,2,100,100])==(0,1,2)

def test_recursive_velocity_split_and_exact_boundary():
    v=np.array([0.,1.,10.,11.,100.]);d=abs(v[:,None]-v[None,:])
    parts,_=partition(d)
    assert parts==((0,1),(2,3),(4,))
    v=np.array([0.,1.,3.]);d=abs(v[:,None]-v[None,:])
    assert len(partition(d)[0])==1

def test_missing_rv_channel_cannot_split_two_known_sides():
    v=np.array([0.,1.,10.]);d=abs(v[:,None]-v[None,:]);w=np.zeros_like(d)
    assert len(partition(d,w,limited=np.array([False,True,False]))[0])==1
    assert len(partition(d,w,limited=np.array([False,False,True]))[0])==2

def test_two_entities_unassessed():
    s=SimpleNamespace(labels=['a','b'],valid=np.array([True,True]),N_RV=np.array([4,4]),uvw=np.zeros((2,3)))
    row,core=review_state(s,SRIConfig(),True)
    assert row['KPD_mark'] is None and core is None


def test_rv_limited_entity_joins_nearest_known_block(monkeypatch):
    import sri.kpd as kpd
    rng=np.random.default_rng(13)
    labels=list('abcde')
    free={k:rng.normal(size=(24,2))+(0 if k in 'abe' else 100) for k in labels}
    spatial={k:rng.normal(size=(24,3)) for k in labels}
    s=SimpleNamespace(labels=labels,valid=np.array([True]*4+[False]),N_RV=np.array([5]*4+[0]),
        uvw=np.array([[0,0,0],[1,0,0],[10,0,0],[11,0,0],[np.nan]*3]),free=free,spatial=spatial)
    # Hold space and 2D cuts absent to isolate the approved assignment rule.
    monkeypatch.setattr(kpd,'cloud_matrices',lambda clouds:(np.ones((len(clouds),len(clouds)))-np.eye(len(clouds)),np.zeros((len(clouds),len(clouds)))))
    row,core=kpd.review_state(s,SRIConfig(),True)
    assert row['velocity3d_split'] and not row['tangent2d_split']
    assert set(core)==set('abe') and row['partition_complete']
    assert np.isnan(s.uvw[-1]).all()
