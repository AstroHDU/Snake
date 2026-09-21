from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from sri import SRIConfig, score_state
from sri.kpd import choose_core

def _unique_largest_core(parts):
    return choose_core(parts,[1]*sum(map(len,parts)))
from sri.raw_prepare import load_catalogue, build_state, build_core_state


def test_unique_largest_core_321_keeps_only_three():
    parts=((0,1,2),(3,4),(5,))
    assert _unique_largest_core(parts)==(0,1,2)


def test_equal_largest_components_have_no_unique_core():
    assert _unique_largest_core(((0,1,2),(3,4,5))) is None
    assert _unique_largest_core(((0,1),(2,3),(4,))) is None


def test_core_state_preserves_only_selected_entities():
    nodes,stars,_=load_catalogue(ROOT/'examples/raw_basic')
    # Keep this unit test focused on fixed-core reconstruction rather than the
    # bridge/background Monte Carlo.
    stars=stars.loc[stars.id_node.notna()].copy()
    full=build_state(nodes,stars,compute_covariance=False)
    keep=tuple(full.labels[:3])
    core=build_core_state(full,nodes,stars,keep,compute_covariance=False)
    assert tuple(core.labels)==keep
    row,_=score_state(core,SRIConfig())
    assert row['Nentity']==3
    assert row['SRI']>=0
