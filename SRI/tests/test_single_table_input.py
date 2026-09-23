from pathlib import Path
import sys
import pandas as pd
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sri.simple_input import normalize_source_table

RAW=ROOT/'examples/raw_basic/Snake_Member_Catalogue.csv'

def test_parent_none_drops_blank_node_rows():
    n=normalize_source_table(RAW,parent=None)
    assert n.members.id_node.notna().all()
    assert n.metadata['bridge_rows']==0
    assert n.metadata['parent_mode']=='none'

def test_parent_true_keeps_group_blank_node_rows():
    n=normalize_source_table(RAW,parent=True)
    assert n.metadata['bridge_rows']>0
    assert n.members.id_node.isna().any()
    assert n.metadata['parent_mode']=='group_blank_node_rows'
    assert n.metadata['bridge_scope_mode']=='group'
    # With no scope column, every row within one Snake receives the same
    # synthetic internal part/scope regardless of any input id_part values.
    for _,g in n.members.groupby('Snake'):
        assert g.id_part.nunique()==1


def test_parent_true_does_not_require_id_part_column(tmp_path):
    d=pd.read_csv(RAW,dtype={'source_id':str,'id_node':str}).copy()
    if 'id_part' in d.columns:
        d=d.drop(columns=['id_part'])
    p=tmp_path/'no_id_part.csv'; d.to_csv(p,index=False)
    n=normalize_source_table(p,parent=True)
    assert n.metadata['bridge_rows']>0
    assert n.metadata['bridge_scope_mode']=='group'

def test_bridge_scope_col_is_optional_restriction_not_bridge_identifier():
    n=normalize_source_table(RAW,parent=True,bridge_scope_col='id_part')
    assert n.metadata['bridge_rows']>0
    assert n.metadata['parent_mode']=='group_blank_node_rows'
    assert n.metadata['bridge_scope_mode']=='column:id_part'
    assert n.members.id_part.nunique()>=2

def test_bridge_scope_requires_parent_true():
    with pytest.raises(ValueError):
        normalize_source_table(RAW,parent=None,bridge_scope_col='id_part')

def test_string_parent_mode_is_rejected():
    with pytest.raises(ValueError):
        normalize_source_table(RAW,parent='id_part')

def test_string_group_and_repeated_node_are_namespaced(tmp_path):
    d=pd.read_csv(RAW,dtype={'source_id':str,'id_node':str,'id_part':str}).head(20).copy()
    # Build two small groups sharing the same node label; we only test input
    # normalisation here, not the scientific score.
    a=d.iloc[:10].copy(); b=d.iloc[10:20].copy()
    a['sys']='A'; b['sys']='B'
    a['node']=['same']*5+['A2']*5
    b['node']=['same']*5+['B2']*5
    x=pd.concat([a,b],ignore_index=True)
    x['source_id']=[f'x{i}' for i in range(len(x))]
    p=tmp_path/'x.csv'; x.to_csv(p,index=False)
    n=normalize_source_table(p,group_col='sys',node_col='node',parent=None)
    assert n.members.Snake.nunique()==2
    vals=set(n.members.id_node.dropna().astype(str).unique())
    assert '0::same' in vals and '1::same' in vals
    assert 'A2' in vals and 'B2' in vals
