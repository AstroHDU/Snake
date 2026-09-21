import pandas as pd
from run_sri import _public_table

def test_original_identifier_not_internal_index():
    original=pd.DataFrame({'Snake':[0,1],'rng_id':[0,1],'system_id':['P213','G106'],'SRI':[.5,.6]})
    out=_public_table(original)
    assert out.columns.tolist()==['system_id','SRI']
    assert out.system_id.tolist()==['P213','G106']
    assert 'Snake' in original

def test_empty_failure_schema():
    out=_public_table(pd.DataFrame(columns=['Snake','system_id','error']))
    assert out.columns.tolist()==['system_id','error']

def test_generic_group_cli(tmp_path,monkeypatch):
    import run_sri
    from pathlib import Path
    raw=pd.read_csv(Path(__file__).resolve().parents[1]/'examples/raw_basic/Snake_Member_Catalogue.csv',dtype={'source_id':str})
    raw=raw.rename(columns={'Snake':'association'});raw['association']='P213'
    source=tmp_path/'members.csv';raw.to_csv(source,index=False)
    monkeypatch.setattr(run_sri,'PARENT',None)
    monkeypatch.setattr(run_sri,'BRIDGE_SCOPE_COL',None)
    out=tmp_path/'out'
    assert run_sri.main(['--input',str(source),'--output',str(out),'--group-col','association'])==0
    for file in ['SRI.csv','KPD.csv','Core_SRI.csv','failures.csv','Derived_Entity_Kinematics.csv','Derived_Node_Catalogue.csv']:
        d=pd.read_csv(out/file,dtype={'system_id':str})
        assert 'Snake' not in d and 'rng_id' not in d
        assert 'system_id' in d
        if len(d):assert set(d.system_id)=={'P213'}
