from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from sri.observables import derive_member_observables, derive_node_table
from sri.raw_prepare import load_catalogue


def _raw():
    return pd.read_csv(
        ROOT/'examples/raw_basic/Snake_Member_Catalogue.csv',
        dtype={'source_id':str,'id_node':str,'id_part':str},
    )


def test_basic_input_has_no_required_precomputed_coordinates():
    raw=_raw()
    for name in ['X','Y','Z','Vra','Vdec']:
        assert name not in raw.columns
    derived=derive_member_observables(raw)
    assert derived[['X','Y','Z','Vra','Vdec']].notna().any().all()


def test_known_first_source_coordinate_conversion():
    raw=_raw().iloc[:1].copy()
    d=derive_member_observables(raw).iloc[0]
    # Reference values are fixed regression targets and are tested
    # here solely as a coordinate-conversion regression target.
    assert abs(d['X'] - (-319.297475)) < 2e-5
    assert abs(d['Y'] - (-22.013949)) < 2e-5
    assert abs(d['Z'] - (-62.135980)) < 2e-5
    assert abs(d['Vra'] - (4.74047*float(raw.iloc[0].pmra)/float(raw.iloc[0].parallax))) < 1e-12
    assert abs(d['Vdec'] - (4.74047*float(raw.iloc[0].pmdec)/float(raw.iloc[0].parallax))) < 1e-12


def test_node_summaries_are_derived_from_members():
    derived=derive_member_observables(_raw())
    nodes=derive_node_table(derived)
    row=nodes.loc[nodes.id_node.eq('layer10_83459')].iloc[0]
    assert int(row.N)==163
    assert int(row.N_RV)==50
    assert abs(row.RA-85.55164541440192)<1e-10
    assert abs(row.DEC-12.776561359338675)<1e-10
    assert abs(row.RV-20.882021)<1e-10
    assert np.isfinite(row.Vra) and np.isfinite(row.Vdec)
    assert np.isfinite(row.e_Vra) and row.e_Vra>0
    assert np.isfinite(row.e_Vdec) and row.e_Vdec>0


def test_raw_loader_needs_member_and_optional_complex_only():
    nodes,stars,complexes=load_catalogue(ROOT/'examples/raw_basic')
    assert len(nodes)==7
    assert set(['X','Y','Z','Vra','Vdec']).issubset(stars.columns)
    assert set(['X','Y','Z','Vra','Vdec','RV','e_RV']).issubset(nodes.columns)
    assert set(nodes.Snake)=={0}
    assert set(complexes.Snake)=={0}

def test_astrometric_error_columns_are_optional():
    raw=_raw().drop(columns=[c for c in ['parallax_error','pmra_error','pmdec_error','parallax_pmra_corr','parallax_pmdec_corr'] if c in _raw().columns])
    derived=derive_member_observables(raw)
    nodes=derive_node_table(derived)
    assert len(nodes)==7
    assert nodes[['e_Vra','e_Vdec']].notna().all().all()
