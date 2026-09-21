from __future__ import annotations

"""User-facing single-table input normalisation for SRI.

The public runner accepts one source/member table with user-configurable
system and node columns.  It converts that table to the internal SRI schema
without requiring pre-computed XYZ or tangential velocities.

Bridge/background stars are identified *only* by a missing node label within
a scored group.  An optional bridge-scope column can further restrict which
nodes/entities those stars are allowed to support, but it is never required
to decide whether a row is a bridge/background row.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import numpy as np
import pandas as pd

BASE_NUMERIC_COLUMNS = ("ra", "dec", "parallax", "pmra", "pmdec")
OPTIONAL_RV_COLUMNS = ("radial_velocity", "radial_velocity_error")
OPTIONAL_ERROR_COLUMNS = (
    "ra_error", "dec_error", "parallax_error", "pmra_error", "pmdec_error",
    "ra_dec_corr", "ra_parallax_corr", "ra_pmra_corr", "ra_pmdec_corr",
    "dec_parallax_corr", "dec_pmra_corr", "dec_pmdec_corr",
    "parallax_pmra_corr", "parallax_pmdec_corr", "pmra_pmdec_corr",
)

@dataclass
class NormalizedInput:
    members: pd.DataFrame
    group_map: dict[int, str]
    node_map: dict[str, str]
    metadata: dict[str, Any]


def _missing_text(s: pd.Series) -> pd.Series:
    t=s.astype("string").str.strip()
    return s.isna() | t.isin(["", "nan", "None", "<NA>", "null", "NULL"])


def _internal_group_ids(values: pd.Series) -> tuple[np.ndarray, dict[int,str]]:
    raw=values.astype("string").str.strip()
    if raw.isna().any() or raw.eq("").any():
        raise ValueError("group column contains missing/blank values")
    uniques=list(pd.unique(raw))
    # Preserve exact integer identifiers when possible; otherwise use a stable
    # 0..N-1 internal mapping and always report the original system_id.
    parsed=[]; preserve=True
    for x in uniques:
        try:
            f=float(x); i=int(f)
            if not np.isfinite(f) or f!=i: preserve=False; break
            parsed.append(i)
        except Exception:
            preserve=False; break
    if preserve and len(set(parsed))==len(parsed):
        mapping={u:i for u,i in zip(uniques,parsed)}
        reverse={i:str(u) for u,i in zip(uniques,parsed)}
    else:
        mapping={u:i for i,u in enumerate(uniques)}
        reverse={i:str(u) for i,u in enumerate(uniques)}
    return raw.map(mapping).to_numpy(int), reverse


def normalize_source_table(
    csv_path: str|Path,
    *,
    group_col: str="Snake",
    node_col: str="id_node",
    source_col: str="source_id",
    parent: None|bool=None,
    bridge_scope_col: str|None=None,
) -> NormalizedInput:
    """Normalize one source-level CSV to the internal SRI member schema.

    Parameters
    ----------
    group_col
        Column identifying each Snake/system.
    node_col
        Column identifying the overdense/base node.  A blank value means that
        the row is *not* a node member.  Such rows can optionally be used as
        bridge/background stars when ``parent=True``.
    parent
        * ``None``/``False``: do not calculate bridge/background support;
          rows with blank ``node_col`` are ignored.
        * ``True``: within each group, all rows with blank ``node_col`` are
          treated as bridge/background rows for that group.

        Importantly, bridge/background identity is determined only by whether
        ``node_col`` is blank.  It is not determined by ``id_part`` or any
        other hierarchy column.
    bridge_scope_col
        Optional advanced restriction used only when ``parent=True``.  If
        supplied, this column is mapped to the internal ``id_part`` field so a
        bridge/background row can support only node/entity members with the
        same scope value.  If omitted, one synthetic scope is used per group,
        so every blank-node row in the group can support any compatible entity
        in that group.
    """
    csv_path=Path(csv_path)
    if not csv_path.exists(): raise FileNotFoundError(csv_path)
    header=pd.read_csv(csv_path,nrows=0).columns.tolist()
    for c in [group_col,node_col,source_col,*BASE_NUMERIC_COLUMNS]:
        if c not in header: raise ValueError(f"input missing required column: {c}")
    if bridge_scope_col is not None and bridge_scope_col not in header:
        raise ValueError(f"bridge scope column {bridge_scope_col!r} is not present")
    if bridge_scope_col is not None and not bool(parent):
        raise ValueError("bridge_scope_col is only meaningful when parent=True")

    dtype={source_col:"string",node_col:"string"}
    if bridge_scope_col is not None: dtype[bridge_scope_col]="string"
    raw=pd.read_csv(csv_path,dtype=dtype,float_precision="round_trip")
    if raw.empty: raise ValueError("input CSV is empty")
    if raw[source_col].isna().any(): raise ValueError("source_id column contains missing values")

    gid, reverse=_internal_group_ids(raw[group_col])
    work=raw.copy(); work["__sri_group__"]=gid
    source_text=work[source_col].astype("string")
    # A source may appear in two independently scored groups. Namespace only
    # that cross-group duplication. Repeated rows for the same source within
    # the same group are ambiguous and remain an input error.
    within=pd.DataFrame({"g":gid,"s":source_text}).duplicated(["g","s"],keep=False)
    if within.any():
        dup=pd.DataFrame({"g":gid,"s":source_text}).loc[within].head().to_dict("records")
        raise ValueError(f"source IDs must be unique within each group; examples: {dup}")
    cross=(pd.DataFrame({"g":gid,"s":source_text}).groupby("s")["g"].nunique()>1)
    cross_ids=set(cross[cross].index.astype(str))
    work["__sri_source__"]=[f"{int(g)}::{s}" if str(s) in cross_ids else str(s) for g,s in zip(gid,source_text)]
    node_missing=_missing_text(work[node_col])

    if parent is None or parent is False:
        # Explicitly no bridge/background correction.  Only node-assigned rows
        # are passed to the scientific scoring engine.
        work=work.loc[~node_missing].copy()
        if work.empty: raise ValueError("no node-assigned rows remain with parent=None")
        work["__sri_part__"]=work["__sri_group__"].astype(str)
        parent_mode="none"
        scope_mode="none"
    elif parent is True:
        if bridge_scope_col is None:
            # The intended public default when bridge/background support is
            # requested: all blank-node rows in a group share one synthetic
            # scope and can be assigned to the nearest compatible entity.
            work["__sri_part__"]=work["__sri_group__"].astype(str)
            parent_mode="group_blank_node_rows"
            scope_mode="group"
        else:
            # Advanced optional restriction.  Blank-node rows are still
            # identified solely by NODE_COL being missing.  The scope column
            # only controls which entities they may support.
            scope_missing=_missing_text(work[bridge_scope_col])
            keep=(~node_missing) | (~scope_missing)
            work=work.loc[keep].copy()
            node_missing=_missing_text(work[node_col])
            scope_missing=_missing_text(work[bridge_scope_col])
            work["__sri_part__"]=work[bridge_scope_col].astype("string")
            # Node rows without a scope stay usable by assigning them a
            # group-local fallback scope. They simply cannot capture bridge
            # rows carrying a different explicit scope.
            fallback=scope_missing & ~node_missing
            work.loc[fallback,"__sri_part__"]=(
                "__group__"+work.loc[fallback,"__sri_group__"].astype(str)
            )
            parent_mode="group_blank_node_rows"
            scope_mode=f"column:{bridge_scope_col}"
    else:
        raise ValueError("parent must be None, False, or True")

    # Ensure base-node labels are globally unique for internal preprocessing while
    # preserving the original label for user-facing output.
    original_node=work[node_col].astype("string")
    miss=_missing_text(work[node_col])
    pairs=pd.DataFrame({"g":work["__sri_group__"],"n":original_node})
    repeated_across=(pairs.loc[~miss].groupby("n")["g"].nunique()>1)
    repeated=set(repeated_across[repeated_across].index.astype(str))
    internal_node=[]; node_map={}
    for g,n,is_missing in zip(work["__sri_group__"],original_node,miss):
        if is_missing:
            internal_node.append(pd.NA); continue
        nn=str(n)
        internal=f"{int(g)}::{nn}" if nn in repeated else nn
        internal_node.append(internal); node_map[internal]=nn

    out=pd.DataFrame(index=work.index)
    out["Snake"]=work["__sri_group__"].astype(int)
    out["id_part"]=work["__sri_part__"].astype("string")
    out["id_node"]=pd.Series(internal_node,index=work.index,dtype="string")
    out["source_id"]=work["__sri_source__"].astype("string")
    for c in BASE_NUMERIC_COLUMNS:
        out[c]=pd.to_numeric(work[c],errors="coerce")
    for c in OPTIONAL_RV_COLUMNS:
        out[c]=pd.to_numeric(work[c],errors="coerce") if c in work else np.nan
    for c in OPTIONAL_ERROR_COLUMNS:
        if c in work: out[c]=pd.to_numeric(work[c],errors="coerce")

    # Node-assigned rows need finite basic astrometry. Bridge/background rows
    # with unusable astrometry are silently dropped because they cannot
    # contribute spatially.
    finite=np.ones(len(out),bool)
    for c in BASE_NUMERIC_COLUMNS: finite &= np.isfinite(out[c].to_numpy(float))
    finite &= out["parallax"].to_numpy(float)>0
    node_assigned=out["id_node"].notna().to_numpy()
    bad_node=node_assigned & ~finite
    if bad_node.any():
        examples=out.loc[bad_node,["Snake","id_node","source_id"]].head().to_dict("records")
        raise ValueError(f"node-assigned rows require finite ra/dec/parallax>0/pmra/pmdec; examples: {examples}")
    out=out.loc[node_assigned | finite].copy().reset_index(drop=True)

    node_counts=out.loc[out.id_node.notna()].groupby("Snake")["id_node"].nunique()
    bad=node_counts[node_counts<2]
    if len(bad):
        raise ValueError(f"each scored group needs at least two node labels; internal groups failing: {bad.to_dict()}")

    metadata={
        "input_csv":str(csv_path),"group_col":group_col,"node_col":node_col,
        "source_col":source_col,"parent_mode":parent_mode,
        "bridge_scope_col":bridge_scope_col,"bridge_scope_mode":scope_mode,
        "rows_input":int(len(raw)),"rows_used":int(len(out)),
        "groups":int(out.Snake.nunique()),"node_rows":int(out.id_node.notna().sum()),
        "bridge_rows":int(out.id_node.isna().sum()),
        # Metadata alias retained for readers that use the parent_rows field.
        "parent_rows":int(out.id_node.isna().sum()),
    }
    return NormalizedInput(out,reverse,node_map,metadata)


def save_mapping(path: str|Path, norm: NormalizedInput) -> None:
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps({
        "group_map":{str(k):v for k,v in norm.group_map.items()},
        "node_map":norm.node_map,"metadata":norm.metadata,
    },indent=2,ensure_ascii=False),encoding="utf-8")
