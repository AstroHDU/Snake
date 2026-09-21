#!/usr/bin/env python3
"""One-command SRI runner.

Typical use
-----------
1. Put ONE source/member CSV in ``input/``.
2. Edit the small USER DEFAULTS block below if your group/node columns differ.
3. Run::

       python run_sri.py

You can override every user default from the command line; run
``python run_sri.py --help`` for details.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import shutil
import sys
import tempfile
import time
import warnings
from dataclasses import replace
import pandas as pd
from tqdm.auto import tqdm


# Relative input/output paths are resolved from the chosen working directory.
ROOT = Path.cwd().resolve()

from sri import SRIConfig, score_state, review_state
from sri.raw_prepare import load_catalogue, build_state, build_core_state
from sri.entity_kinematics import diagnostic_table
from sri.observables import derive_node_table
from sri.simple_input import normalize_source_table, save_mapping



# ---------------------------------------------------------------------------
# USER DEFAULTS -- most users only need to edit this block.
# ---------------------------------------------------------------------------
INPUT = './input/Snake_Member_ms10.csv'         # auto-detect the single CSV in ./input/
OUTPUT = None         # None -> ./output/<input_csv_stem>/
GROUP_COL = "Snake"   # column identifying each system/Snake
NODE_COL = "id_node"  # base-node label; blank means non-node support-candidate row
SOURCE_COL = "source_id"

# Bridge/background-star policy:
#   None/False : do NOT calculate bridge/background support; blank-node rows ignored.
#   True       : within each GROUP, rows with blank NODE_COL are bridge/background support candidates.
# Bridge/background identity is ALWAYS determined by NODE_COL being blank.
PARENT = True

# Optional advanced restriction used only when PARENT=True.
# None      : every blank-node row may support any compatible entity in its GROUP.
# "id_part" : a blank-node row may support only nodes/entities with the same id_part.
# This column never decides whether a row is a bridge/background support candidate.
BRIDGE_SCOPE_COL = 'id_part'

CONFIG = None  # None uses packaged scientific defaults; or set an explicit JSON path
SAVE_DERIVED_TABLES = True
RUN_KPD = True
COMPUTE_COVARIANCE = False  # optional duplicate diagnostic storage; USE_UNCERTAINTY controls scoring
USE_UNCERTAINTY = None  # None follows config/defaults.json (true); True/False explicitly overrides it
# ---------------------------------------------------------------------------


def _parse_parent(value):
    if value is None: return PARENT
    if isinstance(value,bool): return value
    s=str(value).strip().lower()
    if s in {"none","null","false","off","0"}: return None
    if s in {"true","on","1","group"}: return True
    raise SystemExit("--parent accepts only none/false or true. Use --bridge-scope-col for an optional scope column.")


def _resolve_input(value) -> Path:
    if value is None:
        folder=ROOT/"input"
        folder.mkdir(exist_ok=True)
        files=sorted(p for p in folder.glob("*.csv") if p.is_file())
        if len(files)!=1:
            names=[p.name for p in files]
            raise SystemExit(
                "INPUT is None, so run_sri.py expects exactly one CSV in ./input/. "
                f"Found {len(files)}: {names}. Set INPUT at the top of run_sri.py "
                "or use --input PATH."
            )
        return files[0]
    p=Path(value).expanduser()
    if p.is_dir():
        files=sorted(p.glob("*.csv"))
        if len(files)!=1:
            raise SystemExit(f"directory input requires exactly one CSV; found {[x.name for x in files]}")
        return files[0]
    if not p.is_absolute():
        # Prefer a path relative to the current working directory; fall back to
        # the package root for convenient INPUT='input/foo.csv' edits.
        cwd=(Path.cwd()/p).resolve()
        p=cwd if cwd.exists() else (ROOT/p).resolve()
    if not p.exists(): raise SystemExit(f"input CSV not found: {p}")
    return p


def _resolve_output(value, input_csv: Path) -> Path:
    if value is None:
        return ROOT/"output"/input_csv.stem
    p=Path(value).expanduser()
    # A bare name is interpreted as ./output/<name>; an explicit path is used
    # as given (relative paths are resolved from the current working directory).
    if not p.is_absolute() and p.parent==Path('.'):
        return ROOT/"output"/p.name
    return p if p.is_absolute() else (Path.cwd()/p).resolve()


def _replace_node_labels(text, mapping):
    if not isinstance(text,str) or not text: return text
    # Longest first avoids a short label replacing a prefix of a longer one.
    for internal, original in sorted(mapping.items(),key=lambda kv:len(kv[0]),reverse=True):
        text=text.replace(internal,original)
    return text


def _public_table(frame):
    """Keep original system_id; internal numeric IDs belong in mapping JSON."""
    out=frame.drop(columns=['Snake','rng_id'],errors='ignore').copy()
    if 'system_id' in out:
        out=out[['system_id']+[c for c in out if c!='system_id']]
    return out


def _filter_system_ids(norm, requested):
    if not requested: return None
    wanted={str(x).strip() for x in requested.split(',') if str(x).strip()}
    reverse=norm.group_map
    ids=[k for k,v in reverse.items() if str(v) in wanted or str(k) in wanted]
    missing=wanted-{str(reverse[k]) for k in ids}-{str(k) for k in ids}
    if missing: warnings.warn(f"requested systems not found: {sorted(missing)}")
    return set(ids)


def main(argv=None):
    ap=argparse.ArgumentParser(description="Run SRI directly from one source/member CSV.")
    ap.add_argument("--input",default=None,help="Source CSV or directory containing exactly one CSV. Default: ./input/*.csv")
    ap.add_argument("--output",default=None,help="Output folder/name. Bare name -> ./output/<name>; default -> ./output/<input stem>/")
    ap.add_argument("--group-col",default=None,help=f"System/group column (default {GROUP_COL!r})")
    ap.add_argument("--node-col",default=None,help=f"Node column (default {NODE_COL!r})")
    ap.add_argument("--source-col",default=None,help=f"Source ID column (default {SOURCE_COL!r})")
    ap.add_argument("--parent",default=None,help="none | true. true means blank NODE_COL rows are bridge/background support candidates within each group")
    ap.add_argument("--bridge-scope-col",default=None,help="Optional scope column used only with --parent true (e.g. id_part)")
    ap.add_argument("--config",type=Path,default=None,help="Optional scoring JSON; otherwise use packaged scientific defaults")
    ap.add_argument("--systems",default=None,help="Optional comma-separated original group values/internal IDs")
    ap.add_argument("--no-kpd",action="store_true",help="Skip review partition and retained-core SRI diagnostic")
    ap.add_argument("--no-derived",action="store_true",help="Do not save derived node table")
    ap.add_argument('--use-uncertainty',action=argparse.BooleanOptionalAction,default=USE_UNCERTAINTY,help='Strict errors or explicit nominal mode; never auto-detect')
    args=ap.parse_args(argv)

    input_csv=_resolve_input(args.input if args.input is not None else INPUT)
    output_dir=_resolve_output(args.output if args.output is not None else OUTPUT,input_csv)
    group_col=args.group_col or GROUP_COL; node_col=args.node_col or NODE_COL; source_col=args.source_col or SOURCE_COL
    parent=_parse_parent(args.parent)
    bridge_scope_col=args.bridge_scope_col if args.bridge_scope_col is not None else BRIDGE_SCOPE_COL
    if bridge_scope_col is not None and not bool(parent):
        raise SystemExit("BRIDGE_SCOPE_COL/--bridge-scope-col requires PARENT=True/--parent true")
    config_path=args.config or CONFIG
    cfg=SRIConfig.from_json(config_path) if config_path is not None else SRIConfig()
    if args.use_uncertainty is not None:cfg=replace(cfg,use_uncertainty=args.use_uncertainty)

    if output_dir.exists():
        # Keep behaviour explicit and reproducible: outputs from a previous run
        # are preserved in place only if the user chose that path, but result
        # files produced below are overwritten.  No input files are touched.
        output_dir.mkdir(parents=True,exist_ok=True)
    else: output_dir.mkdir(parents=True,exist_ok=True)

    t0=time.time()
    norm=normalize_source_table(
        input_csv,group_col=group_col,node_col=node_col,source_col=source_col,
        parent=parent,bridge_scope_col=bridge_scope_col,
    )
    selected=_filter_system_ids(norm,args.systems)
    if selected is not None:
        norm.members=norm.members.loc[norm.members.Snake.isin(selected)].copy()
        norm.metadata["systems_filter"]=sorted(map(int,selected))

    # The existing preparation engine expects its canonical filename.  A
    # temporary normalized directory keeps the public user interface to ONE CSV.
    with tempfile.TemporaryDirectory(prefix="sri_input_") as td:
        td=Path(td)
        norm.members.to_csv(td/"Snake_Member_Catalogue.csv",index=False)
        nodes,stars,_=load_catalogue(td)
        ng={int(k):g for k,g in nodes.groupby("Snake",sort=True)}
        mg={int(k):g for k,g in stars.groupby("Snake",sort=True)}
        score_rows=[]; review_rows=[]; failures=[]; entity_tables=[]
        system_ids=sorted(ng)
        progress=tqdm(system_ids,total=len(system_ids),desc="Computing SRI",unit="system",dynamic_ncols=True)
        for idx,sid in enumerate(progress,1):
            progress.set_postfix(system=norm.group_map.get(int(sid),str(sid)),failed=len(failures),refresh=False)
            try:
                draws=None
                state=build_state(ng[sid],mg[sid],draws,compute_covariance=COMPUTE_COVARIANCE,
                    velocity_center_method=cfg.velocity_center_method,use_uncertainty=cfg.use_uncertainty,spatial_mem_scale=cfg.spatial_mem_scale)
                row,_=score_state(state,cfg,dataset=input_csv.stem)
                row["system_id"]=norm.group_map.get(int(sid),str(sid))
                entity_table=diagnostic_table(state,'full')
                entity_table['system_id']=row['system_id']
                entity_table['entity']=entity_table.entity.map(lambda x:_replace_node_labels(str(x),norm.node_map))
                entity_tables.append(entity_table)
                row["ineligible_3d"]=_replace_node_labels(row.get("ineligible_3d",""),norm.node_map)

                if RUN_KPD and not args.no_kpd:
                    rr,core_labels=review_state(state,cfg,return_core_labels=True)
                    rr["system_id"]=norm.group_map.get(int(sid),str(sid))
                    core_status='pending' if rr['core_selection_status']=='selected' else rr['core_selection_status']
                    row['KPD_mark']=rr['KPD_mark']
                    row['KPD_partition']=_replace_node_labels(rr['partition'],norm.node_map)
                    core_row=None
                    core_error=''
                    if core_status=='pending':
                        try:
                            # Recompute from the source-level inputs using only the
                            # largest review component (member count breaks entity-count ties).  Smaller components
                            # are excluded entirely: e.g. 3+2+1 -> score only the 3.
                            core_state=build_core_state(
                                state,ng[sid],mg[sid],core_labels,draws=draws,
                                compute_covariance=COMPUTE_COVARIANCE,
                            )
                            core_row,_=score_state(core_state,cfg,dataset=input_csv.stem)
                            entity_table=diagnostic_table(core_state,'core')
                            entity_table['system_id']=row['system_id']
                            entity_table['entity']=entity_table.entity.map(lambda x:_replace_node_labels(str(x),norm.node_map))
                            entity_tables.append(entity_table)
                            core_status='computed'
                        except Exception as core_exc:
                            if cfg.use_uncertainty:
                                raise RuntimeError(f'core_prediction_failed: {core_exc}') from core_exc
                            core_status='failed'
                            core_error=repr(core_exc)
                            tqdm.write(
                                f"[warning] core SRI failed for system {norm.group_map.get(int(sid),sid)}: {core_exc}",
                                file=sys.stderr,
                            )

                    row['core_status']=core_status
                    row['core_entities']=_replace_node_labels(rr.get('core_entities',''),norm.node_map)
                    row['core_error']=core_error
                    core_keys=[
                        'Nentity','M_RV','group','S_mem','S_cen','S_cen_active','V_mem','V_cen',
                        'S','V','R0','D_cross_kms','Cross_quality','SRI','grade',
                        'rv_coverage','ineligible_3d',
                    ]
                    for key in core_keys:
                        value=(core_row.get(key) if core_row is not None else ('' if key in {'group','grade','ineligible_3d'} else float('nan')))
                        if key=='ineligible_3d' and core_row is not None:
                            value=_replace_node_labels(value,norm.node_map)
                        row[f'core_{key}']=value

                    rr['core_status']=core_status
                    rr['core_error']=core_error
                    for key in core_keys:
                        rr[f'core_{key}']=row[f'core_{key}']
                    rr['core_SRI']=core_row.get('SRI',float('nan')) if core_row is not None else float('nan')
                    rr['core_grade']=core_row.get('grade','') if core_row is not None else ''
                    for c in ["partition","spatial_partition","unassessed_entities","core_entities"]:
                        if c in rr: rr[c]=_replace_node_labels(rr[c],norm.node_map)
                    review_rows.append(rr)

                score_rows.append(row)
            except Exception as exc:
                entity_tables=[t for t in entity_tables if not (t['system_id']==norm.group_map.get(int(sid),str(sid))).any()]
                failures.append({"Snake":int(sid),"system_id":norm.group_map.get(int(sid),str(sid)),"status":"failed","use_uncertainty":cfg.use_uncertainty,"error_type":type(exc).__name__,"error":_replace_node_labels(str(exc),norm.node_map)})
                tqdm.write(f"[warning] failed system {norm.group_map.get(int(sid),sid)}: {exc}",file=sys.stderr)

        scores=pd.DataFrame(score_rows)
        if scores.empty:scores=pd.DataFrame(columns=['system_id','Snake','SRI','grade','use_uncertainty'])
        if len(scores):
            front=[c for c in [
                "system_id","Snake","SRI","grade","group","Nentity","M_RV","D_cross_kms","Cross_quality",
                "core_status","core_SRI","core_grade","core_Nentity","core_M_RV","core_group","core_entities"
            ] if c in scores]
            scores=scores[front+[c for c in scores.columns if c not in front]]
        scores=_public_table(scores)
        scores.to_csv(output_dir/"SRI.csv",index=False)
        core_columns=["system_id"]+[c for c in scores if c.startswith("core_")]
        if "core_status" in scores:
            scores.loc[scores.core_status.eq("computed"),core_columns].to_csv(output_dir/"Core_SRI.csv",index=False)
        else:
            pd.DataFrame(columns=core_columns).to_csv(output_dir/"Core_SRI.csv",index=False)
        if RUN_KPD and not args.no_kpd:
            _public_table(pd.DataFrame(review_rows,columns=list(review_rows[0]) if review_rows else ["system_id","KPD_mark","candidate","partition","core_status"])).to_csv(output_dir/"KPD.csv",index=False)
        _public_table(pd.DataFrame(failures,columns=['Snake','system_id','status','use_uncertainty','error_type','error'])).to_csv(output_dir/"failures.csv",index=False)
        if entity_tables:
            _public_table(pd.concat(entity_tables,ignore_index=True)).to_csv(output_dir/'Derived_Entity_Kinematics.csv',index=False)
        else:
            pd.DataFrame(columns=['system_id','entity','scope']).to_csv(output_dir/'Derived_Entity_Kinematics.csv',index=False)

        if SAVE_DERIVED_TABLES and not args.no_derived:
            derived_nodes=nodes.copy()
            derived_nodes.insert(0,"system_id",derived_nodes.Snake.map(norm.group_map).fillna(derived_nodes.Snake.astype(str)))
            # restore original node labels where a namespace was needed
            derived_nodes["id_node"]=derived_nodes.id_node.astype(str).map(lambda x:norm.node_map.get(x,x))
            _public_table(derived_nodes).to_csv(output_dir/"Derived_Node_Catalogue.csv",index=False)

    save_mapping(output_dir/"input_mapping.json",norm)
    run_info={
        "scorer":"SRI",
        "input":str(input_csv.resolve()),"output":str(output_dir.resolve()),
        "group_col":group_col,"node_col":node_col,"source_col":source_col,
        "parent":parent,"bridge_scope_col":bridge_scope_col,
        "normalization":norm.metadata,"config":cfg.to_dict(),
        "systems_scored":int(len(score_rows)),"systems_failed":int(len(failures)),
        "run_kpd":bool(RUN_KPD and not args.no_kpd),
        "compute_covariance":bool(COMPUTE_COVARIANCE),
        "velocity_estimation_unit":"independent_real_source_observations",
        "velocity_center_method":cfg.velocity_center_method,
        "uncertainty_model":"component_systemic_errors",
        "covariance_propagation":"mean_sightline_rotation_of_component_systemic_errors",
        "core_policy":"entity_count_then_member_count; exact_ties_unresolved",
        "elapsed_seconds":time.time()-t0,
    }
    (output_dir/"run_config.json").write_text(json.dumps(run_info,indent=2,ensure_ascii=False,default=str),encoding="utf-8")
    grades=pd.Series([r["grade"] for r in score_rows]).value_counts().to_dict() if score_rows else {}
    candidates=sum(bool(r.get("candidate")) for r in review_rows)
    print(json.dumps({"input":input_csv.name,"output":str(output_dir),"systems":len(score_rows),"failed":len(failures),"grades":grades,"review_candidates":candidates},ensure_ascii=False,indent=2))
    return 0 if not failures else 2

if __name__=="__main__":
    raise SystemExit(main())
