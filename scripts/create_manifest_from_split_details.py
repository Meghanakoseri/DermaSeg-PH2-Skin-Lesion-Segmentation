#!/usr/bin/env python3
"""Recreate a machine-local JSON manifest from the committed case/split CSV.

This preserves the exact existing 140/30/30 case allocation instead of drawing
anew. It does not inspect model outputs or calculate test metrics.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--audit',default='results/audit/pairs.csv')
p.add_argument('--details',default='data/splits/ph2_seed42_details.csv')
p.add_argument('--output',default='data/splits/ph2_seed42.json')
a=p.parse_args()
aud=pd.read_csv(a.audit); det=pd.read_csv(a.details)
for required,frame,name in [({'case_id','image','mask'},aud,'audit'),({'case_id','split'},det,'split-details')]:
    missing=required-set(frame.columns)
    if missing: raise SystemExit(f'{name} missing columns: {sorted(missing)}')
aud['case_id']=aud.case_id.astype(str); det['case_id']=det.case_id.astype(str)
if aud.case_id.duplicated().any() or det.case_id.duplicated().any():
    raise SystemExit('Duplicate case IDs in audit or split details')
if set(aud.case_id)!=set(det.case_id):
    print('In audit but not split details:',sorted(set(aud.case_id)-set(det.case_id))[:10])
    print('In split details but not audit:',sorted(set(det.case_id)-set(aud.case_id))[:10])
    raise SystemExit('Case IDs do not match exactly; do not regenerate manifest until resolved')
merged=det.merge(aud[['case_id','image','mask']],on='case_id',how='inner',validate='one_to_one')
normalize={'train':'train','training':'train','validation':'validation','val':'validation','test':'test'}
merged['split_norm']=merged['split'].astype(str).str.strip().str.lower().map(normalize)
if merged.split_norm.isna().any(): raise SystemExit('Unrecognized split labels in details CSV')
manifest={'seed':42}
for name in ['train','validation','test']:
    part=merged.loc[merged.split_norm.eq(name)].sort_values('case_id')
    manifest[name]=[{'case_id':r.case_id,'image':str(Path(r.image).resolve()),'mask':str(Path(r.mask).resolve())} for r in part.itertuples()]
counts={k:len(manifest[k]) for k in ['train','validation','test']}
if counts!={'train':140,'validation':30,'test':30}: raise SystemExit(f'Unexpected counts: {counts}')
allids=[x['case_id'] for k in ['train','validation','test'] for x in manifest[k]]
if len(allids)!=len(set(allids)): raise SystemExit('Split overlap detected')
out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('Rebuilt manifest using the exact committed case allocation:',counts)
print('Saved',out.resolve())
