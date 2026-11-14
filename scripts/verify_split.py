#!/usr/bin/env python3
"""Verify split counts, case isolation, path availability and optional strata.

This is a metadata-only integrity check. It does not load images into a model,
calculate predictions or access test labels for performance evaluation.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--manifest',default='data/splits/ph2_seed42.json')
p.add_argument('--expected',default='140,30,30')
p.add_argument('--details',default='data/splits/ph2_seed42_details.csv')
p.add_argument('--check-paths',action='store_true')
a=p.parse_args()
data=json.loads(Path(a.manifest).read_text(encoding='utf-8'))
expected=[int(x) for x in a.expected.split(',')]
splits=['train','validation','test']
if len(expected)!=3: raise SystemExit('--expected must contain three comma-separated counts')
ids={s:[str(x['case_id']) for x in data[s]] for s in splits}
all_ids=[cid for s in splits for cid in ids[s]]
print('Split seed:',data.get('seed','not recorded'))
for s,n in zip(splits,expected):
    print(f'{s:12s}: {len(ids[s])} (expected {n})')
    if len(ids[s])!=n: raise SystemExit(f'FAIL: {s} count mismatch')
if len(all_ids)!=len(set(all_ids)): raise SystemExit('FAIL: duplicate case IDs across splits')
if any(not x.get('image') or not x.get('mask') for s in splits for x in data[s]):
    raise SystemExit('FAIL: missing image or mask path')
if a.check_paths:
    missing=[]
    for split in splits:
        for row in data[split]:
            for key in ('image','mask'):
                if not Path(row[key]).is_file(): missing.append((row['case_id'],key,row[key]))
    if missing: raise SystemExit(f'FAIL: {len(missing)} missing data paths; first: {missing[:5]}')
    print('Image/mask paths: all exist.')
if Path(a.details).is_file():
    details=pd.read_csv(a.details)
    if {'case_id','split'}.issubset(details.columns):
        if details.case_id.astype(str).duplicated().any(): raise SystemExit('FAIL: duplicate IDs in split details')
        if set(details.case_id.astype(str)) != set(all_ids): raise SystemExit('FAIL: details IDs differ from manifest IDs')
        bin_col=next((c for c in ('area_bin','lesion_area_bin','stratum','bin') if c in details.columns),None)
        if bin_col:
            ctab=details.groupby([bin_col,'split']).size().unstack(fill_value=0)
            print('Area-bin × split counts:')
            print(ctab.to_string())
            required={'train':28,'validation':6,'test':6}
            for col,val in required.items():
                if col not in ctab.columns or not (ctab[col]==val).all():
                    raise SystemExit(f'FAIL: expected {val} cases per area bin in {col}')
            print('Area-stratum allocation: 28/6/6 in every bin.')
print(f'PASS: {len(all_ids)} unique case IDs; no overlap; expected counts verified.')
