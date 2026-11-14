#!/usr/bin/env python3
"""Generate predictions on the validation split only for supplementary analysis.

Never accesses test data. Intended to supply reproducible validation predictions
for the separate boundary_metrics_from_predictions.py script.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
from src.dataset import PH2Dataset, load_split
from src.model import UNet
from src.utils import get_device, load_config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/learning_rate_tuning/config_lr_0005.yaml')
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--experiment',required=True)
    p.add_argument('--output-root',default='results/predictions/validation_only')
    p.add_argument('--batch-size',type=int,default=8)
    a=p.parse_args()
    cfg=load_config(a.config); device=get_device()
    records=load_split(cfg['data']['split_manifest'],'validation')
    ds=PH2Dataset(records,cfg['project']['image_size'],augment=False)
    dl=DataLoader(ds,batch_size=a.batch_size,shuffle=False,num_workers=0)
    ckpt=torch.load(a.checkpoint,map_location=device,weights_only=False)
    model=UNet(cfg['project']['in_channels'],cfg['project']['num_classes']).to(device)
    model.load_state_dict(ckpt['model_state']); model.eval()
    out=Path(a.output_root)/a.experiment; out.mkdir(parents=True,exist_ok=True)
    threshold=float(cfg.get('evaluation',{}).get('threshold',0.5))
    saved=0
    with torch.inference_mode():
        for images,_,case_ids in dl:
            images=images.to(device)
            probabilities=torch.softmax(model(images),dim=1)[:,1]
            predictions=(probabilities>=threshold).cpu().numpy().astype(np.uint8)*255
            for cid,mask in zip(case_ids,predictions):
                Image.fromarray(mask).save(out/f'{cid}_prediction.png')
                saved += 1
    print(f'Validation-only predictions saved: {saved}')
    print(f'Output: {out.resolve()}')
    print(f'Checkpoint: {a.checkpoint}; epoch={ckpt.get("epoch")}; threshold={threshold}')
    print('No test split or test labels were loaded.')

if __name__=='__main__': main()
