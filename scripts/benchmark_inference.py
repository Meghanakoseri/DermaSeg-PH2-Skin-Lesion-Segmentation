#!/usr/bin/env python3
"""Benchmark single-image inference latency with synthetic tensors only.

No PH2 image or test label is read. Reports the runtime of the local device and
checkpoint, and is intended as an engineering measurement, not clinical timing.
"""
from __future__ import annotations
import argparse, json, platform, statistics, time, math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.model import UNet
from src.utils import get_device, load_config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",default="configs/learning_rate_tuning/config_lr_0005.yaml")
    p.add_argument("--checkpoint",default="models/checkpoints/tune_focal_tversky_lr_0005_best.pt")
    p.add_argument("--warmup",type=int,default=10); p.add_argument("--iterations",type=int,default=100)
    p.add_argument("--threads",type=int,default=2); p.add_argument("--output",default="results/audit/inference_benchmark.json")
    args=p.parse_args()
    torch.set_num_threads(args.threads)
    cfg=load_config(args.config); device=get_device()
    ckpt=torch.load(args.checkpoint,map_location=device,weights_only=False)
    model=UNet(cfg["project"]["in_channels"],cfg["project"]["num_classes"]).to(device)
    model.load_state_dict(ckpt["model_state"]); model.eval()
    x=torch.rand(1,cfg["project"]["in_channels"],cfg["project"]["image_size"],cfg["project"]["image_size"],device=device)
    def sync():
        if device.type=="cuda": torch.cuda.synchronize(device)
    with torch.inference_mode():
        for _ in range(args.warmup): model(x)
        sync(); times=[]
        for _ in range(args.iterations):
            sync(); t=time.perf_counter(); model(x); sync(); times.append((time.perf_counter()-t)*1000)
    times_sorted=sorted(times)
    result={"device":str(device),"platform":platform.platform(),"python":platform.python_version(),
      "torch":torch.__version__,"checkpoint":str(Path(args.checkpoint)),"input_shape":list(x.shape),
      "warmup_iterations":args.warmup,"measured_iterations":args.iterations,"threads":args.threads,
      "latency_ms_mean":statistics.mean(times),"latency_ms_median":statistics.median(times),
      "latency_ms_p95":times_sorted[max(0, math.ceil(0.95*len(times_sorted))-1)],
      "images_per_second_median":1000/statistics.median(times)}
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
if __name__=="__main__": main()
