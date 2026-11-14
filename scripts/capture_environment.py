#!/usr/bin/env python3
"""Capture exact versions/hardware from the currently active Python environment."""
from __future__ import annotations
import argparse, importlib.metadata as md, json, os, platform, subprocess, sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--name',default='local_demo',help='e.g. local_demo, colab_training_runtime, colab_recreated')
p.add_argument('--output-dir',default='results/audit')
a=p.parse_args()
packages=["torch","torchvision","numpy","pandas","Pillow","PyYAML","scikit-learn","matplotlib","tqdm","streamlit"]
versions={}
for name in packages:
    try: versions[name]=md.version(name)
    except md.PackageNotFoundError: versions[name]=None
hardware={"cuda_available":False,"cuda_device":None,"cuda_version":None}
try:
    import torch
    hardware['cuda_available']=torch.cuda.is_available()
    hardware['cuda_version']=torch.version.cuda
    if torch.cuda.is_available(): hardware['cuda_device']=torch.cuda.get_device_name(0)
    hardware['torch_num_threads']=torch.get_num_threads()
except Exception as exc: hardware['torch_probe_error']=str(exc)
pip=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=False)
result={"environment_name":a.name,"python":sys.version,"python_version":platform.python_version(),
        "platform":platform.platform(),"cpu_model":platform.processor(),"logical_cpu_count":os.cpu_count(),
        "packages":versions,"hardware":hardware,"pip_freeze_exit_code":pip.returncode}
out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
(out/f'environment_{a.name}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
(out/f'requirements_lock_{a.name}.txt').write_text(pip.stdout if pip.returncode==0 else '# pip freeze failed\n',encoding='utf-8')
print(json.dumps(result,indent=2)); print(f"Wrote {out/f'environment_{a.name}.json'} and {out/f'requirements_lock_{a.name}.txt'}")
