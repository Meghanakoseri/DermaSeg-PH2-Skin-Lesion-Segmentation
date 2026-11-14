# DermaSeg environment provenance

## Colab training runtime capture

Captured on 2026-11-14 from the original connected Colab runtime after the training runs.

- Python: 3.13.16
- Platform: Linux x86_64, glibc 2.39
- GPU: NVIDIA Tesla T4
- GPU memory reported: 15.64 GB decimal (14.56 GiB)
- PyTorch: 2.14.1+cu130
- CUDA: 13.0
- cuDNN: 9.28
- torchvision: 0.29.1+cu130
- NumPy: 2.4.6
- pandas: 2.2.3
- Pillow: 11.3.0
- PyYAML: 6.0.3
- scikit-learn: 1.6.1
- matplotlib: 3.10.0
- tqdm: 4.67.3
- Streamlit: not installed in the training runtime

The visible active notebook input history was searched for pip/conda/mamba/apt package installation, removal or upgrade commands. No matching commands were found. This is supporting evidence that the captured package state may represent the training runtime, but it is not a formal proof against package changes made through an unrecorded external process. For that reason, describe these files as a post-training capture from the original connected runtime, not as a guaranteed cryptographic snapshot of every historical run.

The JSON capture and pip-freeze lock were generated in the original Drive project during the previous environment-capture step. They were not present in the starter ZIP used to assemble this local release, so they are not bundled here. Copy the original files from the Drive project if the TA requests the raw provenance artifacts.

## Local demo runtime capture

Captured from the Windows 11 laptop virtual environment:

- Python: 3.12.4
- PyTorch: 2.15.0+cpu
- torchvision: 0.29.2+cpu
- NumPy: 2.5.4
- pandas: 3.1.1
- Pillow: 13.0.0
- PyYAML: 6.0.3
- scikit-learn: 1.9.1
- matplotlib: 3.11.2
- tqdm: 4.70.1
- Streamlit: 1.67.0
- CUDA: unavailable
- CPU: Intel64 Family 6 Model 154; 8 logical CPUs reported

The JSON capture and pip-freeze lock were generated in the earlier Windows project workspace. They were not present in the starter ZIP used to assemble this local release, so they are not bundled here. Copy the original files from that workspace if raw environment records are required.

Keep the Colab and laptop lockfiles separate. Do not combine them into one cross-platform requirements file.
