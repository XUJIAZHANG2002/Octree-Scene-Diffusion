# Octree Latent Diffusion for Semantic 3D Scene Generation and Completion

Official implementation of:

**Octree Latent Diffusion for Semantic 3D Scene Generation and Completion**  
Accepted at **ICRA 2026**.

---

## 🚧 Status

Code release is currently under construction.

Most of the code will be completed & uploaded before April 2026.

The full training and inference pipeline will be released soon.

---

## Overview

Octree Scene Diffusion is a structured generative framework for semantic 3D scene generation and completion.  
It leverages hierarchical octree representations to enable scalable and memory-efficient diffusion modeling of large 3D environments.

More details will be added with the full code release.

---
## Dataset
(TODO)
## Installation 
The enviroment builds upon [OctFusion](https://github.com/octree-nn/octfusion).

1. Clone this repository
```bash
git clone https://github.com/XUJIAZHANG2002/Octree-Scene-Diffusion.git
cd Octree-Scene-Diffusion
```
2. Create a `Conda` environment.
```bash
conda create -n octfusion python=3.11 -y && conda activate octfusion
```

3. Install PyTorch with Conda
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

4. Install other requirements.
```bash
pip3 install -r requirements.txt 
```
---

## Training
🚀 Training
The pipeline consists of four models trained in two stages (Structure and Semantics). You can run the full pipeline using the provided script:
Bash

```bash 
scripts/train_full_pipeline.sh
```
Alternatively, you can train each component individually:

Stage A: Structural Prior

Structure VAE: Learns the global occupancy latent space.
```python
python train_structure_vae.py
```
Structure Diffusion: Generates structural latents.
```python
python train_structure_diffusion.py
```
Stage B: Semantic Refinement

Semantic VAE: Learns the octree-based semantic features.
```python
python train_vae.py
```
Semantic Diffusion: Generates semantic labels conditioned on the structure.
```python
python train_sem_diffusion.py
```
## Citation

If you find this work useful, please consider citing:

```bibtex
@misc{zhang2025octreelatentdiffusionsemantic,
      title={Octree Latent Diffusion for Semantic 3D Scene Generation and Completion}, 
      author={Xujia Zhang and Brendan Crowe and Christoffer Heckman},
      year={2025},
      eprint={2509.16483},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.16483}, 
}
```
## Acknowledgements

This codebase is heavily inspired by and builds upon:
- [OctFusion](https://arxiv.org/abs/2408.14732)
- [SemCity](https://arxiv.org/abs/2403.07773)