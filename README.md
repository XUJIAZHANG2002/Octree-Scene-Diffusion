# Octree Diffusion for Semantic Scene Generation and Completion

Official implementation of:

**Octree Diffusion for Semantic Scene Generation and Completion**  
Accepted at **ICRA 2026**.
[arxiv link](https://arxiv.org/abs/2509.16483)
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

Download the Velodyne point clouds (80 GB) from the official KITTI website:
* https://www.semantic-kitti.org/dataset.html

For indoor scene generation, we tested our model on Replica.

* https://github.com/facebookresearch/Replica-Dataset
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

The pipeline consists of four models trained in two stages (Structure and Semantics). 

 In this project, we used patch_size = 2 for indoor scene and patch_size = 4 for ourdoor scenes. Please adjust as needed in configs/sem_vae_config.yaml.

**Stage A: Structural Generation**

1. **Structure VAE**: Learns the occupancy latent space.
   ```bash
   python main.py --stage str_vae
2. **Structure Diffusion**: Generates structural latents.
      ```python
      python main.py --stage str_diff
      ```
**Stage B: Semantic Generation**

1. **Semantic VAE**: Learns the octree-based semantic features.
      ```python
      python main.py --stage sem_vae
      ```
2. **Semantic Diffusion**: Generates semantic labels conditioned on the structure.
      ```python
      python main.py --stage sem_diff
      ```

## Inference

**Generation**

**Semantic Scene Compeletion**

**Scene Extension**


## Citation

If you find this work useful, please consider citing:

```bibtex
@misc{zhang2026octreediffusionsemanticscene,
      title={Octree Diffusion for Semantic Scene Generation and Completion}, 
      author={Xujia Zhang and Brendan Crowe and Christoffer Heckman},
      year={2026},
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
