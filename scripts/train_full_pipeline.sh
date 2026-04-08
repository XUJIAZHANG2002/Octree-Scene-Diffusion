#!/bin/bash

# Exit on error
set -e

echo "Starting Phase 1: Structure VAE"
python train_structure_vae.py

echo "Starting Phase 2: Structure Diffusion"
python train_structure_diffusion.py

echo "Starting Phase 3: Semantic VAE"
python train_vae.py

echo "Starting Phase 4: Semantic Diffusion"
python train_sem_diffusion.py

echo "Pipeline Training Complete!"