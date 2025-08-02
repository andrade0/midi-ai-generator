#!/bin/bash

echo "🎵 Configuration de l'environnement MIDI AI Factory pour Apple Silicon"

# Créer l'environnement conda
conda env create -f environment.yml

# Activer l'environnement
echo "Pour activer l'environnement :"
echo "conda activate midi-ai"

# Vérifier PyTorch MPS
echo ""
echo "Après activation, vérifiez le support MPS avec :"
echo "python -c 'import torch; print(f\"MPS disponible: {torch.backends.mps.is_available()}\")'\"