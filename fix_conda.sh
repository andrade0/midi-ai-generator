#!/bin/bash

echo "🔧 Diagnostic et correction de Conda"
echo "===================================="
echo ""

# 1. Vérifier l'installation de conda
echo "1. Recherche de conda..."
if [ -f ~/miniconda3/bin/conda ]; then
    echo "✅ Conda trouvé dans ~/miniconda3"
    CONDA_PATH=~/miniconda3/bin/conda
elif [ -f ~/anaconda3/bin/conda ]; then
    echo "✅ Conda trouvé dans ~/anaconda3"
    CONDA_PATH=~/anaconda3/bin/conda
elif [ -f /opt/homebrew/anaconda3/bin/conda ]; then
    echo "✅ Conda trouvé dans /opt/homebrew/anaconda3"
    CONDA_PATH=/opt/homebrew/anaconda3/bin/conda
elif [ -f /usr/local/anaconda3/bin/conda ]; then
    echo "✅ Conda trouvé dans /usr/local/anaconda3"
    CONDA_PATH=/usr/local/anaconda3/bin/conda
else
    echo "❌ Conda non trouvé dans les chemins standards"
    echo ""
    echo "Installation recommandée:"
    echo "  brew install --cask miniconda"
    echo "ou téléchargez depuis: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo ""
echo "2. Utilisation du chemin complet de conda:"
echo "   $CONDA_PATH"
echo ""

# 3. Créer l'environnement avec le chemin complet
echo "3. Création de l'environnement midi-ai..."
echo "   Commande: $CONDA_PATH env create -f environment.yml"
echo ""
echo "Appuyez sur Entrée pour continuer ou Ctrl+C pour annuler..."
read

$CONDA_PATH env create -f environment.yml

echo ""
echo "✅ Environnement créé!"
echo ""
echo "4. Pour activer l'environnement, utilisez:"
echo "   $CONDA_PATH activate midi-ai"
echo "   ou"
echo "   source $($CONDA_PATH info --base)/etc/profile.d/conda.sh && conda activate midi-ai"
echo ""