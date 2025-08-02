#!/usr/bin/env python3
"""
Script alternatif pour setup sans conda
Utilise pip dans un environnement virtuel Python
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Vérifie la version Python"""
    version = sys.version_info
    print(f"🐍 Python {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("❌ Python 3.10+ requis")
        return False
    return True

def create_venv():
    """Crée un environnement virtuel Python"""
    print("\n📦 Création de l'environnement virtuel...")
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✅ Environnement virtuel créé")
        return True
    except:
        print("❌ Erreur lors de la création du venv")
        return False

def get_pip_cmd():
    """Retourne la commande pip selon l'OS"""
    if platform.system() == "Windows":
        return os.path.join("venv", "Scripts", "pip")
    else:
        return os.path.join("venv", "bin", "pip")

def install_packages():
    """Installe les packages nécessaires"""
    pip_cmd = get_pip_cmd()
    
    print("\n📥 Installation des dépendances...")
    
    # Packages essentiels
    packages = [
        "torch==2.1.0",
        "numpy==1.24.3",
        "scipy",
        "pandas",
        "matplotlib",
        "pretty_midi==0.2.10",
        "miditok==2.1.8",
        "music21==9.1.0",
        "tqdm",
        "tensorboard",
        "einops"
    ]
    
    for package in packages:
        print(f"  Installing {package}...")
        try:
            subprocess.run([pip_cmd, "install", package], check=True, capture_output=True)
        except:
            print(f"  ⚠️  Erreur avec {package}, tentative sans version...")
            subprocess.run([pip_cmd, "install", package.split("==")[0]], check=True)
    
    print("✅ Toutes les dépendances installées")

def check_mps():
    """Vérifie le support MPS"""
    print("\n🔍 Vérification du support Apple Silicon...")
    activate_cmd = "source venv/bin/activate" if platform.system() != "Windows" else "venv\\Scripts\\activate"
    
    test_code = """
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"MPS disponible: {torch.backends.mps.is_available()}")
if torch.backends.mps.is_available():
    print("✅ GPU Apple Silicon prêt!")
else:
    print("⚠️  MPS non disponible, utilisation CPU")
"""
    
    with open("test_mps.py", "w") as f:
        f.write(test_code)
    
    if platform.system() == "Windows":
        python_cmd = os.path.join("venv", "Scripts", "python")
    else:
        python_cmd = os.path.join("venv", "bin", "python")
    
    subprocess.run([python_cmd, "test_mps.py"])
    os.remove("test_mps.py")

def main():
    print("🎹 MIDI AI Factory - Setup alternatif (sans conda)")
    print("=" * 50)
    
    if not check_python_version():
        return
    
    if not create_venv():
        return
    
    install_packages()
    check_mps()
    
    print("\n✨ Installation terminée!")
    print("\n📌 Pour activer l'environnement:")
    if platform.system() == "Windows":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    
    print("\n🚀 Puis lancez:")
    print("   python main.py all --epochs 50")

if __name__ == "__main__":
    main()