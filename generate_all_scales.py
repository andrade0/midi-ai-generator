#!/usr/bin/env python3
"""
Script pour générer des fichiers MIDI dans toutes les gammes disponibles
"""

import os
import subprocess
import sys
from datetime import datetime

def generate_scale_batch(scale, n_files=50):
    """Génère un batch de fichiers pour une gamme donnée"""
    print(f"\n{'='*60}")
    print(f"🎵 Génération pour la gamme : {scale}")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, 
        "generate_harmonic.py",
        "--scale", scale,
        "--n_melodies", str(n_files),
        "--n_chords", str(n_files),
        "--n_basslines", str(n_files)
    ]
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors de la génération pour {scale}: {e}")
        return False

def main():
    print("🎹 MIDI AI Factory - Génération Multi-Gammes")
    print("=" * 60)
    print("Génération de 150 fichiers (50 mélodies, 50 accords, 50 basses)")
    print("pour chaque gamme disponible")
    print("=" * 60)
    
    # Liste des gammes à générer
    scales = [
        'C_major',    # Do majeur
        'A_minor',    # La mineur (relative de Do majeur)
        'G_major',    # Sol majeur (1 dièse)
        'E_minor',    # Mi mineur (relative de Sol majeur)
        'D_major',    # Ré majeur (2 dièses)
        'B_minor',    # Si mineur (relative de Ré majeur)
        'F_major',    # Fa majeur (1 bémol)
        'D_minor',    # Ré mineur (relative de Fa majeur)
    ]
    
    start_time = datetime.now()
    success_count = 0
    
    # Créer le dossier principal pour toutes les gammes
    os.makedirs('./output_midi/all_scales', exist_ok=True)
    
    # Générer pour chaque gamme
    for i, scale in enumerate(scales, 1):
        print(f"\n📊 Progression : {i}/{len(scales)} gammes")
        
        # Créer les dossiers spécifiques pour cette gamme
        scale_dir = f'./output_midi/all_scales/{scale}'
        os.makedirs(f'{scale_dir}/melodies', exist_ok=True)
        os.makedirs(f'{scale_dir}/chords', exist_ok=True)
        os.makedirs(f'{scale_dir}/basslines', exist_ok=True)
        
        if generate_scale_batch(scale, n_files=50):
            success_count += 1
            
            # Déplacer les fichiers générés dans le bon dossier
            try:
                # Déplacer les mélodies
                os.system(f'mv ./output_midi/harmonic/melodies/*{scale}* {scale_dir}/melodies/ 2>/dev/null')
                # Déplacer les accords
                os.system(f'mv ./output_midi/harmonic/chords/*{scale}* {scale_dir}/chords/ 2>/dev/null')
                # Déplacer les basslines
                os.system(f'mv ./output_midi/harmonic/basslines/*{scale}* {scale_dir}/basslines/ 2>/dev/null')
            except:
                pass
    
    # Résumé final
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*60)
    print("✅ GÉNÉRATION TERMINÉE !")
    print("="*60)
    print(f"📊 Statistiques finales:")
    print(f"   - Gammes traitées : {success_count}/{len(scales)}")
    print(f"   - Fichiers par gamme : 150 (50 mélodies + 50 accords + 50 basses)")
    print(f"   - Total fichiers générés : {success_count * 150}")
    print(f"   - Durée totale : {duration}")
    print(f"\n📁 Fichiers organisés dans : ./output_midi/all_scales/")
    print(f"   Chaque gamme a son propre dossier avec:")
    print(f"   - /melodies/")
    print(f"   - /chords/")
    print(f"   - /basslines/")
    
    # Afficher la structure des dossiers
    print("\n📂 Structure créée:")
    for scale in scales:
        if os.path.exists(f'./output_midi/all_scales/{scale}'):
            print(f"   ├── {scale}/")
            print(f"   │   ├── melodies/ (50 fichiers)")
            print(f"   │   ├── chords/ (50 fichiers)")
            print(f"   │   └── basslines/ (50 fichiers)")

if __name__ == "__main__":
    main()