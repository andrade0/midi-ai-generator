#!/usr/bin/env python3
"""
MIDI AI Factory - Script principal
Générateur de patterns MIDI avec IA sur Apple Silicon
"""

import os
import sys
import argparse
from datetime import datetime

def print_banner():
    print("""
    ╔═══════════════════════════════════════╗
    ║      🎹 MIDI AI Factory 🎹           ║
    ║   Générateur IA pour Apple Silicon    ║
    ╚═══════════════════════════════════════╝
    """)

def check_environment():
    """Vérifie que l'environnement est correctement configuré"""
    try:
        import torch
        import miditok
        import pretty_midi
        
        if torch.backends.mps.is_available():
            print("✅ PyTorch MPS (Apple Silicon) disponible")
        else:
            print("⚠️  MPS non disponible, utilisation du CPU")
            
        return True
    except ImportError as e:
        print(f"❌ Dépendance manquante: {e}")
        print("\nActivez l'environnement conda:")
        print("  conda activate midi-ai")
        return False

def main():
    print_banner()
    
    parser = argparse.ArgumentParser(description="MIDI AI Factory - Pipeline complet")
    parser.add_argument('command', choices=['analyze', 'tokenize', 'train', 'generate', 'all'],
                       help='Commande à exécuter')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Nombre d\'epochs pour l\'entraînement')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Taille de batch')
    parser.add_argument('--style', type=str, default='random',
                       help='Style de génération (chord/melody/bass/random)')
    parser.add_argument('--n-clips', type=int, default=5,
                       help='Nombre de clips à générer')
    
    args = parser.parse_args()
    
    if not check_environment():
        sys.exit(1)
    
    if args.command == 'analyze' or args.command == 'all':
        print("\n📊 Analyse des fichiers MIDI...")
        from midi_analyzer import MidiAnalyzer
        analyzer = MidiAnalyzer()
        analyzer.analyze_all()
        
    if args.command == 'tokenize' or args.command == 'all':
        print("\n🔤 Tokenization des fichiers MIDI...")
        from midi_tokenizer import MidiTokenizer
        tokenizer = MidiTokenizer()
        tokenizer.process_dataset()
        
    if args.command == 'train' or args.command == 'all':
        print("\n🎯 Entraînement du modèle...")
        # Créer un fichier de config temporaire
        config = f"""
import train
train.main()
"""
        exec(config)
        
    if args.command == 'generate':
        print("\n🎼 Génération de patterns MIDI...")
        # Trouver le dernier checkpoint
        checkpoint_dirs = [d for d in os.listdir('./checkpoints') if os.path.isdir(f'./checkpoints/{d}')]
        if not checkpoint_dirs:
            print("❌ Aucun modèle entraîné trouvé. Lancez d'abord l'entraînement.")
            sys.exit(1)
            
        latest_dir = sorted(checkpoint_dirs)[-1]
        checkpoint_path = f'./checkpoints/{latest_dir}/best_model.pt'
        
        if not os.path.exists(checkpoint_path):
            # Essayer avec un checkpoint d'epoch
            checkpoints = [f for f in os.listdir(f'./checkpoints/{latest_dir}') if f.endswith('.pt')]
            if checkpoints:
                checkpoint_path = f'./checkpoints/{latest_dir}/{checkpoints[0]}'
            else:
                print("❌ Aucun checkpoint trouvé")
                sys.exit(1)
                
        from generate import MidiGenerator
        generator = MidiGenerator(checkpoint_path)
        
        if args.n_clips == 1:
            generator.generate_midi_clip(style=args.style)
        else:
            generator.batch_generate(n_clips=args.n_clips)
            
    print("\n✨ Terminé!")

if __name__ == "__main__":
    main()