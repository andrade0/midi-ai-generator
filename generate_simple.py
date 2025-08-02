#!/usr/bin/env python3
"""
Générateur simplifié de MIDI
"""

import os
import torch
import numpy as np
from datetime import datetime
import pretty_midi
from model import MusicTransformer

class SimpleMidiGenerator:
    def __init__(self, checkpoint_path):
        print(f"📦 Chargement du modèle depuis {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Setup device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("🚀 Utilisation du GPU Apple Silicon (MPS)")
        else:
            self.device = torch.device("cpu")
            print("💻 Utilisation du CPU")
            
        # Recréer le modèle
        self.model = MusicTransformer(**checkpoint['model_config'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Créer le dossier de sortie
        os.makedirs('./output_midi', exist_ok=True)
        
    def generate_simple_midi(self, length=256, temperature=1.0):
        """Génère un MIDI simple à partir du modèle"""
        print(f"\n🎼 Génération d'un pattern...")
        
        # Générer des tokens de départ aléatoires
        start_tokens = torch.randint(1, 100, (1, 8), device=self.device)
        
        # Générer la séquence
        with torch.no_grad():
            generated = self.model.generate(
                start_tokens,
                max_length=length,
                temperature=temperature,
                top_k=50,
                top_p=0.95
            )
        
        # Convertir en MIDI simple
        tokens = generated[0].cpu().numpy()
        midi = self._create_simple_midi(tokens)
        
        # Sauvegarder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generated_simple_{timestamp}.mid"
        filepath = os.path.join('./output_midi', filename)
        
        midi.write(filepath)
        print(f"✅ Pattern généré: {filepath}")
        
        return filepath
    
    def _create_simple_midi(self, tokens):
        """Crée un fichier MIDI simple à partir des tokens"""
        midi = pretty_midi.PrettyMIDI(initial_tempo=120)
        instrument = pretty_midi.Instrument(program=0)  # Piano
        
        current_time = 0.0
        note_duration = 0.25  # Durée par défaut (noire)
        
        for i, token in enumerate(tokens):
            # Mapper le token à une note MIDI (simplification)
            pitch = 48 + (token % 24)  # Notes entre C3 et C5
            velocity = 64 + (token % 64)  # Vélocité variable
            
            # Créer la note
            note = pretty_midi.Note(
                velocity=int(velocity),
                pitch=int(pitch),
                start=current_time,
                end=current_time + note_duration
            )
            instrument.notes.append(note)
            
            # Avancer dans le temps
            if token % 4 == 0:  # Variation rythmique simple
                current_time += note_duration * 2
            else:
                current_time += note_duration
                
            # Limiter à 16 secondes
            if current_time >= 16:
                break
        
        midi.instruments.append(instrument)
        return midi
    
    def batch_generate(self, n_clips=5):
        """Génère plusieurs clips"""
        results = []
        
        for i in range(n_clips):
            print(f"\n📍 Génération {i+1}/{n_clips}")
            temperature = np.random.uniform(0.7, 1.2)
            filepath = self.generate_simple_midi(temperature=temperature)
            results.append(filepath)
            
        print(f"\n🎹 {n_clips} patterns générés dans ./output_midi/")
        return results

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Générateur MIDI simplifié")
    parser.add_argument('--checkpoint', type=str, 
                       default='./checkpoints/20250802_182855/best_model.pt',
                       help='Chemin vers le checkpoint')
    parser.add_argument('--n_clips', type=int, default=5,
                       help='Nombre de clips à générer')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        # Trouver le dernier checkpoint
        checkpoint_dirs = sorted([d for d in os.listdir('./checkpoints') 
                                if os.path.isdir(f'./checkpoints/{d}')])
        if checkpoint_dirs:
            args.checkpoint = f'./checkpoints/{checkpoint_dirs[-1]}/best_model.pt'
        else:
            print("❌ Aucun checkpoint trouvé")
            return
    
    generator = SimpleMidiGenerator(args.checkpoint)
    generator.batch_generate(n_clips=args.n_clips)

if __name__ == "__main__":
    main()