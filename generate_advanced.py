#!/usr/bin/env python3
"""
Générateur avancé de MIDI - Mélodies, Accords et Basslines de 60 secondes
"""

import os
import torch
import numpy as np
from datetime import datetime
import pretty_midi
from model import MusicTransformer
from tqdm import tqdm

class AdvancedMidiGenerator:
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
        
        # Créer les dossiers de sortie
        os.makedirs('./output_midi/melodies', exist_ok=True)
        os.makedirs('./output_midi/chords', exist_ok=True)
        os.makedirs('./output_midi/basslines', exist_ok=True)
        
    def generate_melody(self, duration=60, tempo=120, temperature=0.9):
        """Génère une mélodie de 60 secondes"""
        length = int(duration * 4)  # Approximation pour 60s
        
        # Tokens de départ pour mélodie (notes plus hautes)
        start_tokens = torch.randint(60, 84, (1, 8), device=self.device)  # C4 à C6
        
        # Générer la séquence
        with torch.no_grad():
            generated = self.model.generate(
                start_tokens,
                max_length=length,
                temperature=temperature,
                top_k=40,
                top_p=0.9
            )
        
        # Convertir en MIDI
        tokens = generated[0].cpu().numpy()
        midi = self._create_melody_midi(tokens, tempo, duration)
        
        return midi
    
    def generate_chords(self, duration=60, tempo=120, temperature=0.8):
        """Génère une progression d'accords de 60 secondes"""
        length = int(duration * 2)  # Moins de tokens pour les accords
        
        # Tokens de départ pour accords (notes moyennes)
        start_tokens = torch.randint(48, 72, (1, 8), device=self.device)  # C3 à C5
        
        # Générer la séquence
        with torch.no_grad():
            generated = self.model.generate(
                start_tokens,
                max_length=length,
                temperature=temperature,
                top_k=30,
                top_p=0.85
            )
        
        # Convertir en MIDI
        tokens = generated[0].cpu().numpy()
        midi = self._create_chords_midi(tokens, tempo, duration)
        
        return midi
    
    def generate_bassline(self, duration=60, tempo=120, temperature=0.7):
        """Génère une ligne de basse de 60 secondes"""
        length = int(duration * 3)
        
        # Tokens de départ pour basse (notes graves)
        start_tokens = torch.randint(36, 60, (1, 8), device=self.device)  # C2 à C4
        
        # Générer la séquence
        with torch.no_grad():
            generated = self.model.generate(
                start_tokens,
                max_length=length,
                temperature=temperature,
                top_k=25,
                top_p=0.8
            )
        
        # Convertir en MIDI
        tokens = generated[0].cpu().numpy()
        midi = self._create_bassline_midi(tokens, tempo, duration)
        
        return midi
    
    def _create_melody_midi(self, tokens, tempo, duration):
        """Crée un fichier MIDI pour une mélodie"""
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        instrument = pretty_midi.Instrument(program=0)  # Piano
        
        current_time = 0.0
        beat_duration = 60.0 / tempo  # Durée d'un temps en secondes
        
        for i, token in enumerate(tokens):
            if current_time >= duration:
                break
                
            # Mapper le token à une note (mélodie dans les aigus)
            pitch = 60 + (token % 24)  # C4 à C6
            velocity = 70 + (token % 40)
            
            # Durées variées pour la mélodie
            durations = [0.25, 0.5, 0.75, 1.0]  # En beats
            note_duration = durations[token % len(durations)] * beat_duration
            
            # Créer la note
            note = pretty_midi.Note(
                velocity=int(velocity),
                pitch=int(pitch),
                start=current_time,
                end=current_time + note_duration
            )
            instrument.notes.append(note)
            
            # Avancer dans le temps (avec quelques silences)
            if token % 10 < 2:  # 20% de chance de silence
                current_time += note_duration * 0.5
            else:
                current_time += note_duration
        
        midi.instruments.append(instrument)
        return midi
    
    def _create_chords_midi(self, tokens, tempo, duration):
        """Crée un fichier MIDI pour des accords"""
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        instrument = pretty_midi.Instrument(program=0)  # Piano
        
        current_time = 0.0
        beat_duration = 60.0 / tempo
        
        # Progressions d'accords communes
        chord_types = [
            [0, 4, 7],      # Majeur
            [0, 3, 7],      # Mineur
            [0, 4, 7, 11],  # Maj7
            [0, 3, 7, 10],  # m7
            [0, 5, 7],      # Sus4
            [0, 4, 7, 10],  # 7
        ]
        
        for i, token in enumerate(tokens):
            if current_time >= duration:
                break
                
            # Note fondamentale
            root = 48 + (token % 12)  # Une octave à partir de C3
            chord_type = chord_types[token % len(chord_types)]
            velocity = 60 + (token % 30)
            
            # Durée d'accord (généralement plus longue)
            chord_duration = (1 + (token % 4)) * beat_duration  # 1 à 4 beats
            
            # Créer toutes les notes de l'accord
            for interval in chord_type:
                note = pretty_midi.Note(
                    velocity=int(velocity),
                    pitch=int(root + interval),
                    start=current_time,
                    end=current_time + chord_duration * 0.95  # Légère séparation
                )
                instrument.notes.append(note)
            
            current_time += chord_duration
        
        midi.instruments.append(instrument)
        return midi
    
    def _create_bassline_midi(self, tokens, tempo, duration):
        """Crée un fichier MIDI pour une ligne de basse"""
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        instrument = pretty_midi.Instrument(program=33)  # Basse électrique
        
        current_time = 0.0
        beat_duration = 60.0 / tempo
        
        for i, token in enumerate(tokens):
            if current_time >= duration:
                break
                
            # Notes de basse (graves)
            pitch = 36 + (token % 24)  # C2 à C4
            velocity = 80 + (token % 40)
            
            # Patterns rythmiques pour la basse
            patterns = [
                [1.0],           # Noire
                [0.5, 0.5],      # Deux croches
                [0.75, 0.25],    # Croche pointée + double
                [0.25, 0.25, 0.5],  # Syncopé
            ]
            pattern = patterns[token % len(patterns)]
            
            for duration_ratio in pattern:
                note_duration = duration_ratio * beat_duration
                
                note = pretty_midi.Note(
                    velocity=int(velocity),
                    pitch=int(pitch),
                    start=current_time,
                    end=current_time + note_duration * 0.9  # Staccato
                )
                instrument.notes.append(note)
                
                current_time += note_duration
                
                if current_time >= duration:
                    break
        
        midi.instruments.append(instrument)
        return midi
    
    def generate_batch(self, n_melodies=50, n_chords=50, n_basslines=50):
        """Génère un batch complet de fichiers MIDI"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"\n🎼 Génération de {n_melodies + n_chords + n_basslines} fichiers MIDI de 60 secondes...")
        
        # Générer les mélodies
        print(f"\n🎵 Génération de {n_melodies} mélodies...")
        for i in tqdm(range(n_melodies), desc="Mélodies"):
            tempo = np.random.randint(110, 130)
            temperature = np.random.uniform(0.8, 1.1)
            
            midi = self.generate_melody(duration=60, tempo=tempo, temperature=temperature)
            filename = f"melody_{timestamp}_{i+1:03d}_{tempo}bpm.mid"
            filepath = os.path.join('./output_midi/melodies', filename)
            midi.write(filepath)
        
        # Générer les accords
        print(f"\n🎹 Génération de {n_chords} progressions d'accords...")
        for i in tqdm(range(n_chords), desc="Accords"):
            tempo = np.random.randint(100, 125)
            temperature = np.random.uniform(0.7, 0.9)
            
            midi = self.generate_chords(duration=60, tempo=tempo, temperature=temperature)
            filename = f"chords_{timestamp}_{i+1:03d}_{tempo}bpm.mid"
            filepath = os.path.join('./output_midi/chords', filename)
            midi.write(filepath)
        
        # Générer les basslines
        print(f"\n🎸 Génération de {n_basslines} lignes de basse...")
        for i in tqdm(range(n_basslines), desc="Basslines"):
            tempo = np.random.randint(115, 130)
            temperature = np.random.uniform(0.6, 0.8)
            
            midi = self.generate_bassline(duration=60, tempo=tempo, temperature=temperature)
            filename = f"bassline_{timestamp}_{i+1:03d}_{tempo}bpm.mid"
            filepath = os.path.join('./output_midi/basslines', filename)
            midi.write(filepath)
        
        print(f"\n✅ Génération terminée!")
        print(f"📁 Fichiers sauvegardés dans:")
        print(f"   - ./output_midi/melodies/")
        print(f"   - ./output_midi/chords/")
        print(f"   - ./output_midi/basslines/")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Générateur MIDI avancé")
    parser.add_argument('--checkpoint', type=str, 
                       default='./checkpoints/20250802_191918/best_model.pt',
                       help='Chemin vers le checkpoint')
    parser.add_argument('--n_melodies', type=int, default=50,
                       help='Nombre de mélodies à générer')
    parser.add_argument('--n_chords', type=int, default=50,
                       help='Nombre de progressions d\'accords')
    parser.add_argument('--n_basslines', type=int, default=50,
                       help='Nombre de lignes de basse')
    
    args = parser.parse_args()
    
    generator = AdvancedMidiGenerator(args.checkpoint)
    generator.generate_batch(
        n_melodies=args.n_melodies,
        n_chords=args.n_chords,
        n_basslines=args.n_basslines
    )

if __name__ == "__main__":
    main()