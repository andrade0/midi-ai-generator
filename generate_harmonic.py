#!/usr/bin/env python3
"""
Générateur harmonique de MIDI - Tous les fichiers dans la même tonalité
"""

import os
import torch
import numpy as np
from datetime import datetime
import pretty_midi
from model import MusicTransformer
from tqdm import tqdm

class HarmonicMidiGenerator:
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
        os.makedirs('./output_midi/harmonic/melodies', exist_ok=True)
        os.makedirs('./output_midi/harmonic/chords', exist_ok=True)
        os.makedirs('./output_midi/harmonic/basslines', exist_ok=True)
        
        # Définir les gammes disponibles
        self.scales = {
            'C_major': {'root': 60, 'intervals': [0, 2, 4, 5, 7, 9, 11], 'name': 'C Major'},
            'A_minor': {'root': 57, 'intervals': [0, 2, 3, 5, 7, 8, 10], 'name': 'A Minor'},
            'G_major': {'root': 67, 'intervals': [0, 2, 4, 5, 7, 9, 11], 'name': 'G Major'},
            'E_minor': {'root': 64, 'intervals': [0, 2, 3, 5, 7, 8, 10], 'name': 'E Minor'},
            'D_major': {'root': 62, 'intervals': [0, 2, 4, 5, 7, 9, 11], 'name': 'D Major'},
            'B_minor': {'root': 59, 'intervals': [0, 2, 3, 5, 7, 8, 10], 'name': 'B Minor'},
            'F_major': {'root': 65, 'intervals': [0, 2, 4, 5, 7, 9, 11], 'name': 'F Major'},
            'D_minor': {'root': 62, 'intervals': [0, 2, 3, 5, 7, 8, 10], 'name': 'D Minor'},
        }
        
        # Progressions d'accords harmoniques par tonalité
        self.chord_progressions = {
            'major': [
                [0, 4, 7],      # I (Majeur)
                [2, 5, 9],      # ii (mineur)
                [4, 7, 11],     # iii (mineur)
                [5, 9, 0],      # IV (Majeur)
                [7, 11, 2],     # V (Majeur)
                [9, 0, 4],      # vi (mineur)
            ],
            'minor': [
                [0, 3, 7],      # i (mineur)
                [2, 5, 8],      # ii° (diminué)
                [3, 7, 10],     # III (Majeur)
                [5, 8, 0],      # iv (mineur)
                [7, 10, 2],     # v (mineur)
                [8, 0, 3],      # VI (Majeur)
                [10, 2, 5],     # VII (Majeur)
            ]
        }
    
    def get_note_in_scale(self, scale_info, octave_offset=0):
        """Retourne une note dans la gamme spécifiée"""
        root = scale_info['root']
        intervals = scale_info['intervals']
        
        # Choisir un intervalle aléatoire
        interval = np.random.choice(intervals)
        
        # Ajouter l'offset d'octave
        note = root + interval + (octave_offset * 12)
        
        # S'assurer que la note est dans la plage MIDI valide
        note = max(0, min(127, note))
        
        return int(note)
    
    def generate_melody_harmonic(self, scale_key, duration=60, tempo=120, temperature=0.9):
        """Génère une mélodie dans la tonalité spécifiée"""
        scale_info = self.scales[scale_key]
        length = int(duration * 4)
        
        # Générer les tokens de base
        start_tokens = torch.randint(60, 84, (1, 8), device=self.device)
        
        with torch.no_grad():
            generated = self.model.generate(
                start_tokens,
                max_length=length,
                temperature=temperature,
                top_k=40,
                top_p=0.9
            )
        
        # Convertir en MIDI en respectant la gamme
        tokens = generated[0].cpu().numpy()
        midi = self._create_melody_midi_harmonic(tokens, scale_info, tempo, duration)
        
        return midi
    
    def generate_chords_harmonic(self, scale_key, duration=60, tempo=120, temperature=0.8):
        """Génère des accords dans la tonalité spécifiée"""
        scale_info = self.scales[scale_key]
        is_major = 'major' in scale_key.lower()
        length = int(duration * 2)
        
        # Générer les tokens
        start_tokens = torch.randint(48, 72, (1, 8), device=self.device)
        
        with torch.no_grad():
            generated = self.model.generate(
                start_tokens,
                max_length=length,
                temperature=temperature,
                top_k=30,
                top_p=0.85
            )
        
        tokens = generated[0].cpu().numpy()
        midi = self._create_chords_midi_harmonic(tokens, scale_info, is_major, tempo, duration)
        
        return midi
    
    def generate_bassline_harmonic(self, scale_key, duration=60, tempo=120, temperature=0.7):
        """Génère une ligne de basse dans la tonalité spécifiée"""
        scale_info = self.scales[scale_key]
        length = int(duration * 3)
        
        # Générer les tokens
        start_tokens = torch.randint(36, 60, (1, 8), device=self.device)
        
        with torch.no_grad():
            generated = self.model.generate(
                start_tokens,
                max_length=length,
                temperature=temperature,
                top_k=25,
                top_p=0.8
            )
        
        tokens = generated[0].cpu().numpy()
        midi = self._create_bassline_midi_harmonic(tokens, scale_info, tempo, duration)
        
        return midi
    
    def _create_melody_midi_harmonic(self, tokens, scale_info, tempo, duration):
        """Crée une mélodie en respectant la gamme"""
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        instrument = pretty_midi.Instrument(program=0)  # Piano
        
        current_time = 0.0
        beat_duration = 60.0 / tempo
        
        for i, token in enumerate(tokens):
            if current_time >= duration:
                break
            
            # Note dans la gamme (octaves 4-6)
            octave_offset = np.random.choice([0, 1, 2])
            pitch = self.get_note_in_scale(scale_info, octave_offset)
            
            velocity = 70 + (token % 40)
            
            # Durées variées
            durations = [0.25, 0.5, 0.75, 1.0]
            note_duration = durations[token % len(durations)] * beat_duration
            
            note = pretty_midi.Note(
                velocity=int(velocity),
                pitch=pitch,
                start=current_time,
                end=current_time + note_duration
            )
            instrument.notes.append(note)
            
            # Avancer avec possibilité de silences
            if token % 10 < 2:
                current_time += note_duration * 0.5
            else:
                current_time += note_duration
        
        midi.instruments.append(instrument)
        return midi
    
    def _create_chords_midi_harmonic(self, tokens, scale_info, is_major, tempo, duration):
        """Crée des accords harmoniques dans la gamme"""
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        instrument = pretty_midi.Instrument(program=0)  # Piano
        
        current_time = 0.0
        beat_duration = 60.0 / tempo
        
        # Sélectionner les progressions selon le mode
        chord_type = 'major' if is_major else 'minor'
        progressions = self.chord_progressions[chord_type]
        
        # Progression commune I-V-vi-IV ou i-VI-III-VII
        if is_major:
            chord_sequence = [0, 4, 5, 3]  # I-V-vi-IV
        else:
            chord_sequence = [0, 5, 2, 6]  # i-VI-III-VII
        
        chord_idx = 0
        
        for i, token in enumerate(tokens):
            if current_time >= duration:
                break
            
            # Sélectionner l'accord dans la progression
            chord_num = chord_sequence[chord_idx % len(chord_sequence)]
            if chord_num < len(progressions):
                chord_intervals = progressions[chord_num]
            else:
                chord_intervals = progressions[0]
            
            root = scale_info['root']
            velocity = 60 + (token % 30)
            chord_duration = (1 + (token % 4)) * beat_duration
            
            # Créer les notes de l'accord
            for interval in chord_intervals:
                note_pitch = root + scale_info['intervals'][interval % len(scale_info['intervals'])]
                
                note = pretty_midi.Note(
                    velocity=int(velocity),
                    pitch=int(note_pitch),
                    start=current_time,
                    end=current_time + chord_duration * 0.95
                )
                instrument.notes.append(note)
            
            current_time += chord_duration
            chord_idx += 1
        
        midi.instruments.append(instrument)
        return midi
    
    def _create_bassline_midi_harmonic(self, tokens, scale_info, tempo, duration):
        """Crée une ligne de basse dans la gamme"""
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        instrument = pretty_midi.Instrument(program=33)  # Basse électrique
        
        current_time = 0.0
        beat_duration = 60.0 / tempo
        
        # Notes importantes de la gamme pour la basse
        important_degrees = [0, 4, 3, 2]  # Tonique, quinte, quarte, tierce
        
        for i, token in enumerate(tokens):
            if current_time >= duration:
                break
            
            # Choisir une note importante de la gamme
            degree = important_degrees[token % len(important_degrees)]
            interval = scale_info['intervals'][degree % len(scale_info['intervals'])]
            
            # Basse dans les octaves 2-3
            pitch = scale_info['root'] + interval - 24  # 2 octaves plus bas
            
            velocity = 80 + (token % 40)
            
            # Patterns rythmiques
            patterns = [
                [1.0],
                [0.5, 0.5],
                [0.75, 0.25],
                [0.25, 0.25, 0.5],
            ]
            pattern = patterns[token % len(patterns)]
            
            for duration_ratio in pattern:
                note_duration = duration_ratio * beat_duration
                
                note = pretty_midi.Note(
                    velocity=int(velocity),
                    pitch=int(pitch),
                    start=current_time,
                    end=current_time + note_duration * 0.9
                )
                instrument.notes.append(note)
                
                current_time += note_duration
                
                if current_time >= duration:
                    break
        
        midi.instruments.append(instrument)
        return midi
    
    def generate_harmonic_batch(self, scale_key='C_major', n_melodies=50, n_chords=50, n_basslines=50):
        """Génère un batch complet dans la même tonalité"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scale_name = self.scales[scale_key]['name']
        
        print(f"\n🎼 Génération de {n_melodies + n_chords + n_basslines} fichiers MIDI")
        print(f"🎵 Tonalité : {scale_name}")
        print(f"⏱️  Durée : 60 secondes par fichier")
        
        # Générer les mélodies
        print(f"\n🎵 Génération de {n_melodies} mélodies en {scale_name}...")
        for i in tqdm(range(n_melodies), desc=f"Mélodies ({scale_name})"):
            tempo = np.random.randint(110, 130)
            temperature = np.random.uniform(0.8, 1.1)
            
            midi = self.generate_melody_harmonic(scale_key, duration=60, tempo=tempo, temperature=temperature)
            filename = f"melody_{scale_key}_{timestamp}_{i+1:03d}_{tempo}bpm.mid"
            filepath = os.path.join('./output_midi/harmonic/melodies', filename)
            midi.write(filepath)
        
        # Générer les accords
        print(f"\n🎹 Génération de {n_chords} progressions d'accords en {scale_name}...")
        for i in tqdm(range(n_chords), desc=f"Accords ({scale_name})"):
            tempo = np.random.randint(100, 125)
            temperature = np.random.uniform(0.7, 0.9)
            
            midi = self.generate_chords_harmonic(scale_key, duration=60, tempo=tempo, temperature=temperature)
            filename = f"chords_{scale_key}_{timestamp}_{i+1:03d}_{tempo}bpm.mid"
            filepath = os.path.join('./output_midi/harmonic/chords', filename)
            midi.write(filepath)
        
        # Générer les basslines
        print(f"\n🎸 Génération de {n_basslines} lignes de basse en {scale_name}...")
        for i in tqdm(range(n_basslines), desc=f"Basslines ({scale_name})"):
            tempo = np.random.randint(115, 130)
            temperature = np.random.uniform(0.6, 0.8)
            
            midi = self.generate_bassline_harmonic(scale_key, duration=60, tempo=tempo, temperature=temperature)
            filename = f"bassline_{scale_key}_{timestamp}_{i+1:03d}_{tempo}bpm.mid"
            filepath = os.path.join('./output_midi/harmonic/basslines', filename)
            midi.write(filepath)
        
        print(f"\n✅ Génération terminée!")
        print(f"🎵 Tous les fichiers sont en {scale_name}")
        print(f"📁 Fichiers sauvegardés dans:")
        print(f"   - ./output_midi/harmonic/melodies/")
        print(f"   - ./output_midi/harmonic/chords/")
        print(f"   - ./output_midi/harmonic/basslines/")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Générateur MIDI harmonique")
    parser.add_argument('--checkpoint', type=str, 
                       default='./checkpoints/20250802_191918/best_model.pt',
                       help='Chemin vers le checkpoint')
    parser.add_argument('--scale', type=str, default='C_major',
                       choices=['C_major', 'A_minor', 'G_major', 'E_minor', 
                               'D_major', 'B_minor', 'F_major', 'D_minor'],
                       help='Tonalité pour tous les fichiers')
    parser.add_argument('--n_melodies', type=int, default=50,
                       help='Nombre de mélodies à générer')
    parser.add_argument('--n_chords', type=int, default=50,
                       help='Nombre de progressions d\'accords')
    parser.add_argument('--n_basslines', type=int, default=50,
                       help='Nombre de lignes de basse')
    
    args = parser.parse_args()
    
    generator = HarmonicMidiGenerator(args.checkpoint)
    generator.generate_harmonic_batch(
        scale_key=args.scale,
        n_melodies=args.n_melodies,
        n_chords=args.n_chords,
        n_basslines=args.n_basslines
    )

if __name__ == "__main__":
    main()