#!/usr/bin/env python3
"""
MIDI Analyzer - Analyse des fichiers MIDI pour comprendre leur structure
"""

import os
import pretty_midi
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import json

class MidiAnalyzer:
    def __init__(self, midi_dir="./midi_files"):
        self.midi_dir = midi_dir
        self.stats = defaultdict(list)
        
    def analyze_file(self, filepath):
        """Analyse un fichier MIDI individuel"""
        try:
            midi = pretty_midi.PrettyMIDI(filepath)
            
            # Extraire les métadonnées
            tempo = midi.estimate_tempo()
            duration = midi.get_end_time()
            
            # Analyser les notes
            notes = []
            for instrument in midi.instruments:
                if not instrument.is_drum:
                    for note in instrument.notes:
                        notes.append({
                            'pitch': note.pitch,
                            'velocity': note.velocity,
                            'start': note.start,
                            'end': note.end,
                            'duration': note.end - note.start
                        })
            
            return {
                'filename': os.path.basename(filepath),
                'tempo': tempo,
                'duration': duration,
                'n_notes': len(notes),
                'n_instruments': len(midi.instruments),
                'time_signature': midi.time_signature_changes,
                'key_signature': midi.key_signature_changes,
                'notes': notes
            }
            
        except Exception as e:
            print(f"Erreur avec {filepath}: {e}")
            return None
    
    def analyze_all(self):
        """Analyse tous les fichiers MIDI du dossier"""
        midi_files = []
        
        # Collecter tous les fichiers .mid
        for root, dirs, files in os.walk(self.midi_dir):
            for file in files:
                if file.endswith('.mid'):
                    midi_files.append(os.path.join(root, file))
        
        print(f"Trouvé {len(midi_files)} fichiers MIDI")
        
        # Analyser chaque fichier
        results = []
        for filepath in tqdm(midi_files, desc="Analyse des fichiers"):
            result = self.analyze_file(filepath)
            if result:
                results.append(result)
                
                # Collecter les stats globales
                self.stats['tempos'].append(result['tempo'])
                self.stats['durations'].append(result['duration'])
                self.stats['n_notes'].append(result['n_notes'])
                
        # Calculer les stats
        print("\n📊 Statistiques globales:")
        print(f"- Tempo moyen: {np.mean(self.stats['tempos']):.1f} BPM")
        print(f"- Durée moyenne: {np.mean(self.stats['durations']):.1f} secondes")
        print(f"- Notes par fichier: {np.mean(self.stats['n_notes']):.1f}")
        print(f"- Plage de tempos: {min(self.stats['tempos']):.0f}-{max(self.stats['tempos']):.0f} BPM")
        
        return results

if __name__ == "__main__":
    analyzer = MidiAnalyzer()
    results = analyzer.analyze_all()
    
    # Sauvegarder les résultats
    with open('midi_analysis.json', 'w') as f:
        json.dump({
            'stats': {k: list(v) for k, v in analyzer.stats.items()},
            'n_files': len(results)
        }, f, indent=2)